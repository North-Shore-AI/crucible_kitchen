defmodule CrucibleKitchen.Stages.DPOStagesTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.TestSupport.{DPOTestSamplingClient, DPOTestTrainingClient}
  alias CrucibleTrain.Ports.TrainingClient
  alias CrucibleTrain.Supervised.Common, as: SupervisedCommon
  alias CrucibleTrain.Types.{ModelInput, TensorData}

  alias CrucibleKitchen.Stages.{
    BuildPreferenceDataset,
    ComputeReferenceLogprobs,
    DPOForwardBackward,
    GetPreferenceBatch,
    LogDPOMetrics
  }

  describe "BuildPreferenceDataset" do
    test "name returns :build_preference_dataset" do
      assert BuildPreferenceDataset.name() == :build_preference_dataset
    end

    test "builds preference dataset from raw data" do
      raw_dataset = [
        %{"prompt" => "Question 1", "chosen" => "Good answer", "rejected" => "Bad answer"},
        %{"prompt" => "Question 2", "chosen" => "Better", "rejected" => "Worse"}
      ]

      context =
        build_context(%{batch_size: 1})
        |> Context.put_state(:raw_dataset, raw_dataset)

      assert {:ok, result} = BuildPreferenceDataset.execute(context)
      assert result.state.preference_dataset != nil
      assert result.state.preference_dataset.num_pairs == 2
      assert result.state.preference_dataset.num_batches == 2
    end

    test "filters invalid pairs" do
      raw_dataset = [
        %{"prompt" => "Valid", "chosen" => "Yes", "rejected" => "No"},
        # Invalid - empty prompt
        %{"prompt" => "", "chosen" => "Yes", "rejected" => "No"},
        # Invalid - empty chosen
        %{"prompt" => "Valid", "chosen" => "", "rejected" => "No"}
      ]

      context =
        build_context(%{batch_size: 10})
        |> Context.put_state(:raw_dataset, raw_dataset)

      assert {:ok, result} = BuildPreferenceDataset.execute(context)
      assert result.state.num_preference_pairs == 1
    end

    test "validation requires raw_dataset" do
      context = build_context(%{})
      assert {:error, _} = BuildPreferenceDataset.validate(context)
    end
  end

  describe "GetPreferenceBatch" do
    test "name returns :get_preference_batch" do
      assert GetPreferenceBatch.name() == :get_preference_batch
    end

    test "gets batch from preference dataset" do
      pairs = [
        %{prompt: "Q1", chosen: "A1", rejected: "B1"},
        %{prompt: "Q2", chosen: "A2", rejected: "B2"}
      ]

      context =
        build_context(%{})
        |> Context.put_state(:preference_dataset, %{
          pairs: pairs,
          batch_size: 1,
          num_batches: 2,
          num_pairs: 2
        })
        |> Context.put_state(:pref_batches_index, 0)

      assert {:ok, result} = GetPreferenceBatch.execute(context)
      assert length(result.state.preference_batch) == 1
      assert result.state.batch_index == 0
    end
  end

  describe "ComputeReferenceLogprobs" do
    test "name returns :compute_reference_logprobs" do
      assert ComputeReferenceLogprobs.name() == :compute_reference_logprobs
    end

    test "builds datums and computes reference logprobs via sampling client" do
      Process.put(:dpo_test_pid, self())

      tokenizer = %{
        encode: fn text, _opts -> :binary.bin_to_list(text) end,
        decode: fn tokens -> List.to_string(tokens) end
      }

      batch = [%{prompt: "Q", chosen: "A", rejected: "B"}]

      context =
        build_context(%{model: "test-model", max_length: 128})
        |> Context.put_state(:tokenizer, tokenizer)
        |> Context.put_state(:preference_batch, batch)

      assert {:ok, result} = ComputeReferenceLogprobs.execute(context)
      assert length(result.state.preference_datums) == 2
      assert length(result.state.ref_chosen_logprobs) == 1
      assert length(result.state.ref_rejected_logprobs) == 1

      [chosen_datum, _rejected_datum] = result.state.preference_datums

      chosen_target_len =
        chosen_datum.loss_fn_inputs["target_tokens"]
        |> TensorData.to_list()
        |> length()

      assert length(hd(result.state.ref_chosen_logprobs)) == chosen_target_len

      assert_receive {:compute_logprobs, _model_input}
      assert_receive {:compute_logprobs, _model_input}
    end

    test "validation requires preference_batch" do
      context = build_context(%{})
      assert {:error, _} = ComputeReferenceLogprobs.validate(context)
    end
  end

  describe "DPOForwardBackward" do
    test "name returns :dpo_forward_backward" do
      assert DPOForwardBackward.name() == :dpo_forward_backward
    end

    test "uses forward_backward_custom with DPO loss" do
      Process.put(:dpo_test_pid, self())

      {datums, ref_chosen, ref_rejected} = build_dpo_fixture()

      context =
        build_context(%{dpo_beta: 0.1})
        |> Context.put_state(:preference_datums, datums)
        |> Context.put_state(:ref_chosen_logprobs, ref_chosen)
        |> Context.put_state(:ref_rejected_logprobs, ref_rejected)

      assert {:ok, result} = DPOForwardBackward.execute(context)
      assert result.state.dpo_future != nil

      assert_receive {:forward_backward_custom, 2}

      ports = Context.get_train_ports(result)
      assert {:ok, fb_result} = TrainingClient.await(ports, result.state.dpo_future)
      metrics = fb_result.metrics

      assert_in_delta(metrics.loss, 0.616, 0.01)
      assert metrics.accuracy == 1.0
      assert_in_delta(metrics.margin, 0.16, 0.01)
    end

    test "validation requires reference logprobs" do
      context =
        build_context(%{})
        |> Context.put_state(:preference_datums, [])

      assert {:error, msg} = DPOForwardBackward.validate(context)
      assert String.contains?(msg, "ref_chosen_logprobs")
    end
  end

  describe "LogDPOMetrics" do
    test "name returns :log_dpo_metrics" do
      assert LogDPOMetrics.name() == :log_dpo_metrics
    end

    test "logs metrics and increments step" do
      context =
        build_context(%{})
        |> Context.put_state(:dpo_metrics, %{
          loss: 0.5,
          accuracy: 0.7,
          chosen_reward: 1.0,
          rejected_reward: -1.0,
          margin: 2.0
        })
        |> Context.put_state(:global_step, 10)

      assert {:ok, result} = LogDPOMetrics.execute(context)
      assert result.state.global_step == 11

      # Check metrics recorded
      assert Enum.any?(result.metrics, &(&1.name == :dpo_loss))
      assert Enum.any?(result.metrics, &(&1.name == :dpo_accuracy))
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "log-dpo-metrics-test",
        [:crucible_kitchen, :dpo, :step],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context =
        build_context(%{})
        |> Context.put_state(:dpo_metrics, %{loss: 0.3, accuracy: 0.8})
        |> Context.put_state(:global_step, 5)

      {:ok, _} = LogDPOMetrics.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :dpo, :step], measurements, _}
      assert measurements.loss == 0.3

      :telemetry.detach("log-dpo-metrics-test")
    end
  end

  defp build_context(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: DPOTestTrainingClient,
        sampling_client: DPOTestSamplingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end

  defp build_dpo_fixture do
    model_input = ModelInput.from_ints([1, 2, 3, 4])
    weights = [0.0, 1.0, 1.0, 1.0]
    datum = SupervisedCommon.datum_from_model_input_weights(model_input, weights, 128)

    datums = [datum, datum]

    ref_chosen = [[-0.7, -0.7, -0.6]]
    ref_rejected = [[-0.3, -0.3, -0.4]]

    {datums, ref_chosen, ref_rejected}
  end
end
