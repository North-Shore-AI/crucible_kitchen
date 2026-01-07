defmodule CrucibleKitchen.Stages.DistillationStagesTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context

  alias CrucibleKitchen.TestSupport.{
    DistillationSamplingClient,
    DistillationTrainingClient
  }

  alias CrucibleKitchen.Stages.{
    BuildDistillationDataset,
    BuildDistillDatums,
    CleanupTeacher,
    DistillationForwardBackward,
    GetDistillationBatch,
    InitTeacher,
    LogDistillationMetrics,
    TeacherInference
  }

  alias CrucibleTrain.Types.TensorData

  describe "InitTeacher" do
    test "starts a sampling session and stores teacher_session" do
      Process.put(:distill_test_pid, self())

      context = build_context(%{teacher_model: "teacher-model"})
      assert {:ok, result} = InitTeacher.execute(context)
      assert result.state.teacher_session != nil
    end
  end

  describe "BuildDistillationDataset" do
    test "builds dataset from raw samples" do
      dataset_handle = %{
        items: [
          %{"prompt" => "Q1"},
          %{"prompt" => "Q2"}
        ]
      }

      context =
        build_context(%{batch_size: 1})
        |> Context.put_state(:dataset_handle, dataset_handle)

      assert {:ok, result} = BuildDistillationDataset.execute(context)
      assert result.state.distillation_dataset.num_batches == 2
    end

    test "builds dataset from input.problem samples" do
      dataset_handle = %{
        items: [
          %{"input" => %{"problem" => "Q1"}}
        ]
      }

      context =
        build_context(%{batch_size: 1})
        |> Context.put_state(:dataset_handle, dataset_handle)

      assert {:ok, result} = BuildDistillationDataset.execute(context)

      [messages] = result.state.distillation_dataset.samples
      [message] = messages

      assert message.content == "Q1"
      assert message.role == "user"
    end
  end

  describe "GetDistillationBatch" do
    test "returns the expected batch" do
      dataset = %{
        samples: [
          [%{role: "user", content: "Q1"}],
          [%{role: "user", content: "Q2"}]
        ],
        batch_size: 1,
        num_batches: 2
      }

      context =
        build_context(%{})
        |> Context.put_state(:distillation_dataset, dataset)
        |> Context.put_state(:dist_batches_index, 1)

      assert {:ok, result} = GetDistillationBatch.execute(context)
      assert result.state.batch_index == 1
      assert result.state.distillation_batch == [[%{role: "user", content: "Q2"}]]
    end
  end

  describe "TeacherInference" do
    test "samples teacher responses for each prompt" do
      Process.put(:distill_test_pid, self())

      tokenizer = %{
        encode: fn text, _opts -> :binary.bin_to_list(text) end,
        decode: fn tokens -> List.to_string(tokens) end
      }

      context =
        build_context(%{model: "test-model", teacher_model: "teacher-model", max_tokens: 4})
        |> Context.put_state(:tokenizer, tokenizer)
        |> Context.put_state(:teacher_session, %{id: :teacher})
        |> Context.put_state(:distillation_batch, [
          [%{role: "user", content: "Q1"}],
          [%{role: "user", content: "Q2"}]
        ])

      assert {:ok, result} = TeacherInference.execute(context)
      assert length(result.state.teacher_responses) == 2

      assert_receive {:teacher_sample, _model_input, _params}
      assert_receive {:teacher_sample, _model_input, _params}
    end
  end

  describe "BuildDistillDatums" do
    test "builds supervised datums from prompts and teacher responses" do
      tokenizer = %{
        encode: fn text, _opts -> :binary.bin_to_list(text) end,
        decode: fn tokens -> List.to_string(tokens) end
      }

      context =
        build_context(%{model: "test-model", max_length: 128})
        |> Context.put_state(:tokenizer, tokenizer)
        |> Context.put_state(:distillation_batch, [
          [%{role: "user", content: "Q1"}]
        ])
        |> Context.put_state(:teacher_responses, [
          %{role: "assistant", content: "A1"}
        ])

      assert {:ok, result} = BuildDistillDatums.execute(context)
      assert length(result.state.distill_datums) == 1

      [datum] = result.state.distill_datums
      assert TensorData.to_list(datum.loss_fn_inputs["target_tokens"]) != []
    end
  end

  describe "DistillationForwardBackward" do
    test "invokes training_client forward_backward" do
      Process.put(:distill_test_pid, self())

      datum = %{
        model_input: %{chunks: []},
        loss_fn_inputs: %{
          "target_tokens" => TensorData.from_list([1], :int64),
          "weights" => TensorData.from_list([1.0], :float32)
        }
      }

      context =
        build_context(%{})
        |> Context.put_state(:session, %{id: :training})
        |> Context.put_state(:distill_datums, [datum])

      assert {:ok, result} = DistillationForwardBackward.execute(context)
      assert result.state.distillation_future != nil

      assert_receive {:distill_forward_backward, 1, opts}
      assert opts[:loss_fn] == :cross_entropy
    end
  end

  describe "LogDistillationMetrics" do
    test "records metrics and increments step" do
      context =
        build_context(%{})
        |> Context.put_state(:distillation_metrics, %{"loss" => 0.4})
        |> Context.put_state(:global_step, 3)

      assert {:ok, result} = LogDistillationMetrics.execute(context)
      assert result.state.global_step == 4
      assert Enum.any?(result.metrics, &(&1.name == :distill_loss))
    end
  end

  describe "CleanupTeacher" do
    test "clears teacher session" do
      context =
        build_context(%{})
        |> Context.put_state(:teacher_session, %{id: :teacher})

      assert {:ok, result} = CleanupTeacher.execute(context)
      assert result.state.teacher_session == nil
    end
  end

  defp build_context(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: DistillationTrainingClient,
        sampling_client: DistillationSamplingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end
end
