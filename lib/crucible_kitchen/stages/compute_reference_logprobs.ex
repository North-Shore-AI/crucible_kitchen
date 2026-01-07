defmodule CrucibleKitchen.Stages.ComputeReferenceLogprobs do
  @moduledoc """
  Computes log probabilities from frozen reference model for DPO training.

  In DPO, the reference model provides a baseline for the KL divergence
  constraint. This stage computes logprobs for both chosen and rejected
  responses using the reference model (usually the initial model before training).

  ## Context Requirements

  **Input:**
  - State: `:preference_batch` - Current batch of preference pairs
  - State: `:reference_session` - Reference model session (or uses main session)
  - Config: `:reference_model` - Reference model name (optional, defaults to base model)

  **Output:**
  - State: `:ref_chosen_logprobs` - Reference logprobs for chosen responses
  - State: `:ref_rejected_logprobs` - Reference logprobs for rejected responses

  ## Example

      stage(:compute_reference_logprobs, ComputeReferenceLogprobs)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.SamplingClient
  alias CrucibleTrain.Renderers.{Renderer, TrainOnWhat, Types}
  alias CrucibleTrain.Supervised.Common
  alias CrucibleTrain.Types.{ModelInput, TensorData}

  require Logger

  @impl true
  def name, do: :compute_reference_logprobs

  @impl true
  def execute(context) do
    batch = Context.get_state(context, :preference_batch)
    tokenizer = Context.get_state(context, :tokenizer)
    model = Context.get_config(context, :model)
    max_length = Context.get_config(context, :max_length, 8192)

    with {:ok, datums} <- build_preference_datums(batch, tokenizer, model, max_length),
         {:ok, context, ref_session} <- ensure_reference_session(context),
         {:ok, ref_logprobs} <- compute_logprobs(context, ref_session, datums) do
      {ref_chosen, ref_rejected} = split_chosen_rejected(ref_logprobs)

      context
      |> Context.put_state(:preference_datums, datums)
      |> Context.put_state(:ref_chosen_logprobs, ref_chosen)
      |> Context.put_state(:ref_rejected_logprobs, ref_rejected)
      |> then(&{:ok, &1})
    end
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :preference_batch) do
      nil -> {:error, "preference_batch is required in state"}
      _ -> :ok
    end
  end

  defp ensure_reference_session(context) do
    case Context.get_state(context, :reference_session) do
      nil ->
        ports = get_train_ports(context)

        config = %{
          model:
            Context.get_config(context, :reference_model) || Context.get_config(context, :model),
          base_model: Context.get_config(context, :reference_model),
          model_path: Context.get_config(context, :reference_model_path),
          checkpoint_path: Context.get_config(context, :reference_checkpoint_path)
        }

        case SamplingClient.start_session(ports, config) do
          {:ok, session} ->
            context = Context.put_state(context, :reference_session, session)
            {:ok, context, session}

          {:error, reason} ->
            {:error, {:reference_session_failed, reason}}
        end

      session ->
        {:ok, context, session}
    end
  end

  defp compute_logprobs(context, session, datums) do
    Logger.debug("Computing reference logprobs for #{length(datums)} datums")
    ports = get_train_ports(context)

    logprob_futures =
      Enum.map(datums, fn datum ->
        model_input = full_sequence_input(datum)
        SamplingClient.compute_logprobs(ports, session, model_input)
      end)

    logprobs =
      Enum.map(logprob_futures, fn
        {:ok, future} ->
          case SamplingClient.await(ports, future) do
            {:ok, values} -> drop_prompt_logprob(values)
            {:error, reason} -> {:error, reason}
          end

        {:error, reason} ->
          {:error, reason}
      end)

    case Enum.find(logprobs, &match?({:error, _}, &1)) do
      {:error, reason} ->
        Logger.error("Failed to compute reference logprobs: #{inspect(reason)}")
        {:error, {:ref_logprobs_failed, reason}}

      nil ->
        {:ok, logprobs}
    end
  end

  defp build_preference_datums(_batch, nil, _model, _max_length),
    do: {:error, :missing_tokenizer}

  defp build_preference_datums(batch, tokenizer, model, max_length) do
    renderer_module = renderer_for_model(model)

    with {:ok, renderer_state} <- renderer_module.init(tokenizer: tokenizer) do
      datums =
        Enum.flat_map(batch, fn pair ->
          with {:ok, chosen} <-
                 build_datum(pair, :chosen, renderer_module, renderer_state, max_length),
               {:ok, rejected} <-
                 build_datum(pair, :rejected, renderer_module, renderer_state, max_length) do
            [chosen, rejected]
          else
            _ -> []
          end
        end)

      {:ok, datums}
    end
  end

  defp build_datum(pair, key, renderer_module, renderer_state, max_length) do
    prompt = fetch_pair_value(pair, :prompt)
    response = fetch_pair_value(pair, key)

    if is_binary(prompt) and is_binary(response) do
      messages = [
        Types.message("user", prompt),
        Types.message("assistant", response)
      ]

      train_on = TrainOnWhat.last_assistant_message()

      {model_input, weights} =
        Renderer.build_supervised_example(renderer_module, messages, train_on, renderer_state)

      datum = Common.datum_from_model_input_weights(model_input, weights, max_length)
      {:ok, datum}
    else
      {:error, :invalid_pair}
    end
  end

  defp fetch_pair_value(pair, key) when is_map(pair) do
    Map.get(pair, key) || Map.get(pair, Atom.to_string(key))
  end

  defp full_sequence_input(datum) do
    target_tokens =
      datum.loss_fn_inputs["target_tokens"]
      |> TensorData.to_list()

    case target_tokens do
      [] -> datum.model_input
      _ -> ModelInput.append_int(datum.model_input, List.last(target_tokens))
    end
  end

  defp drop_prompt_logprob([_ | rest]), do: rest
  defp drop_prompt_logprob(other), do: other

  defp split_chosen_rejected(logprobs) do
    chosen = Enum.take_every(logprobs, 2)
    rejected = logprobs |> Enum.drop(1) |> Enum.take_every(2)
    {chosen, rejected}
  end

  defp renderer_for_model(model_name) when is_binary(model_name) do
    cond do
      String.contains?(model_name, "Llama-3") -> CrucibleTrain.Renderers.Llama3
      String.contains?(model_name, "Qwen") -> CrucibleTrain.Renderers.Qwen3
      String.contains?(model_name, "DeepSeek") -> CrucibleTrain.Renderers.DeepSeekV3
      true -> CrucibleTrain.Renderers.RoleColon
    end
  end

  defp renderer_for_model(_), do: CrucibleTrain.Renderers.RoleColon
end
