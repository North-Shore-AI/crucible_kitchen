defmodule CrucibleKitchen.Stages.TeacherInference do
  @moduledoc """
  Samples teacher responses for each prompt batch.

  ## Context Requirements

  **Input:**
  - State: `:distillation_batch`
  - State: `:teacher_session`
  - State: `:tokenizer`

  **Config:**
  - `:teacher_model` (optional)
  - `:max_tokens` (default: 256)
  - `:temperature` (default: 1.0)

  **Output:**
  - State: `:teacher_responses`
  - State: `:teacher_metrics`
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.SamplingClient
  alias CrucibleTrain.Renderers.{Helpers, Renderer, Types}

  require Logger

  @impl true
  def name, do: :teacher_inference

  @impl true
  def execute(context) do
    batch = Context.get_state(context, :distillation_batch)
    teacher_session = Context.get_state(context, :teacher_session)
    tokenizer = Context.get_state(context, :tokenizer)

    model = Context.get_config(context, :teacher_model) || Context.get_config(context, :model)
    max_tokens = Context.get_config(context, :max_tokens, 256)
    temperature = Context.get_config(context, :temperature, 1.0)

    renderer_module = renderer_for_model(model)

    with {:ok, renderer_state} <- renderer_module.init(tokenizer: tokenizer) do
      {responses, metrics} =
        batch
        |> Enum.map(&normalize_messages/1)
        |> Enum.map(fn messages ->
          {prompt, _state} =
            Renderer.build_generation_prompt(
              renderer_module,
              messages,
              "assistant",
              nil,
              renderer_state
            )

          sampling_params = %{
            max_tokens: max_tokens,
            temperature: temperature,
            stop: renderer_module.stop_sequences(renderer_state)
          }

          ports = get_train_ports(context)

          with {:ok, future} <-
                 SamplingClient.sample(ports, teacher_session, prompt, sampling_params),
               {:ok, response} <- SamplingClient.await(ports, future) do
            {message, token_count} =
              parse_response(response, renderer_module, renderer_state, tokenizer)

            {message, token_count}
          else
            {:error, reason} ->
              Logger.error("[TeacherInference] Sampling failed: #{inspect(reason)}")
              {Types.message("assistant", ""), 0}
          end
        end)
        |> unzip_responses()

      context
      |> Context.put_state(:teacher_responses, responses)
      |> Context.put_state(:teacher_metrics, metrics)
      |> then(&{:ok, &1})
    end
  end

  @impl true
  def validate(context) do
    cond do
      Context.get_state(context, :distillation_batch) == nil ->
        {:error, "distillation_batch is required in state"}

      Context.get_state(context, :teacher_session) == nil ->
        {:error, "teacher_session is required in state"}

      true ->
        :ok
    end
  end

  defp normalize_messages(messages) when is_list(messages) do
    Enum.map(messages, fn
      %{role: role, content: content} -> Types.message(role, content)
      %{"role" => role, "content" => content} -> Types.message(role, content)
      message -> message
    end)
  end

  defp normalize_messages(_), do: []

  defp parse_response(%{sequences: [sequence | _]}, renderer_module, renderer_state, tokenizer) do
    tokens = Map.get(sequence, :tokens, [])

    if function_exported?(renderer_module, :parse_response, 2) do
      {message, _} = renderer_module.parse_response(tokens, renderer_state)
      {message, length(tokens)}
    else
      content = Helpers.decode(tokenizer, tokens)
      {Types.message("assistant", content), length(tokens)}
    end
  end

  defp parse_response(_, _renderer_module, _renderer_state, _tokenizer) do
    {Types.message("assistant", ""), 0}
  end

  defp unzip_responses(responses_with_counts) do
    {responses, token_counts} = Enum.unzip(responses_with_counts)

    metrics = %{
      num_samples: length(responses),
      num_tokens: Enum.sum(token_counts)
    }

    {responses, metrics}
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
