defmodule CrucibleKitchen.Stages.BuildDistillDatums do
  @moduledoc """
  Builds supervised datums from prompt messages and teacher responses.

  ## Context Requirements

  **Input:**
  - State: `:distillation_batch`
  - State: `:teacher_responses`
  - State: `:tokenizer`

  **Config:**
  - `:model`
  - `:max_length`

  **Output:**
  - State: `:distill_datums`
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Renderers.{Renderer, TrainOnWhat, Types}
  alias CrucibleTrain.Supervised.Common

  @impl true
  def name, do: :build_distill_datums

  @impl true
  def execute(context) do
    batch = Context.get_state(context, :distillation_batch)
    responses = Context.get_state(context, :teacher_responses)
    tokenizer = Context.get_state(context, :tokenizer)
    model = Context.get_config(context, :model)
    max_length = Context.get_config(context, :max_length, 8192)

    renderer_module = renderer_for_model(model)

    with {:ok, renderer_state} <- renderer_module.init(tokenizer: tokenizer) do
      datums =
        Enum.zip(batch, responses)
        |> Enum.map(fn {messages, response} ->
          convo = normalize_messages(messages) ++ [normalize_response(response)]

          {model_input, weights} =
            Renderer.build_supervised_example(
              renderer_module,
              convo,
              TrainOnWhat.last_assistant_message(),
              renderer_state
            )

          Common.datum_from_model_input_weights(model_input, weights, max_length)
        end)

      context
      |> Context.put_state(:distill_datums, datums)
      |> then(&{:ok, &1})
    end
  end

  @impl true
  def validate(context) do
    cond do
      Context.get_state(context, :distillation_batch) == nil ->
        {:error, "distillation_batch is required in state"}

      Context.get_state(context, :teacher_responses) == nil ->
        {:error, "teacher_responses is required in state"}

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

  defp normalize_response(%{role: role, content: content}), do: Types.message(role, content)

  defp normalize_response(%{"role" => role, "content" => content}),
    do: Types.message(role, content)

  defp normalize_response(content) when is_binary(content),
    do: Types.message("assistant", content)

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
