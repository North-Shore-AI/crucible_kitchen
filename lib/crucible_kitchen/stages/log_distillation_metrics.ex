defmodule CrucibleKitchen.Stages.LogDistillationMetrics do
  @moduledoc """
  Logs distillation-specific metrics.

  ## Context Requirements

  **Input:**
  - State: `:distillation_metrics`
  - State: `:teacher_metrics` (optional)
  - State: `:global_step`
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :log_distillation_metrics

  @impl true
  def execute(context) do
    metrics = Context.get_state(context, :distillation_metrics, %{})
    teacher_metrics = Context.get_state(context, :teacher_metrics, %{})
    global_step = Context.get_state(context, :global_step, 0)

    metrics = normalize_metrics(metrics)
    loss = fetch_metric(metrics, :loss, 0.0)
    num_sequences = fetch_metric(metrics, :num_sequences, 0)

    Logger.info(
      "[Step #{global_step}] Distill loss=#{Float.round(loss, 4)} " <>
        "num_sequences=#{num_sequences}"
    )

    emit_telemetry(global_step, metrics, teacher_metrics)

    context
    |> Context.record_metric(:distill_loss, loss, step: global_step)
    |> Context.record_metric(:distill_num_sequences, num_sequences, step: global_step)
    |> increment_step()
    |> then(&{:ok, &1})
  end

  defp emit_telemetry(step, metrics, teacher_metrics) do
    measurements = %{
      loss: fetch_metric(metrics, :loss, 0.0),
      num_sequences: fetch_metric(metrics, :num_sequences, 0),
      teacher_num_samples: Map.get(teacher_metrics, :num_samples, 0),
      teacher_num_tokens: Map.get(teacher_metrics, :num_tokens, 0)
    }

    :telemetry.execute(
      [:crucible_kitchen, :distillation, :step],
      measurements,
      %{step: step}
    )
  end

  defp increment_step(context) do
    current = Context.get_state(context, :global_step, 0)
    Context.put_state(context, :global_step, current + 1)
  end

  defp normalize_metrics(%Tinkex.Types.ForwardBackwardOutput{metrics: metrics}), do: metrics
  defp normalize_metrics(%{metrics: metrics}) when is_map(metrics), do: metrics
  defp normalize_metrics(metrics) when is_map(metrics), do: metrics
  defp normalize_metrics(_), do: %{}

  defp fetch_metric(metrics, key, default) do
    Map.get(metrics, key, Map.get(metrics, Atom.to_string(key), default))
  end
end
