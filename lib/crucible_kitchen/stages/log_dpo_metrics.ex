defmodule CrucibleKitchen.Stages.LogDPOMetrics do
  @moduledoc """
  Logs DPO-specific metrics for monitoring and debugging.

  Emits telemetry events and records metrics for:
  - DPO loss
  - Preference accuracy (how often policy prefers chosen)
  - Chosen/rejected rewards (implicit rewards from logprob differences)
  - Reward margin

  ## Context Requirements

  **Input:**
  - State: `:dpo_metrics` - Metrics from DPOForwardBackward stage
  - State: `:global_step` - Current training step

  **Output:**
  - Recorded metrics in context
  - Telemetry event emitted

  ## Telemetry Events

  Emits `[:crucible_kitchen, :dpo, :step]` with measurements:
  - `loss` - DPO loss value
  - `accuracy` - Preference accuracy
  - `chosen_reward` - Implicit reward for chosen
  - `rejected_reward` - Implicit reward for rejected
  - `margin` - Reward margin (chosen - rejected)

  ## Example

      stage(:log_dpo_metrics, LogDPOMetrics)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :log_dpo_metrics

  @impl true
  def execute(context) do
    metrics = Context.get_state(context, :dpo_metrics, %{})
    metrics = normalize_metrics(metrics)
    global_step = Context.get_state(context, :global_step, 0)
    _batch_index = Context.get_state(context, :batch_index, 0)

    # Extract metrics with defaults
    loss = fetch_metric(metrics, :loss, 0.0)
    accuracy = fetch_metric(metrics, :accuracy, 0.0)
    chosen_reward = fetch_metric(metrics, :chosen_reward, 0.0)
    rejected_reward = fetch_metric(metrics, :rejected_reward, 0.0)
    margin = Map.get(metrics, :margin, chosen_reward - rejected_reward)

    Logger.info(
      "[Step #{global_step}] DPO loss=#{Float.round(loss, 4)} " <>
        "acc=#{Float.round(accuracy * 100, 1)}% " <>
        "margin=#{Float.round(margin, 4)}"
    )

    emit_telemetry(global_step, metrics)

    context
    |> Context.record_metric(:dpo_loss, loss, step: global_step)
    |> Context.record_metric(:dpo_accuracy, accuracy, step: global_step)
    |> Context.record_metric(:dpo_chosen_reward, chosen_reward, step: global_step)
    |> Context.record_metric(:dpo_rejected_reward, rejected_reward, step: global_step)
    |> Context.record_metric(:dpo_margin, margin, step: global_step)
    |> increment_step()
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(_context), do: :ok

  defp emit_telemetry(step, metrics) do
    measurements = %{
      loss: fetch_metric(metrics, :loss, 0.0),
      accuracy: fetch_metric(metrics, :accuracy, 0.0),
      chosen_reward: fetch_metric(metrics, :chosen_reward, 0.0),
      rejected_reward: fetch_metric(metrics, :rejected_reward, 0.0),
      margin: Map.get(metrics, :margin, 0.0)
    }

    metadata = %{
      step: step,
      beta: fetch_metric(metrics, :beta, 0.1),
      num_pairs: fetch_metric(metrics, :num_pairs, 0)
    }

    :telemetry.execute(
      [:crucible_kitchen, :dpo, :step],
      measurements,
      metadata
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
