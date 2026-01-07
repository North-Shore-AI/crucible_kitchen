defmodule CrucibleKitchen.Stages.LogRLMetrics do
  @moduledoc """
  Logs RL-specific metrics for monitoring and debugging.

  Emits telemetry events and records metrics for:
  - Reward statistics (mean, std, min, max)
  - Policy loss and entropy
  - KL divergence
  - Clip fraction (PPO)
  - Advantage statistics

  ## Context Requirements

  **Input:**
  - State: `:rollout_metrics` - Metrics from DoRollout
  - State: `:fb_result` - Forward/backward result metrics
  - State: `:global_step` - Current training step

  **Output:**
  - Recorded metrics in context
  - Telemetry event emitted

  ## Telemetry Events

  Emits `[:crucible_kitchen, :rl, :step]` with measurements:
  - `reward_mean` - Mean reward
  - `reward_std` - Reward standard deviation
  - `policy_loss` - Policy gradient loss
  - `entropy` - Policy entropy
  - `kl_divergence` - KL from old policy

  ## Example

      stage(:log_rl_metrics, LogRLMetrics)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :log_rl_metrics

  @impl true
  def execute(context) do
    rollout_metrics = Context.get_state(context, :rollout_metrics, %{})
    fb_result = Context.get_state(context, :fb_result, %{})
    global_step = Context.get_state(context, :global_step, 0)
    rollout_index = Context.get_state(context, :rollouts_index, 0)

    # Extract metrics
    reward_mean = Map.get(rollout_metrics, :reward_mean, 0.0)
    reward_std = Map.get(rollout_metrics, :reward_std, 0.0)
    train_metrics = fb_result[:metrics] || fb_result["metrics"] || %{}
    policy_loss = metric_value(train_metrics, :policy_loss, :loss)
    entropy = metric_value(train_metrics, :entropy)
    kl_div = metric_value(train_metrics, :kl_divergence)

    Logger.info(
      "[Rollout #{rollout_index + 1}] " <>
        "reward=#{Float.round(reward_mean, 4)}±#{Float.round(reward_std, 4)} " <>
        "policy_loss=#{Float.round(policy_loss, 4)} " <>
        "entropy=#{Float.round(entropy, 4)}"
    )

    combined_metrics = Map.merge(rollout_metrics, train_metrics)
    emit_telemetry(global_step, rollout_index, combined_metrics)

    context
    |> Context.record_metric(:rl_reward_mean, reward_mean, step: global_step)
    |> Context.record_metric(:rl_reward_std, reward_std, step: global_step)
    |> Context.record_metric(:rl_policy_loss, policy_loss, step: global_step)
    |> Context.record_metric(:rl_entropy, entropy, step: global_step)
    |> Context.record_metric(:rl_kl_divergence, kl_div, step: global_step)
    |> increment_step()
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(_context), do: :ok

  defp emit_telemetry(step, rollout_index, metrics) do
    measurements = %{
      reward_mean: Map.get(metrics, :reward_mean, 0.0),
      reward_std: Map.get(metrics, :reward_std, 0.0),
      policy_loss: metric_value(metrics, :policy_loss, :loss),
      value_loss: metric_value(metrics, :value_loss),
      entropy: metric_value(metrics, :entropy),
      kl_divergence: metric_value(metrics, :kl_divergence),
      num_trajectories: Map.get(metrics, :num_trajectories, 0),
      total_steps: Map.get(metrics, :total_steps, 0)
    }

    metadata = %{
      step: step,
      rollout_index: rollout_index
    }

    :telemetry.execute(
      [:crucible_kitchen, :rl, :step],
      measurements,
      metadata
    )
  end

  defp increment_step(context) do
    current = Context.get_state(context, :global_step, 0)
    Context.put_state(context, :global_step, current + 1)
  end

  defp metric_value(metrics, key, fallback_key \\ nil) do
    Map.get(metrics, key) ||
      Map.get(metrics, to_string(key)) ||
      if(fallback_key,
        do: Map.get(metrics, fallback_key) || Map.get(metrics, to_string(fallback_key)),
        else: nil
      ) ||
      0.0
  end
end
