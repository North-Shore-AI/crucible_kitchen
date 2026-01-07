defmodule CrucibleKitchen.Stages.ComputeAdvantages do
  @moduledoc """
  Computes advantages for RL training from trajectory rewards.

  ## Context Requirements

  **Input:**
  - State: `:trajectory_group` - Trajectories from DoRollout
  **Output:**
  - State: `:advantages_g` - Computed advantages per group
  - State: `:advantages` - Flattened advantages

  ## Algorithm

  Advantages are centered by the mean reward within each group.

  ## Example

      stage(:compute_advantages, ComputeAdvantages)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.RL.DataProcessing

  require Logger

  @impl true
  def name, do: :compute_advantages

  @impl true
  def execute(context) do
    trajectory_group = Context.get_state(context, :trajectory_group)
    Logger.debug("Computing advantages from trajectory group")

    advantages_g = DataProcessing.compute_advantages([trajectory_group])
    advantages = List.flatten(advantages_g)

    Logger.debug(
      "Computed advantages: mean=#{Float.round(mean(advantages), 4)} " <>
        "std=#{Float.round(std(advantages), 4)}"
    )

    emit_telemetry(advantages)

    context
    |> Context.put_state(:advantages_g, advantages_g)
    |> Context.put_state(:advantages, advantages)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :trajectory_group) do
      nil -> {:error, "trajectory_group is required (run DoRollout first)"}
      _ -> :ok
    end
  end

  defp mean([]), do: 0.0
  defp mean(list), do: Enum.sum(list) / length(list)

  defp std([]), do: 0.0
  defp std([_]), do: 0.0

  defp std(list) do
    m = mean(list)

    variance =
      list
      |> Enum.map(fn v -> (v - m) * (v - m) end)
      |> Enum.sum()
      |> Kernel./(length(list))

    :math.sqrt(variance)
  end

  defp emit_telemetry(advantages) do
    :telemetry.execute(
      [:crucible_kitchen, :rl, :advantages_computed],
      %{
        advantage_mean: mean(advantages),
        advantage_std: std(advantages),
        num_steps: length(advantages)
      },
      %{}
    )
  end
end
