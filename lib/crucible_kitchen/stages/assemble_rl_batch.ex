defmodule CrucibleKitchen.Stages.AssembleRLBatch do
  @moduledoc """
  Assembles trajectory data into training batch for PPO update.

  Converts the collected trajectories with computed advantages into
  a format suitable for policy gradient training.

  ## Context Requirements

  **Input:**
  - State: `:trajectory_group` - Trajectories from DoRollout
  - State: `:advantages_g` - Advantages per group from ComputeAdvantages

  **Output:**
  - State: `:current_batch` - Assembled training datums
  - State: `:rl_batch_metadata` - Per-datum metadata
  - State: `:rl_batch_size` - Batch size

  ## Example

      stage(:assemble_rl_batch, AssembleRLBatch)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.RL.DataProcessing

  require Logger

  @impl true
  def name, do: :assemble_rl_batch

  @impl true
  def execute(context) do
    trajectory_group = Context.get_state(context, :trajectory_group)
    advantages_g = Context.get_state(context, :advantages_g)

    {datums, metadata} =
      DataProcessing.assemble_training_data(
        [trajectory_group],
        advantages_g
      )

    batch_size = length(datums)

    Logger.debug("Assembled RL batch with #{batch_size} datums")

    emit_telemetry(batch_size, length(trajectory_group.trajectories_G))

    context
    |> Context.put_state(:current_batch, datums)
    |> Context.put_state(:rl_batch_metadata, metadata)
    |> Context.put_state(:rl_batch_size, batch_size)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    cond do
      Context.get_state(context, :trajectory_group) == nil ->
        {:error, "trajectory_group is required (run DoRollout first)"}

      Context.get_state(context, :advantages_g) == nil ->
        {:error, "advantages_g is required (run ComputeAdvantages first)"}

      true ->
        :ok
    end
  end

  defp emit_telemetry(batch_size, num_trajectories) do
    :telemetry.execute(
      [:crucible_kitchen, :rl, :batch_assembled],
      %{
        batch_size: batch_size,
        num_trajectories: num_trajectories
      },
      %{}
    )
  end
end
