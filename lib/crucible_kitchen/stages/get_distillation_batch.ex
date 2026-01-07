defmodule CrucibleKitchen.Stages.GetDistillationBatch do
  @moduledoc """
  Retrieves the next batch of prompts/messages for distillation.

  ## Context Requirements

  **Input:**
  - State: `:distillation_dataset`
  - State: `:dist_batches_index` - Current batch index (from workflow loop)

  **Output:**
  - State: `:distillation_batch`
  - State: `:batch_index`
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :get_distillation_batch

  @impl true
  def execute(context) do
    dataset = Context.get_state(context, :distillation_dataset)
    batch_index = Context.get_state(context, :dist_batches_index, 0)

    batch = get_batch(dataset, batch_index)

    Logger.debug(
      "[GetDistillationBatch] batch #{batch_index + 1}/#{dataset.num_batches} " <>
        "(#{length(batch)} samples)"
    )

    context
    |> Context.put_state(:distillation_batch, batch)
    |> Context.put_state(:batch_index, batch_index)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :distillation_dataset) do
      nil -> {:error, "distillation_dataset is required in state"}
      _ -> :ok
    end
  end

  defp get_batch(dataset, batch_index) do
    start_idx = batch_index * dataset.batch_size
    end_idx = min(start_idx + dataset.batch_size, length(dataset.samples))
    Enum.slice(dataset.samples, start_idx, end_idx - start_idx)
  end
end
