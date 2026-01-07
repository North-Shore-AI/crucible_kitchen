defmodule CrucibleKitchen.Stages.BuildDistillationDataset do
  @moduledoc """
  Builds a prompt-only distillation dataset from raw samples.

  ## Context Requirements

  **Input:**
  - State: `:dataset_handle` - Dataset handle from LoadDataset

  **Config:**
  - `:batch_size` - Batch size for distillation

  **Output:**
  - State: `:distillation_dataset` - Dataset with `samples`, `batch_size`, `num_batches`
  - State: `:num_distillation_samples`
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.DatasetStore
  alias CrucibleTrain.Renderers.Types

  require Logger

  @impl true
  def name, do: :build_distillation_dataset

  @impl true
  def execute(context) do
    dataset_handle = Context.get_state(context, :dataset_handle)
    batch_size = Context.get_config(context, :batch_size, 32)

    ports = get_train_ports(context)

    case DatasetStore.to_list(ports, dataset_handle) do
      {:ok, samples} ->
        messages = Enum.flat_map(samples, &sample_to_messages/1)
        num_samples = length(messages)
        num_batches = div(num_samples + batch_size - 1, batch_size)

        dataset = %{
          samples: messages,
          batch_size: batch_size,
          num_batches: num_batches
        }

        Logger.info("[BuildDistillationDataset] #{num_samples} samples, #{num_batches} batches")

        context
        |> Context.put_state(:distillation_dataset, dataset)
        |> Context.put_state(:num_distillation_samples, num_samples)
        |> Context.record_metric(:distillation_samples, num_samples)
        |> then(&{:ok, &1})

      {:error, reason} ->
        Logger.error("[BuildDistillationDataset] Failed to load samples: #{inspect(reason)}")
        {:error, {:dataset_to_list_failed, reason}}
    end
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :dataset_handle) do
      nil -> {:error, "dataset_handle is required in state"}
      _ -> :ok
    end
  end

  defp sample_to_messages(%{"messages" => messages}) when is_list(messages),
    do: [normalize_messages(messages)]

  defp sample_to_messages(%{messages: messages}) when is_list(messages),
    do: [normalize_messages(messages)]

  defp sample_to_messages(sample) when is_map(sample) do
    prompt =
      first_binary([
        sample["prompt"],
        sample[:prompt],
        sample["instruction"],
        sample[:instruction],
        sample["input"],
        sample[:input],
        sample["text"],
        sample[:text],
        get_in(sample, ["input", "problem"]),
        get_in(sample, [:input, :problem]),
        get_in(sample, ["input", "question"]),
        get_in(sample, [:input, :question])
      ])

    if is_binary(prompt) do
      [[Types.message("user", prompt)]]
    else
      []
    end
  end

  defp sample_to_messages(_), do: []

  defp first_binary(values) do
    Enum.find(values, &is_binary/1)
  end

  defp normalize_messages(messages) do
    Enum.map(messages, fn
      %{role: role, content: content} ->
        Types.message(role, content)

      %{"role" => role, "content" => content} ->
        Types.message(role, content)

      message ->
        message
    end)
  end
end
