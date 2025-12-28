defmodule CrucibleKitchen.Stages do
  @moduledoc """
  Built-in stage implementations.

  These stages implement common operations for ML training workflows.
  Custom stages can be created by implementing the `CrucibleKitchen.Stage` behaviour.

  ## Available Stages

  ### Setup Stages
  - `LoadDataset` - Load dataset from configured source
  - `InitSession` - Initialize training session with backend
  - `InitTokenizer` - Load tokenizer for rendering
  - `BuildSupervisedDataset` - Build batched dataset for training

  ### Training Stages
  - `SetEpoch` - Set current epoch and shuffle dataset
  - `GetBatch` - Get current batch from dataset
  - `ForwardBackward` - Compute gradients
  - `AwaitFuture` - Await async operation
  - `OptimStep` - Apply gradients with learning rate
  - `LogStepMetrics` - Log training step metrics
  - `LogEpochMetrics` - Log epoch summary metrics

  ### Checkpoint/Eval Stages
  - `SaveCheckpoint` - Save training checkpoint
  - `Evaluate` - Run evaluation
  - `SaveFinalWeights` - Save final model weights
  - `Cleanup` - Clean up resources

  ### Utility Stages
  - `Noop` - No-operation (placeholder)
  """

  # Re-export all stage modules for convenience
  defdelegate noop(), to: CrucibleKitchen.Stages.Noop, as: :name
end

# Placeholder implementations - these will be fully implemented in Phase 3

defmodule CrucibleKitchen.Stages.LoadDataset do
  @moduledoc "Load dataset from configured source."
  use CrucibleKitchen.Stage

  def name, do: :load_dataset

  def execute(context) do
    dataset_store = get_adapter(context, :dataset_store)
    dataset_name = get_config(context, :dataset, "no_robots")

    case dataset_store.load(dataset_name, split: "train") do
      {:ok, samples} ->
        {:ok, put_state(context, :samples, samples)}

      {:error, reason} ->
        {:error, {:dataset_load_failed, reason}}
    end
  end
end

defmodule CrucibleKitchen.Stages.InitSession do
  @moduledoc "Initialize training session with backend."
  use CrucibleKitchen.Stage

  def name, do: :init_session

  def execute(context) do
    training_client = get_adapter(context, :training_client)

    session_config = %{
      model: get_config(context, :model),
      lora_rank: get_config(context, :lora_rank, 32),
      learning_rate: get_config(context, :learning_rate, 2.0e-4)
    }

    case training_client.start_session(session_config) do
      {:ok, session} ->
        {:ok, put_state(context, :session, session)}

      {:error, reason} ->
        {:error, {:session_init_failed, reason}}
    end
  end
end

defmodule CrucibleKitchen.Stages.InitTokenizer do
  @moduledoc "Load tokenizer for rendering."
  use CrucibleKitchen.Stage

  def name, do: :init_tokenizer

  def execute(context) do
    training_client = get_adapter(context, :training_client)
    session = get_state(context, :session)

    case training_client.get_tokenizer(session) do
      {:ok, tokenizer} ->
        {:ok, put_state(context, :tokenizer, tokenizer)}

      {:error, reason} ->
        {:error, {:tokenizer_init_failed, reason}}
    end
  end
end

defmodule CrucibleKitchen.Stages.BuildSupervisedDataset do
  @moduledoc "Build batched dataset for training."
  use CrucibleKitchen.Stage

  def name, do: :build_supervised_dataset

  def execute(context) do
    samples = get_state(context, :samples, [])
    batch_size = get_config(context, :batch_size, 128)

    # Simple batching - real implementation would use CrucibleTrain.Supervised.Dataset
    num_batches = ceil(length(samples) / batch_size)

    dataset = %{
      samples: samples,
      batch_size: batch_size,
      num_batches: num_batches,
      current_epoch: 0
    }

    {:ok, put_state(context, :dataset, dataset)}
  end
end

defmodule CrucibleKitchen.Stages.SetEpoch do
  @moduledoc "Set current epoch and shuffle dataset."
  use CrucibleKitchen.Stage

  def name, do: :set_epoch

  def execute(context) do
    current_epoch = get_state(context, :epochs_current, 0)
    dataset = get_state(context, :dataset)

    dataset = %{dataset | current_epoch: current_epoch}

    context
    |> put_state(:dataset, dataset)
    |> put_state(:current_epoch, current_epoch)
    |> then(&{:ok, &1})
  end
end

defmodule CrucibleKitchen.Stages.GetBatch do
  @moduledoc "Get current batch from dataset."
  use CrucibleKitchen.Stage

  def name, do: :get_batch

  def execute(context) do
    batch_idx = get_state(context, :batches_current, 0)
    dataset = get_state(context, :dataset)

    # Simple batch extraction
    start_idx = batch_idx * dataset.batch_size
    batch = Enum.slice(dataset.samples, start_idx, dataset.batch_size)

    {:ok, put_state(context, :current_batch, batch)}
  end
end

defmodule CrucibleKitchen.Stages.ForwardBackward do
  @moduledoc "Compute gradients."
  use CrucibleKitchen.Stage

  def name, do: :forward_backward

  def execute(context) do
    training_client = get_adapter(context, :training_client)
    session = get_state(context, :session)
    batch = get_state(context, :current_batch)

    case training_client.forward_backward(session, batch) do
      {:ok, future} ->
        {:ok, put_state(context, :fb_future, future)}

      {:error, reason} ->
        {:error, {:forward_backward_failed, reason}}
    end
  end
end

defmodule CrucibleKitchen.Stages.AwaitFuture do
  @moduledoc "Await async operation."
  use CrucibleKitchen.Stage

  def name, do: :await_future

  def execute(%{stage_opts: opts} = context) do
    key = Keyword.get(opts, :key, :future)
    future = get_state(context, key)

    if future do
      training_client = get_adapter(context, :training_client)

      case training_client.await(future) do
        {:ok, result} ->
          result_key = :"#{key}_result"
          {:ok, put_state(context, result_key, result)}

        {:error, reason} ->
          {:error, {:await_failed, key, reason}}
      end
    else
      {:ok, context}
    end
  end
end

defmodule CrucibleKitchen.Stages.OptimStep do
  @moduledoc "Apply gradients with learning rate."
  use CrucibleKitchen.Stage

  def name, do: :optim_step

  def execute(context) do
    training_client = get_adapter(context, :training_client)
    session = get_state(context, :session)
    global_step = get_state(context, :global_step, 0)
    total_steps = get_state(context, :total_steps, 1)

    # Compute learning rate (with optional schedule)
    base_lr = get_config(context, :learning_rate, 2.0e-4)
    lr_schedule = get_config(context, :lr_schedule, :linear)
    lr = compute_lr(base_lr, global_step, total_steps, lr_schedule)

    case training_client.optim_step(session, lr) do
      {:ok, future} ->
        context
        |> put_state(:optim_future, future)
        |> put_state(:current_lr, lr)
        |> then(&{:ok, &1})

      {:error, reason} ->
        {:error, {:optim_step_failed, reason}}
    end
  end

  defp compute_lr(base_lr, step, total_steps, schedule) do
    progress = step / max(total_steps - 1, 1)

    case schedule do
      :constant -> base_lr
      :linear -> base_lr * (1.0 - progress)
      :cosine -> base_lr * 0.5 * (1.0 + :math.cos(:math.pi() * progress))
      _ -> base_lr
    end
  end
end

defmodule CrucibleKitchen.Stages.LogStepMetrics do
  @moduledoc "Log training step metrics."
  use CrucibleKitchen.Stage

  def name, do: :log_step_metrics

  def execute(context) do
    global_step = get_state(context, :global_step, 0)
    lr = get_state(context, :current_lr, 0.0)
    fb_result = get_state(context, :fb_future_result, %{})
    loss = Map.get(fb_result, :loss, 0.0)

    # Emit telemetry
    CrucibleKitchen.Telemetry.emit_step(
      %{loss: loss, lr: lr},
      %{step: global_step, total_steps: get_state(context, :total_steps, 0)}
    )

    # Record metric
    context = record_metric(context, :loss, loss, step: global_step)
    context = record_metric(context, :lr, lr, step: global_step)

    # Increment step counter
    {:ok, put_state(context, :global_step, global_step + 1)}
  end
end

defmodule CrucibleKitchen.Stages.LogEpochMetrics do
  @moduledoc "Log epoch summary metrics."
  use CrucibleKitchen.Stage

  def name, do: :log_epoch_metrics

  def execute(context) do
    current_epoch = get_state(context, :current_epoch, 0)
    total_epochs = get_config(context, :epochs, 1)

    CrucibleKitchen.Telemetry.emit_epoch(
      %{},
      %{epoch: current_epoch, total_epochs: total_epochs}
    )

    {:ok, context}
  end
end

defmodule CrucibleKitchen.Stages.SaveCheckpoint do
  @moduledoc "Save training checkpoint."
  use CrucibleKitchen.Stage

  def name, do: :save_checkpoint

  def execute(context) do
    training_client = get_adapter(context, :training_client)
    session = get_state(context, :session)
    global_step = get_state(context, :global_step, 0)

    checkpoint_name = "step_#{String.pad_leading(Integer.to_string(global_step), 6, "0")}"

    case training_client.save_checkpoint(session, checkpoint_name) do
      {:ok, path} ->
        CrucibleKitchen.Telemetry.emit_checkpoint(
          %{},
          %{checkpoint_name: checkpoint_name, path: path}
        )

        {:ok, context}

      {:error, reason} ->
        {:error, {:checkpoint_failed, reason}}
    end
  end
end

defmodule CrucibleKitchen.Stages.Evaluate do
  @moduledoc "Run evaluation."
  use CrucibleKitchen.Stage

  def name, do: :evaluate

  def execute(context) do
    # Placeholder - real implementation would run eval dataset
    {:ok, context}
  end
end

defmodule CrucibleKitchen.Stages.SaveFinalWeights do
  @moduledoc "Save final model weights."
  use CrucibleKitchen.Stage

  def name, do: :save_final_weights

  def execute(context) do
    training_client = get_adapter(context, :training_client)
    session = get_state(context, :session)

    case training_client.save_checkpoint(session, "final") do
      {:ok, path} ->
        {:ok, put_state(context, :final_weights_path, path)}

      {:error, reason} ->
        {:error, {:save_final_failed, reason}}
    end
  end
end

defmodule CrucibleKitchen.Stages.Cleanup do
  @moduledoc "Clean up resources."
  use CrucibleKitchen.Stage

  def name, do: :cleanup

  def execute(context) do
    training_client = get_adapter(context, :training_client)
    session = get_state(context, :session)

    if session do
      training_client.close_session(session)
    end

    {:ok, put_state(context, :final_weights_saved, true)}
  end
end
