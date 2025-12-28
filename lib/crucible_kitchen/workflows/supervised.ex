defmodule CrucibleKitchen.Workflows.Supervised do
  @moduledoc """
  Standard supervised learning workflow.

  This workflow implements the classic SFT (Supervised Fine-Tuning) pipeline:

  1. Load dataset
  2. Initialize training session
  3. For each epoch:
     a. For each batch:
        - Render messages to tokens
        - Forward-backward pass
        - Optimizer step
        - Log metrics
     b. Maybe checkpoint
     c. Maybe evaluate
  4. Save final weights

  ## Usage

      CrucibleKitchen.run(:supervised, %{
        model: "meta-llama/Llama-3.1-8B",
        epochs: 3,
        batch_size: 128,
        learning_rate: 2.0e-4
      }, adapters: my_adapters)

  ## Required Adapters

  - `:training_client` - Backend for training operations
  - `:dataset_store` - Dataset loading

  ## Optional Adapters

  - `:blob_store` - Checkpoint storage
  - `:metrics_store` - Metrics persistence
  """

  use CrucibleKitchen.Workflow

  alias CrucibleKitchen.Stages

  workflow do
    # Setup
    stage(:load_dataset, Stages.LoadDataset)
    stage(:init_session, Stages.InitSession)
    stage(:init_tokenizer, Stages.InitTokenizer)
    stage(:build_dataset, Stages.BuildSupervisedDataset)

    # Training loop
    loop :epochs, over: fn ctx -> epochs_range(ctx) end do
      stage(:set_epoch, Stages.SetEpoch)

      loop :batches, over: fn ctx -> batches_range(ctx) end do
        stage(:get_batch, Stages.GetBatch)
        stage(:forward_backward, Stages.ForwardBackward)
        stage(:await_fb, Stages.AwaitFuture, key: :fb_future)
        stage(:optim_step, Stages.OptimStep)
        stage(:await_optim, Stages.AwaitFuture, key: :optim_future)
        stage(:log_step_metrics, Stages.LogStepMetrics)
      end

      stage(:log_epoch_metrics, Stages.LogEpochMetrics)

      conditional fn ctx -> should_checkpoint?(ctx) end do
        stage(:checkpoint, Stages.SaveCheckpoint)
      end

      conditional fn ctx -> should_evaluate?(ctx) end do
        stage(:evaluate, Stages.Evaluate)
      end
    end

    # Finalize
    stage(:save_final, Stages.SaveFinalWeights)
    stage(:cleanup, Stages.Cleanup)
  end

  defp epochs_range(ctx) do
    num_epochs = CrucibleKitchen.Context.get_config(ctx, :epochs, 1)
    0..(num_epochs - 1)
  end

  defp batches_range(ctx) do
    dataset = CrucibleKitchen.Context.get_state(ctx, :dataset)
    if dataset, do: 0..(dataset.num_batches - 1), else: []
  end

  defp should_checkpoint?(ctx) do
    save_every = CrucibleKitchen.Context.get_config(ctx, :save_every, 0)
    global_step = CrucibleKitchen.Context.get_state(ctx, :global_step, 0)
    save_every > 0 and rem(global_step + 1, save_every) == 0
  end

  defp should_evaluate?(ctx) do
    eval_every = CrucibleKitchen.Context.get_config(ctx, :eval_every, 0)
    global_step = CrucibleKitchen.Context.get_state(ctx, :global_step, 0)
    eval_every > 0 and rem(global_step + 1, eval_every) == 0
  end
end
