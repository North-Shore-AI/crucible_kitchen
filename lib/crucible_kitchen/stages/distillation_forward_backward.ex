defmodule CrucibleKitchen.Stages.DistillationForwardBackward do
  @moduledoc """
  Runs forward-backward for distillation datums.

  ## Context Requirements

  **Input:**
  - State: `:distill_datums`
  - State: `:session`

  **Config:**
  - `:distill_loss_fn` (default: :cross_entropy)
  - `:distill_loss_fn_config` (optional)

  **Output:**
  - State: `:distillation_future`
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :distillation_forward_backward

  @impl true
  def execute(context) do
    session = Context.get_state(context, :session)
    datums = Context.get_state(context, :distill_datums, [])

    loss_fn = Context.get_config(context, :distill_loss_fn, :cross_entropy)
    loss_fn_config = Context.get_config(context, :distill_loss_fn_config)

    ports = get_train_ports(context)

    Logger.debug("[DistillationForwardBackward] loss_fn=#{inspect(loss_fn)}")

    future =
      TrainingClient.forward_backward(ports, session, datums,
        loss_fn: loss_fn,
        loss_fn_config: loss_fn_config
      )

    context
    |> Context.put_state(:distillation_future, future)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    cond do
      Context.get_state(context, :session) == nil ->
        {:error, "session is required in state"}

      Context.get_state(context, :distill_datums) == nil ->
        {:error, "distill_datums is required in state"}

      true ->
        :ok
    end
  end
end
