defmodule CrucibleKitchen.Stages.ForwardBackward do
  @moduledoc """
  Stage for running forward-backward pass on a batch.

  Takes the current batch from state and runs forward-backward
  through the training client. Stores the result future in state.

  ## State Requirements

  - `:session` - Training session (from InitSession)
  - `:current_batch` - Current batch of datums (from GetBatch)

  ## State Updates

  - `:fb_future` - Future for the forward-backward result
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :forward_backward

  @impl true
  def validate(context) do
    session = get_state(context, :session)
    batch = get_state(context, :current_batch)

    cond do
      is_nil(session) -> {:error, "session not initialized"}
      is_nil(batch) -> {:error, "no batch available"}
      true -> :ok
    end
  end

  @impl true
  def execute(context) do
    session = get_state(context, :session)
    batch = get_state(context, :current_batch)
    loss_fn = get_state(context, :loss_fn) || get_config(context, :loss_fn, nil)

    loss_fn_config =
      get_state(context, :loss_fn_config) || get_config(context, :loss_fn_config, nil)

    ports = get_train_ports(context)

    opts =
      []
      |> maybe_put(:loss_fn, loss_fn)
      |> maybe_put(:loss_fn_config, loss_fn_config)

    case TrainingClient.forward_backward(ports, session, batch, opts) do
      {:error, reason} ->
        Logger.error("[ForwardBackward] Failed: #{inspect(reason)}")
        {:error, {:forward_backward_failed, reason}}

      future ->
        context = put_state(context, :fb_future, future)
        {:ok, context}
    end
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
