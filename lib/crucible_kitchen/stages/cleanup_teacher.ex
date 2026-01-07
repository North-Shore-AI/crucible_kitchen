defmodule CrucibleKitchen.Stages.CleanupTeacher do
  @moduledoc """
  Cleans up the teacher sampling session after distillation.
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.SamplingClient

  require Logger

  @impl true
  def name, do: :cleanup_teacher

  @impl true
  def execute(context) do
    case Context.get_state(context, :teacher_session) do
      nil ->
        {:ok, context}

      session ->
        Logger.info("[CleanupTeacher] Closing teacher sampling session")
        ports = get_train_ports(context)
        SamplingClient.close_session(ports, session)

        context
        |> Context.put_state(:teacher_session, nil)
        |> then(&{:ok, &1})
    end
  end
end
