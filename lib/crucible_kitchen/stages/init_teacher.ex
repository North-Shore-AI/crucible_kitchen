defmodule CrucibleKitchen.Stages.InitTeacher do
  @moduledoc """
  Initializes the teacher sampling session for distillation.

  ## Context Requirements

  **Config:**
  - `:teacher_model` - Teacher model name (defaults to `:model`)
  - `:teacher_checkpoint_path` - Optional checkpoint path for teacher

  **Output:**
  - State: `:teacher_session`
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Ports.SamplingClient

  require Logger

  @impl true
  def name, do: :init_teacher

  @impl true
  def execute(context) do
    case Context.get_state(context, :teacher_session) do
      nil ->
        ports = get_train_ports(context)
        config = sampling_config(context)

        case SamplingClient.start_session(ports, config) do
          {:ok, session} ->
            Logger.info("[InitTeacher] Teacher session initialized")

            context
            |> Context.put_state(:teacher_session, session)
            |> then(&{:ok, &1})

          {:error, reason} ->
            {:error, {:teacher_session_failed, reason}}
        end

      _session ->
        {:ok, context}
    end
  end

  defp sampling_config(context) do
    %{
      model: Context.get_config(context, :teacher_model) || Context.get_config(context, :model),
      base_model: Context.get_config(context, :teacher_model),
      model_path: Context.get_config(context, :teacher_model_path),
      checkpoint_path: Context.get_config(context, :teacher_checkpoint_path)
    }
  end
end
