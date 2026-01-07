defmodule CrucibleKitchen.Stages.BuildEnvGroup do
  @moduledoc """
  Builds environment group for RL rollout collection.

  Creates an environment group builder that produces parallel environments
  for trajectory collection. The environments implement the `CrucibleTrain.RL.Env`
  behaviour.

  ## Context Requirements

  **Input:**
  - Config: `:env_group_builder` - EnvGroupBuilder struct (preferred)
  - Config: `:env_thunk` - Zero-arity function to build envs (fallback)
  - Config: `:env` - Environment type (atom or module, fallback)
  - Config: `:group_size` - Number of environments per group (default: 4)
  - Config: `:groups_per_batch` - Number of groups per batch (default: 100)
  - State: `:raw_dataset` - Dataset for environment prompts (optional)

  **Output:**
  - State: `:env_group` - Environment group builder
  - State: `:env_config` - Environment configuration

  ## Example

      stage(:build_env_group, BuildEnvGroup)
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context

  require Logger

  @impl true
  def name, do: :build_env_group

  @impl true
  def execute(context) do
    env_group_builder = Context.get_config(context, :env_group_builder)
    env_thunk = Context.get_config(context, :env_thunk)
    env_type = Context.get_config(context, :env, :noop)
    group_size = Context.get_config(context, :group_size, 4)
    groups_per_batch = Context.get_config(context, :groups_per_batch, 100)
    raw_dataset = Context.get_state(context, :raw_dataset)

    Logger.info(
      "Building env group: group_size=#{group_size} groups_per_batch=#{groups_per_batch}"
    )

    env_config = %{
      env_type: env_type,
      group_size: group_size,
      groups_per_batch: groups_per_batch,
      dataset: raw_dataset
    }

    env_group =
      resolve_env_group_builder(env_group_builder, env_thunk, env_type, env_config, context)

    emit_telemetry(env_group, env_type, group_size, groups_per_batch)

    context
    |> Context.put_state(:env_group, env_group)
    |> Context.put_state(:env_config, env_config)
    |> then(&{:ok, &1})
  end

  @impl true
  def validate(context) do
    env_group_builder = Context.get_config(context, :env_group_builder)
    env_thunk = Context.get_config(context, :env_thunk)
    env_type = Context.get_config(context, :env, :noop)

    cond do
      is_function(env_group_builder, 1) -> :ok
      env_group_builder != nil -> :ok
      is_function(env_thunk, 0) -> :ok
      env_type != :noop -> :ok
      true -> {:error, "env_group_builder or env_thunk is required"}
    end
  end

  defp resolve_env_group_builder(env_group_builder, _env_thunk, _env_type, _config, context)
       when is_function(env_group_builder, 1) do
    env_group_builder.(context)
  end

  defp resolve_env_group_builder(%_{} = builder, _env_thunk, _env_type, _config, _context),
    do: builder

  defp resolve_env_group_builder(nil, env_thunk, _env_type, config, _context)
       when is_function(env_thunk, 0) do
    %CrucibleTrain.RL.ProblemGroupBuilder{
      env_thunk: env_thunk,
      num_envs: config.group_size,
      dataset_name: "env_group"
    }
  end

  defp resolve_env_group_builder(nil, _env_thunk, :noop, config, _context) do
    %CrucibleTrain.RL.ProblemGroupBuilder{
      env_thunk: fn -> %CrucibleKitchen.RL.NoopEnv{} end,
      num_envs: config.group_size,
      dataset_name: "noop"
    }
  end

  defp resolve_env_group_builder(nil, _env_thunk, env_module, config, _context)
       when is_atom(env_module) do
    %CrucibleTrain.RL.ProblemGroupBuilder{
      env_thunk: fn -> struct(env_module) end,
      num_envs: config.group_size,
      dataset_name: Atom.to_string(env_module)
    }
  end

  defp emit_telemetry(env_group, env_type, group_size, groups_per_batch) do
    group_size = Map.get(env_group, :num_envs, group_size)

    :telemetry.execute(
      [:crucible_kitchen, :rl, :env_group_built],
      %{group_size: group_size, groups_per_batch: groups_per_batch},
      %{env_type: env_type}
    )
  end
end
