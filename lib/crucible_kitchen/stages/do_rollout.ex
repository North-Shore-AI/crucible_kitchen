defmodule CrucibleKitchen.Stages.DoRollout do
  @moduledoc """
  Collects trajectories via parallel rollouts for RL training.

  Executes rollouts in parallel across the environment group, collecting
  trajectories with observations, actions, rewards, and done flags.

  ## Context Requirements

  **Input:**
  - State: `:env_group` - Environment group builder from BuildEnvGroup
  - State: `:token_completer` - Token completer (optional)
  - State: `:sampling_session` - Sampling session (optional)
  - Config: `:max_tokens` - Maximum tokens per response (default: 512)
  - Config: `:temperature` - Sampling temperature (default: 1.0)

  **Output:**
  - State: `:trajectory_group` - Collected trajectories with rewards
  - State: `:rollout_metrics` - Metrics from rollout collection

  ## Trajectory Structure

  Each trajectory contains:
  - `observations` - List of observations (prompts)
  - `actions` - List of actions (responses)
  - `rewards` - List of rewards
  - `dones` - List of done flags
  - `logprobs` - List of action log probabilities

  ## Example

      loop :rollouts, over: :rollouts_range do
        stage(:do_rollout, DoRollout)
        # ... advantage computation and training
      end
  """

  use CrucibleKitchen.Stage

  alias CrucibleKitchen.Context
  alias CrucibleTrain.Completers.PortsTokenCompleter
  alias CrucibleTrain.Ports.SamplingClient
  alias CrucibleTrain.RL.{Rollouts, TrajectoryGroup}

  require Logger

  @impl true
  def name, do: :do_rollout

  @impl true
  def execute(context) do
    env_group = Context.get_state(context, :env_group)
    rollout_index = Context.get_state(context, :rollouts_index, 0)
    max_tokens = Context.get_config(context, :max_tokens, 512)
    temperature = Context.get_config(context, :temperature, 1.0)

    with {:ok, context, completer} <- ensure_token_completer(context, max_tokens, temperature) do
      run_rollout(context, env_group, completer, rollout_index)
    end
  end

  @impl true
  def validate(context) do
    case Context.get_state(context, :env_group) do
      nil -> {:error, "env_group is required (run BuildEnvGroup first)"}
      _ -> :ok
    end
  end

  defp run_rollout(context, env_group, completer, rollout_index) do
    Logger.debug("Starting rollout #{rollout_index + 1}")

    start_time = System.monotonic_time(:millisecond)

    trajectory_group = Rollouts.do_group_rollout(env_group, completer)
    duration_ms = System.monotonic_time(:millisecond) - start_time
    rollout_metrics = compute_rollout_metrics(trajectory_group, duration_ms)

    Logger.debug(
      "Rollout #{rollout_index + 1} complete: " <>
        "#{rollout_metrics.num_trajectories} trajectories, " <>
        "reward_mean=#{Float.round(rollout_metrics.reward_mean, 4)}"
    )

    emit_telemetry(rollout_index, rollout_metrics)

    context
    |> Context.put_state(:trajectory_group, trajectory_group)
    |> Context.put_state(:rollout_metrics, rollout_metrics)
    |> then(&{:ok, &1})
  end

  defp ensure_token_completer(context, max_tokens, temperature) do
    case Context.get_state(context, :token_completer) do
      nil ->
        ports = get_train_ports(context)

        case get_sampling_session(context, ports) do
          {:ok, context, session} ->
            completer =
              PortsTokenCompleter.new(
                ports: ports,
                session: session,
                max_tokens: max_tokens,
                temperature: temperature
              )

            context =
              context
              |> Context.put_state(:token_completer, completer)

            {:ok, context, completer}

          {:error, reason} ->
            {:error, reason}
        end

      completer ->
        {:ok, context, completer}
    end
  end

  defp get_sampling_session(context, ports) do
    case Context.get_state(context, :sampling_session) do
      nil ->
        config = sampling_config(context)

        case SamplingClient.start_session(ports, config) do
          {:ok, session} ->
            {:ok, Context.put_state(context, :sampling_session, session), session}

          {:error, reason} ->
            {:error, {:sampling_session_failed, reason}}
        end

      session ->
        {:ok, context, session}
    end
  end

  defp sampling_config(context) do
    %{
      model: Context.get_config(context, :model),
      base_model: Context.get_config(context, :base_model),
      model_path: Context.get_config(context, :sampling_model_path),
      checkpoint_path: Context.get_config(context, :checkpoint_path)
    }
  end

  defp compute_rollout_metrics(%TrajectoryGroup{} = group, duration_ms) do
    rewards = TrajectoryGroup.get_total_rewards(group)
    reward_mean = if rewards == [], do: 0.0, else: Enum.sum(rewards) / length(rewards)
    reward_std = compute_std(rewards, reward_mean)
    num_trajectories = length(group.trajectories_G)

    total_steps =
      Enum.reduce(group.trajectories_G, 0, fn traj, acc ->
        acc + length(traj.transitions)
      end)

    %{
      duration_ms: duration_ms,
      num_trajectories: num_trajectories,
      total_steps: total_steps,
      reward_mean: reward_mean,
      reward_std: reward_std
    }
  end

  defp compute_std([], _mean), do: 0.0

  defp compute_std(values, mean) do
    variance =
      values
      |> Enum.map(fn v -> (v - mean) * (v - mean) end)
      |> Enum.sum()
      |> Kernel./(length(values))

    :math.sqrt(variance)
  end

  defp emit_telemetry(rollout_index, metrics) do
    :telemetry.execute(
      [:crucible_kitchen, :rl, :rollout_complete],
      %{
        duration_ms: metrics.duration_ms,
        num_trajectories: metrics.num_trajectories,
        total_steps: metrics.total_steps,
        reward_mean: metrics.reward_mean,
        reward_std: metrics.reward_std
      },
      %{rollout_index: rollout_index}
    )
  end
end
