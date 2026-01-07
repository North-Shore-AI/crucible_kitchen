defmodule CrucibleKitchen.Stages.RLStagesTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context

  alias CrucibleKitchen.Stages.{
    AssembleRLBatch,
    BuildEnvGroup,
    ComputeAdvantages,
    DoRollout,
    LogRLMetrics
  }

  alias CrucibleTrain.Completers.TokenCompleter
  alias CrucibleTrain.RL.{StepResult, Trajectory, TrajectoryGroup, Transition}
  alias CrucibleTrain.Types.{ModelInput, TokensWithLogprobs}

  defmodule TestEnv do
    @behaviour CrucibleTrain.RL.Env

    defstruct [:id]

    def initial_observation(_env) do
      {ModelInput.from_ints([1, 2]), []}
    end

    def step(_env, _action) do
      %StepResult{
        reward: 1.0,
        episode_done: true,
        next_observation: ModelInput.empty(),
        next_stop_condition: [],
        metrics: %{}
      }
    end
  end

  defmodule TestTokenCompleter do
    @behaviour TokenCompleter

    defstruct []

    def complete(_completer, _model_input, _stop) do
      {:ok, %TokensWithLogprobs{tokens: [10, 11], maybe_logprobs: [-1.0, -1.0]}}
    end
  end

  describe "BuildEnvGroup" do
    test "name returns :build_env_group" do
      assert BuildEnvGroup.name() == :build_env_group
    end

    test "stores env_group_builder from config" do
      builder = build_env_group_builder(2)
      context = build_context(%{env_group_builder: builder})

      assert {:ok, result} = BuildEnvGroup.execute(context)
      assert result.state.env_group == builder
    end

    test "builds env_group_builder from context-aware function" do
      builder_fun = fn ctx ->
        group_size = Context.get_config(ctx, :group_size, 1)

        %CrucibleTrain.RL.ProblemGroupBuilder{
          env_thunk: fn -> %TestEnv{} end,
          num_envs: group_size,
          dataset_name: "context_builder"
        }
      end

      context = build_context(%{env_group_builder: builder_fun, group_size: 4})

      assert {:ok, result} = BuildEnvGroup.execute(context)
      assert result.state.env_group.num_envs == 4
      assert result.state.env_group.dataset_name == "context_builder"
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "build-env-group-test",
        [:crucible_kitchen, :rl, :env_group_built],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      builder = build_env_group_builder(3)
      context = build_context(%{env_group_builder: builder})
      {:ok, _} = BuildEnvGroup.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :rl, :env_group_built], measurements, _}
      assert measurements.group_size == 3

      :telemetry.detach("build-env-group-test")
    end
  end

  describe "DoRollout" do
    test "name returns :do_rollout" do
      assert DoRollout.name() == :do_rollout
    end

    test "rolls out trajectories using token completer" do
      builder = build_env_group_builder(2)

      context =
        build_context(%{})
        |> Context.put_state(:env_group, builder)
        |> Context.put_state(:token_completer, %TestTokenCompleter{})
        |> Context.put_state(:rollouts_index, 0)

      assert {:ok, result} = DoRollout.execute(context)
      assert %TrajectoryGroup{} = result.state.trajectory_group
      assert length(result.state.trajectory_group.trajectories_G) == 2
      assert result.state.rollout_metrics.num_trajectories == 2
    end

    test "validation requires env_group" do
      context = build_context(%{})
      assert {:error, _} = DoRollout.validate(context)
    end
  end

  describe "ComputeAdvantages" do
    test "name returns :compute_advantages" do
      assert ComputeAdvantages.name() == :compute_advantages
    end

    test "computes advantages from trajectory group" do
      trajectory_group = build_trajectory_group([1.0, 0.5, 0.0])

      context =
        build_context(%{})
        |> Context.put_state(:trajectory_group, trajectory_group)

      assert {:ok, result} = ComputeAdvantages.execute(context)
      assert length(result.state.advantages) == 3

      mean = Enum.sum(result.state.advantages) / 3
      assert abs(mean) < 1.0e-6
    end

    test "validation requires trajectory_group" do
      context = build_context(%{})
      assert {:error, _} = ComputeAdvantages.validate(context)
    end
  end

  describe "AssembleRLBatch" do
    test "name returns :assemble_rl_batch" do
      assert AssembleRLBatch.name() == :assemble_rl_batch
    end

    test "assembles datums from trajectory group" do
      trajectory_group = build_trajectory_group([1.0])

      context =
        build_context(%{})
        |> Context.put_state(:trajectory_group, trajectory_group)
        |> Context.put_state(:advantages_g, [[0.0]])

      assert {:ok, result} = AssembleRLBatch.execute(context)
      assert [%CrucibleTrain.Types.Datum{}] = result.state.current_batch
      assert result.state.rl_batch_size == 1
    end
  end

  describe "LogRLMetrics" do
    test "name returns :log_rl_metrics" do
      assert LogRLMetrics.name() == :log_rl_metrics
    end

    test "logs metrics and increments step" do
      context =
        build_context(%{})
        |> Context.put_state(:rollout_metrics, %{
          reward_mean: 0.75,
          reward_std: 0.1,
          num_trajectories: 10
        })
        |> Context.put_state(:fb_result, %{metrics: %{loss: 0.1}})
        |> Context.put_state(:global_step, 5)

      assert {:ok, result} = LogRLMetrics.execute(context)
      assert result.state.global_step == 6

      assert Enum.any?(result.metrics, &(&1.name == :rl_reward_mean))
      assert Enum.any?(result.metrics, &(&1.name == :rl_policy_loss))
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "log-rl-metrics-test",
        [:crucible_kitchen, :rl, :step],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context =
        build_context(%{})
        |> Context.put_state(:rollout_metrics, %{reward_mean: 0.5})
        |> Context.put_state(:fb_result, %{metrics: %{loss: 0.2}})
        |> Context.put_state(:global_step, 10)

      {:ok, _} = LogRLMetrics.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :rl, :step], measurements, _}
      assert measurements.reward_mean == 0.5

      :telemetry.detach("log-rl-metrics-test")
    end
  end

  defp build_env_group_builder(group_size) do
    %CrucibleTrain.RL.ProblemGroupBuilder{
      env_thunk: fn -> %TestEnv{} end,
      num_envs: group_size,
      dataset_name: "test"
    }
  end

  defp build_trajectory_group(rewards) do
    trajectories =
      Enum.map(rewards, fn reward ->
        %Trajectory{
          transitions: [
            %Transition{
              ob: ModelInput.from_ints([1]),
              ac: %TokensWithLogprobs{tokens: [2], maybe_logprobs: [-1.0]},
              reward: reward,
              episode_done: true,
              metrics: %{}
            }
          ],
          final_ob: ModelInput.empty()
        }
      end)

    %TrajectoryGroup{
      trajectories_G: trajectories,
      final_rewards_G: List.duplicate(0.0, length(trajectories)),
      metrics_G: List.duplicate(%{}, length(trajectories))
    }
  end

  defp build_context(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        sampling_client: CrucibleKitchen.Adapters.Noop.SamplingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end
end
