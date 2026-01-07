defmodule CrucibleKitchen.RL.NoopEnv do
  @moduledoc """
  Minimal no-op environment for RL workflows.
  """

  @behaviour CrucibleTrain.RL.Env

  alias CrucibleTrain.RL.StepResult
  alias CrucibleTrain.Types.ModelInput

  defstruct observation_tokens: [1], reward: 0.0

  @impl true
  def initial_observation(%__MODULE__{observation_tokens: tokens}) do
    {ModelInput.from_ints(tokens), []}
  end

  @impl true
  def step(%__MODULE__{reward: reward}, _action) do
    %StepResult{
      reward: reward,
      episode_done: true,
      next_observation: ModelInput.empty(),
      next_stop_condition: [],
      metrics: %{}
    }
  end
end
