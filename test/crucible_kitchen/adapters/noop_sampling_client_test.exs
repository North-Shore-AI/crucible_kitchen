defmodule CrucibleKitchen.Adapters.Noop.SamplingClientTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.SamplingClient

  test "sample returns a task with deterministic tokens" do
    assert {:ok, task} = SamplingClient.sample([], :session, :input, %{max_tokens: 2}, [])
    assert {:ok, response} = Task.await(task, :infinity)

    [sequence | _] = response.sequences
    assert sequence.tokens == [1, 2]
    assert sequence.logprobs == [-1.0, -1.0]
  end
end
