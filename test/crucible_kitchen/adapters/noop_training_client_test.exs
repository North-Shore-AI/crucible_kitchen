defmodule CrucibleKitchen.Adapters.Noop.TrainingClientTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.TrainingClient

  test "forward_backward/4 includes loss_fn metadata" do
    session = %{id: "session-1"}
    datums = [%{}, %{}]

    future =
      TrainingClient.forward_backward(
        [],
        session,
        datums,
        loss_fn: :ppo,
        loss_fn_config: %{gamma: 0.99}
      )

    assert future.type == :forward_backward
    assert future.loss_fn == :ppo
    assert future.loss_fn_config == %{gamma: 0.99}
    assert future.batch_size == 2
  end

  test "forward_backward_custom/5 returns a future" do
    session = %{id: "session-2"}
    datums = [%{}]
    loss_fn = fn _data, _logprobs -> {:loss, %{}} end

    future = TrainingClient.forward_backward_custom([], session, datums, loss_fn, foo: :bar)

    assert future.type == :forward_backward_custom
    assert future.loss_fn == loss_fn
    assert future.session_id == "session-2"
    assert future.opts == [foo: :bar]
  end
end
