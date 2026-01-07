defmodule CrucibleKitchen.Adapters.SnakebridgeTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Snakebridge
  alias CrucibleKitchen.Adapters.Snakebridge.MathVerify
  alias CrucibleKitchen.TestSupport.SnakebridgeRuntimeStub

  setup do
    Application.put_env(:snakebridge, :runtime_client, SnakebridgeRuntimeStub)
    Process.put(:snakebridge_runtime_pid, self())

    on_exit(fn ->
      Application.delete_env(:snakebridge, :runtime_client)
      Process.delete(:snakebridge_runtime_pid)
      Process.delete(:snakebridge_runtime_responder)
    end)

    :ok
  end

  test "call passes timeout to runtime opts" do
    Process.put(:snakebridge_runtime_responder, fn _cmd, _payload, _opts -> {:ok, :ok} end)

    assert {:ok, :ok} = Snakebridge.call(MathVerify, :parse, ["$x$"], timeout_ms: 1234)

    assert_received {:snakebridge_execute, "snakebridge.call", _payload, opts}
    assert Keyword.get(opts, :timeout) == 1234
  end

  test "call wraps snakepit errors" do
    Process.put(:snakebridge_runtime_responder, fn _cmd, _payload, _opts ->
      {:error, Snakepit.Error.timeout_error("boom")}
    end)

    assert {:error, {:snakebridge_error, %Snakepit.Error{category: :timeout}}} =
             Snakebridge.call(MathVerify, :parse, ["$x$"])
  end

  test "math_verify wraps inputs and verifies" do
    Process.put(:snakebridge_runtime_responder, fn
      "snakebridge.call", %{"function" => "parse", "args" => [arg]}, _opts ->
        {:ok, {:parsed, arg}}

      "snakebridge.call", %{"function" => "verify", "args" => [given, ground]}, _opts ->
        {:ok, given == {:parsed, "$2+2$"} and ground == {:parsed, "$4$"}}
    end)

    assert {:ok, true} = MathVerify.verify("2+2", "4")

    assert_received {:snakebridge_execute, "snakebridge.call",
                     %{"function" => "parse", "args" => ["$2+2$"]}, _opts}

    assert_received {:snakebridge_execute, "snakebridge.call",
                     %{"function" => "parse", "args" => ["$4$"]}, _opts}

    assert_received {:snakebridge_execute, "snakebridge.call",
                     %{"function" => "verify", "args" => [_, _]}, _opts}
  end

  test "math_verify preserves existing dollar delimiters" do
    Process.put(:snakebridge_runtime_responder, fn
      "snakebridge.call", %{"function" => "parse", "args" => [arg]}, _opts ->
        {:ok, {:parsed, arg}}

      "snakebridge.call", %{"function" => "verify"}, _opts ->
        {:ok, true}
    end)

    assert {:ok, true} = MathVerify.verify("$x$", "$x$")

    assert_received {:snakebridge_execute, "snakebridge.call",
                     %{"function" => "parse", "args" => ["$x$"]}, _opts}

    assert_received {:snakebridge_execute, "snakebridge.call",
                     %{"function" => "parse", "args" => ["$x$"]}, _opts}
  end
end
