defmodule CrucibleKitchen.TestSupport.SnakebridgeRuntimeStub do
  @moduledoc false

  def execute(command, payload, opts) do
    notify({:snakebridge_execute, command, payload, opts})

    case Process.get(:snakebridge_runtime_responder) do
      responder when is_function(responder, 3) -> responder.(command, payload, opts)
      _ -> {:ok, payload}
    end
  end

  def execute_stream(command, payload, callback, opts) do
    notify({:snakebridge_execute_stream, command, payload, opts})
    callback.(:eof)
    :ok
  end

  defp notify(message) do
    pid = Process.get(:snakebridge_runtime_pid) || self()
    send(pid, message)
  end
end
