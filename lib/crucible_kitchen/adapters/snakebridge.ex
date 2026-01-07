defmodule CrucibleKitchen.Adapters.Snakebridge do
  @moduledoc """
  Thin adapter for SnakeBridge runtime calls with timeout and error wrapping.
  """

  @type call_opts :: [
          timeout_ms: pos_integer() | nil,
          runtime_opts: keyword()
        ]

  @spec call(module(), atom() | String.t(), list(), call_opts()) ::
          {:ok, term()} | {:error, {:snakebridge_error, term()}}
  def call(module, function, args \\ [], opts \\ []) do
    {runtime_opts, passthrough_opts} = runtime_opts(opts)

    module
    |> SnakeBridge.Runtime.call(function, args, passthrough_opts ++ runtime_opts)
    |> wrap_result()
  end

  @spec call_helper(String.t(), list(), call_opts()) ::
          {:ok, term()} | {:error, {:snakebridge_error, term()}}
  def call_helper(helper, args \\ [], opts \\ []) do
    {runtime_opts, passthrough_opts} = runtime_opts(opts)

    helper
    |> SnakeBridge.Runtime.call_helper(args, passthrough_opts ++ runtime_opts)
    |> wrap_result()
  end

  defp runtime_opts(opts) do
    {timeout_ms, opts} = Keyword.pop(opts, :timeout_ms)
    {runtime_opts, opts} = Keyword.pop(opts, :runtime_opts, [])
    runtime_opts = runtime_opts |> add_timeout(timeout_ms)

    if runtime_opts == [] do
      {[], opts}
    else
      {[__runtime__: runtime_opts], opts}
    end
  end

  defp add_timeout(runtime_opts, nil), do: runtime_opts

  defp add_timeout(runtime_opts, timeout_ms) do
    Keyword.put_new(runtime_opts, :timeout, timeout_ms)
  end

  defp wrap_result({:ok, result}), do: {:ok, result}
  defp wrap_result({:error, reason}), do: {:error, {:snakebridge_error, reason}}
end
