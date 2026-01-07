defmodule CrucibleKitchen.Adapters.Snakebridge.MathVerify do
  @moduledoc """
  Wrapper for math_verify parse/verify helpers via SnakeBridge.
  """

  alias CrucibleKitchen.Adapters.Snakebridge

  @default_timeout_ms 1000

  @spec verify(String.t(), String.t(), keyword()) ::
          {:ok, boolean()} | {:error, {:snakebridge_error, term()}}
  def verify(given_answer, ground_truth, opts \\ []) do
    opts = Keyword.put_new(opts, :timeout_ms, @default_timeout_ms)
    given_answer = wrap_if_needed(given_answer)
    ground_truth = wrap_if_needed(ground_truth)

    with {:ok, given_parsed} <- Snakebridge.call(__MODULE__, :parse, [given_answer], opts),
         {:ok, ground_parsed} <- Snakebridge.call(__MODULE__, :parse, [ground_truth], opts) do
      Snakebridge.call(__MODULE__, :verify, [given_parsed, ground_parsed], opts)
    end
  end

  @spec reward(String.t(), String.t(), keyword()) ::
          {:ok, float()} | {:error, {:snakebridge_error, term()}}
  def reward(given_answer, ground_truth, opts \\ []) do
    case verify(given_answer, ground_truth, opts) do
      {:ok, true} -> {:ok, 1.0}
      {:ok, false} -> {:ok, 0.0}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc false
  def __snakebridge_python_name__, do: "math_verify"

  @doc false
  def __snakebridge_library__, do: "math_verify"

  defp wrap_if_needed(value) do
    if String.starts_with?(value, "$") or String.ends_with?(value, "$") do
      value
    else
      "$" <> value <> "$"
    end
  end
end
