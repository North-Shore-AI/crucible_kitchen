defmodule CrucibleKitchen.Adapters.Noop.SamplingClient do
  @moduledoc """
  Noop adapter for SamplingClient port.
  """

  @behaviour CrucibleTrain.Ports.SamplingClient

  @impl true
  def start_session(_opts, config) do
    session = %{
      id: generate_id(),
      model: Map.get(config, :model) || Map.get(config, "model") || "noop-model",
      config: config
    }

    {:ok, session}
  end

  @impl true
  def sample(_opts, _session, _model_input, params, _opts_kw) do
    max_tokens = Map.get(params, :max_tokens, 2)
    tokens = Enum.to_list(1..max_tokens)
    logprobs = List.duplicate(-1.0, max_tokens)

    response = %{
      sequences: [
        %{
          tokens: tokens,
          text: "noop",
          logprobs: logprobs,
          stop_reason: "end_of_turn"
        }
      ],
      type: "sample"
    }

    {:ok, Task.async(fn -> {:ok, response} end)}
  end

  @impl true
  def sample_stream(_opts, _session, _model_input, params, _opts_kw) do
    max_tokens = Map.get(params, :max_tokens, 2)

    stream =
      1..max_tokens
      |> Stream.map(fn token ->
        %{token: token, text: Integer.to_string(token)}
      end)

    {:ok, stream}
  end

  @impl true
  def compute_logprobs(_opts, _session, _model_input, _opts_kw) do
    {:ok, Task.async(fn -> {:ok, [-1.0]} end)}
  end

  @impl true
  def await(_opts, %Task{} = task), do: Task.await(task, :infinity)
  def await(_opts, result), do: {:ok, result}

  @impl true
  def close_session(_opts, _session), do: :ok

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
