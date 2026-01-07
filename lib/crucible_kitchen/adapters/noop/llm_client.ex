defmodule CrucibleKitchen.Adapters.Noop.LLMClient do
  @moduledoc """
  No-op adapter for LLM inference.
  """

  @behaviour CrucibleTrain.Ports.LLMClient

  alias CrucibleTrain.Ports.Error

  @impl true
  def chat(_opts, _messages, _opts2) do
    {:error, Error.new(:llm_client, __MODULE__, "LLM adapter is not configured")}
  end
end
