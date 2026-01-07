defmodule CrucibleKitchen.Adapters.Noop.EmbeddingClient do
  @moduledoc """
  No-op adapter for embedding generation.
  """

  @behaviour CrucibleTrain.Ports.EmbeddingClient

  alias CrucibleTrain.Ports.Error

  defp error do
    Error.new(:embedding_client, __MODULE__, "Embedding adapter is not configured")
  end

  @impl true
  def embed_texts(_opts, _texts, _opts2), do: {:error, error()}
end
