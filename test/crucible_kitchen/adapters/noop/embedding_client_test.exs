defmodule CrucibleKitchen.Adapters.Noop.EmbeddingClientTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.EmbeddingClient
  alias CrucibleTrain.Ports.Error

  test "embed_texts returns configuration error" do
    assert {:error, %Error{port: :embedding_client, adapter: EmbeddingClient}} =
             EmbeddingClient.embed_texts([], ["hello"], [])
  end
end
