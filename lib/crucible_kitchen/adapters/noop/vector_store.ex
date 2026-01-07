defmodule CrucibleKitchen.Adapters.Noop.VectorStore do
  @moduledoc """
  No-op adapter for vector store operations.
  """

  @behaviour CrucibleTrain.Ports.VectorStore

  alias CrucibleTrain.Ports.Error

  defp error do
    Error.new(:vector_store, __MODULE__, "Vector store adapter is not configured")
  end

  @impl true
  def create_collection(_opts, _name, _metadata), do: {:error, error()}

  @impl true
  def get_collection(_opts, _name), do: {:error, error()}

  @impl true
  def get_or_create_collection(_opts, _name, _metadata), do: {:error, error()}

  @impl true
  def delete_collection(_opts, _collection_or_name), do: {:error, error()}

  @impl true
  def add(_opts, _collection, _payload), do: {:error, error()}

  @impl true
  def query(_opts, _collection, _embeddings, _opts2), do: {:error, error()}
end
