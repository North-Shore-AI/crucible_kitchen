defmodule CrucibleKitchen.Adapters.VectorStore.Chroma do
  @moduledoc """
  ChromaDB adapter for the VectorStore port.
  """

  @behaviour CrucibleTrain.Ports.VectorStore

  alias CrucibleTrain.Ports.Error

  @impl true
  def create_collection(_opts, name, metadata) do
    wrap(:vector_store, fn -> Chroma.Collection.create(name, metadata) end)
  end

  @impl true
  def get_collection(_opts, name) do
    wrap(:vector_store, fn -> Chroma.Collection.get(name) end)
  end

  @impl true
  def get_or_create_collection(_opts, name, metadata) do
    wrap(:vector_store, fn -> Chroma.Collection.get_or_create(name, metadata) end)
  end

  @impl true
  def delete_collection(_opts, %Chroma.Collection{} = collection) do
    wrap_ok(:vector_store, fn -> Chroma.Collection.delete(collection) end)
  end

  def delete_collection(_opts, name) when is_binary(name) do
    collection = %Chroma.Collection{name: name}
    delete_collection([], collection)
  end

  @impl true
  def add(_opts, %Chroma.Collection{} = collection, payload) when is_map(payload) do
    wrap_ok(:vector_store, fn -> Chroma.Collection.add(collection, payload) end)
  end

  @impl true
  def query(_opts, %Chroma.Collection{} = collection, embeddings, opts) do
    wrap(:vector_store, fn ->
      Chroma.Collection.query(
        collection,
        query_embeddings: embeddings,
        results: Keyword.get(opts, :results, 10),
        where: Keyword.get(opts, :where, %{}),
        where_document: Keyword.get(opts, :where_document, %{}),
        include: Keyword.get(opts, :include, ["metadatas", "documents", "distances"])
      )
    end)
  end

  defp wrap(port, fun) do
    case fun.() do
      {:ok, value} -> {:ok, value}
      {:error, reason} -> {:error, Error.new(port, __MODULE__, "request failed", reason)}
    end
  rescue
    exception ->
      {:error, Error.new(port, __MODULE__, Exception.message(exception), exception)}
  end

  defp wrap_ok(port, fun) do
    case fun.() do
      {:ok, _value} -> :ok
      {:error, reason} -> {:error, Error.new(port, __MODULE__, "request failed", reason)}
      _ -> :ok
    end
  rescue
    exception ->
      {:error, Error.new(port, __MODULE__, Exception.message(exception), exception)}
  end
end
