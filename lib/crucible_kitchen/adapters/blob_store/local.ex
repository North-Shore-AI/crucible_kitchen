defmodule CrucibleKitchen.Adapters.BlobStore.Local do
  @moduledoc """
  Local filesystem adapter for BlobStore.
  """

  @behaviour CrucibleTrain.Ports.BlobStore

  alias CrucibleTrain.Ports.Error

  @impl true
  def read(_opts, path) do
    case File.read(path) do
      {:ok, data} -> {:ok, data}
      {:error, reason} -> {:error, Error.new(:blob_store, __MODULE__, "read failed", reason)}
    end
  end

  @impl true
  def stream(_opts, path) do
    {:ok, File.stream!(path)}
  rescue
    exception ->
      {:error, Error.new(:blob_store, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def write(_opts, path, data) do
    case File.write(path, data) do
      :ok -> :ok
      {:error, reason} -> {:error, Error.new(:blob_store, __MODULE__, "write failed", reason)}
    end
  end

  @impl true
  def exists?(_opts, path), do: File.exists?(path)
end
