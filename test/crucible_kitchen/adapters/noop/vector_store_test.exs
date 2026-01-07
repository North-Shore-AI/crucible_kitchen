defmodule CrucibleKitchen.Adapters.Noop.VectorStoreTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.VectorStore
  alias CrucibleTrain.Ports.Error

  test "create_collection returns configuration error" do
    assert {:error, %Error{port: :vector_store, adapter: VectorStore}} =
             VectorStore.create_collection([], "test", %{})
  end
end
