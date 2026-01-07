defmodule CrucibleKitchen.Ports.AdapterContractTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Ports

  test "validate accepts noop adapters for training and LLM ports" do
    adapters = %{
      training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
      sampling_client: CrucibleKitchen.Adapters.Noop.SamplingClient,
      dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore,
      blob_store: CrucibleKitchen.Adapters.Noop.BlobStore,
      hub_client: CrucibleKitchen.Adapters.Noop.HubClient,
      llm_client: CrucibleKitchen.Adapters.Noop.LLMClient,
      embedding_client: CrucibleKitchen.Adapters.Noop.EmbeddingClient,
      vector_store: CrucibleKitchen.Adapters.Noop.VectorStore
    }

    assert :ok =
             Ports.validate(adapters, [
               :training_client,
               :sampling_client,
               :dataset_store,
               :blob_store,
               :hub_client,
               :llm_client,
               :embedding_client,
               :vector_store
             ])
  end
end
