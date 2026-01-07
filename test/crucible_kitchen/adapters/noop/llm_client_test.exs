defmodule CrucibleKitchen.Adapters.Noop.LLMClientTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.LLMClient
  alias CrucibleTrain.Ports.Error

  test "chat returns configuration error" do
    assert {:error, %Error{port: :llm_client, adapter: LLMClient}} =
             LLMClient.chat([], [%{role: "user", content: "hi"}], [])
  end
end
