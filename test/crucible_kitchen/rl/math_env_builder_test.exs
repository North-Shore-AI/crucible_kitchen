defmodule CrucibleKitchen.RL.MathEnvBuilderTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.RL.{MathEnv, MathEnvBuilder}

  defmodule DatasetStoreStub do
    @behaviour CrucibleTrain.Ports.DatasetStore

    def load_dataset(_opts, _repo_id, _load_opts), do: {:ok, :dataset}
    def get_split(_opts, dataset, _split), do: {:ok, dataset}
    def shuffle(_opts, dataset, _opts2), do: {:ok, dataset}
    def take(_opts, dataset, _count), do: {:ok, dataset}
    def skip(_opts, dataset, _count), do: {:ok, dataset}
    def select(_opts, dataset, _selection), do: {:ok, dataset}

    def to_list(_opts, _dataset) do
      {:ok, [%{"question" => "2+2", "answer" => "4"}]}
    end
  end

  test "build/1 returns a ProblemGroupBuilder that yields math envs" do
    tokenizer = %{
      encode: fn text, _opts -> String.to_charlist(text) end,
      decode: fn tokens -> List.to_string(tokens) end
    }

    config = %{model: "meta-llama/Llama-3.1-8B", group_size: 2, seed: 0}
    adapters = %{dataset_store: {DatasetStoreStub, []}}

    context =
      Context.new(config, adapters)
      |> Context.put_state(:dataset_handle, :dataset)
      |> Context.put_state(:tokenizer, tokenizer)

    builder = MathEnvBuilder.build(context)
    assert %CrucibleTrain.RL.ProblemGroupBuilder{} = builder

    envs = builder.__struct__.make_envs(builder)
    assert length(envs) == 2
    assert %MathEnv{} = hd(envs)
    assert MathEnv.get_question(hd(envs)) == "2+2 Write your answer in \\boxed{} format."
  end
end
