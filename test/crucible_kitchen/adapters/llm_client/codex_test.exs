defmodule CrucibleKitchen.Adapters.LLMClient.CodexTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.LLMClient.Codex

  defmodule OptionsStub do
    def new(opts) do
      send(self(), {:codex_options_new, opts})
      {:ok, {:codex_opts, opts}}
    end
  end

  defmodule ThreadOptionsStub do
    def new(opts) do
      send(self(), {:codex_thread_options_new, opts})
      {:ok, {:thread_opts, opts}}
    end
  end

  defmodule ThreadStub do
    def build(codex_opts, thread_opts) do
      send(self(), {:codex_thread_build, codex_opts, thread_opts})
      :thread
    end

    def run(:thread, prompt, turn_opts) do
      send(self(), {:codex_thread_run, prompt, turn_opts})
      {:ok, %{prompt: prompt, turn_opts: turn_opts}}
    end
  end

  test "formats messages and forwards output_schema" do
    schema = %{"type" => "object"}

    adapter_opts = [
      options_module: OptionsStub,
      thread_options_module: ThreadOptionsStub,
      thread_module: ThreadStub,
      codex_opts: [api_key: "test"],
      thread_opts: [sandbox: :read_only],
      turn_opts: %{temperature: 0.0}
    ]

    messages = [%{role: "user", content: "Summarize"}]

    assert {:ok, result} = Codex.chat(adapter_opts, messages, output_schema: schema)

    assert_received {:codex_options_new, [api_key: "test"]}
    assert_received {:codex_thread_options_new, [sandbox: :read_only]}

    assert_received {:codex_thread_build, {:codex_opts, [api_key: "test"]},
                     {:thread_opts, [sandbox: :read_only]}}

    assert_received {:codex_thread_run, "User: Summarize",
                     %{output_schema: ^schema, temperature: temperature}}

    assert temperature == 0.0

    assert result == %{
             prompt: "User: Summarize",
             turn_opts: %{output_schema: schema, temperature: temperature}
           }
  end
end
