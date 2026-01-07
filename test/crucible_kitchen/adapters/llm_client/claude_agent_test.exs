defmodule CrucibleKitchen.Adapters.LLMClient.ClaudeAgentTest do
  use ExUnit.Case, async: true

  alias ClaudeAgentSDK.Message
  alias CrucibleKitchen.Adapters.LLMClient.ClaudeAgent

  defmodule QueryStub do
    def run(prompt, options) do
      send(self(), {:claude_query, prompt, options})

      [
        %Message{
          type: :assistant,
          subtype: nil,
          data: %{message: %{"content" => "Summary"}}
        },
        %Message{
          type: :result,
          subtype: :success,
          data: %{session_id: "session-1"}
        }
      ]
    end
  end

  test "formats prompt and maps output_schema to output_format" do
    schema = %{"type" => "object"}
    options = %ClaudeAgentSDK.Options{}

    adapter_opts = [
      query_module: QueryStub,
      options: options
    ]

    messages = [%{role: "user", content: "Summarize"}]

    assert {:ok, %{content: "Summary", messages: results}} =
             ClaudeAgent.chat(adapter_opts, messages, output_schema: schema)

    assert_received {:claude_query, "User: Summarize",
                     %ClaudeAgentSDK.Options{output_format: {:json_schema, ^schema}}}

    assert length(results) == 2
  end
end
