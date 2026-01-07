defmodule CrucibleKitchen.Adapters.LLMClient.ClaudeAgent do
  @moduledoc """
  LLM adapter backed by the Claude Code CLI via claude_agent_sdk.
  """

  @behaviour CrucibleTrain.Ports.LLMClient

  alias ClaudeAgentSDK.{ContentExtractor, Message, Options}
  alias CrucibleTrain.Ports.Error

  @impl true
  def chat(adapter_opts, messages, opts) do
    prompt = format_prompt(messages)
    output_schema = Keyword.get(opts, :output_schema)
    query_module = Keyword.get(adapter_opts, :query_module, ClaudeAgentSDK.Query)

    options =
      adapter_opts
      |> Keyword.get(:options)
      |> build_options()
      |> maybe_put_output_schema(output_schema)

    try do
      results = query_module.run(prompt, options) |> Enum.to_list()
      content = extract_assistant_text(results)
      {:ok, %{content: content, messages: results}}
    rescue
      error ->
        {:error, Error.new(:llm_client, __MODULE__, "Claude query failed", error)}
    end
  end

  defp build_options(nil), do: %Options{}
  defp build_options(%Options{} = options), do: options
  defp build_options(options) when is_list(options), do: options |> Map.new() |> build_options()
  defp build_options(options) when is_map(options), do: struct(Options, options)

  defp maybe_put_output_schema(options, nil), do: options

  defp maybe_put_output_schema(%Options{} = options, schema) do
    %{options | output_format: {:json_schema, schema}}
  end

  defp extract_assistant_text(messages) when is_list(messages) do
    messages
    |> Enum.reverse()
    |> Enum.find(&match?(%Message{type: :assistant}, &1))
    |> case do
      nil -> nil
      message -> ContentExtractor.extract_text(message)
    end
  end

  defp format_prompt(messages) when is_list(messages) do
    messages
    |> Enum.map(&format_message/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("\n")
  end

  defp format_prompt(message), do: format_message(message)

  defp format_message(%{role: role, content: content}) do
    role =
      role
      |> to_string()
      |> String.trim()
      |> String.capitalize()

    content = to_string(content)

    if role == "" do
      content
    else
      "#{role}: #{content}"
    end
  end

  defp format_message(message) when is_binary(message), do: message
  defp format_message(message) when is_atom(message), do: Atom.to_string(message)
  defp format_message(message), do: inspect(message)
end
