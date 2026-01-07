defmodule CrucibleKitchen.Adapters.LLMClient.Codex do
  @moduledoc """
  LLM adapter backed by the Codex CLI via codex_sdk.
  """

  @behaviour CrucibleTrain.Ports.LLMClient

  alias CrucibleTrain.Ports.Error

  @impl true
  def chat(adapter_opts, messages, opts) do
    prompt = format_prompt(messages)
    output_schema = Keyword.get(opts, :output_schema)

    options_module = Keyword.get(adapter_opts, :options_module, Codex.Options)

    thread_options_module =
      Keyword.get(adapter_opts, :thread_options_module, Codex.Thread.Options)

    thread_module = Keyword.get(adapter_opts, :thread_module, Codex.Thread)

    codex_opts = Keyword.get(adapter_opts, :codex_opts, [])
    thread_opts = Keyword.get(adapter_opts, :thread_opts, [])
    turn_opts = adapter_opts |> Keyword.get(:turn_opts, %{}) |> Map.new()

    turn_opts =
      if output_schema do
        Map.put(turn_opts, :output_schema, output_schema)
      else
        turn_opts
      end

    with {:ok, codex_opts} <- options_module.new(codex_opts),
         {:ok, thread_opts} <- thread_options_module.new(thread_opts) do
      thread = thread_module.build(codex_opts, thread_opts)

      case thread_module.run(thread, prompt, turn_opts) do
        {:ok, result} ->
          {:ok, result}

        {:error, reason} ->
          {:error, Error.new(:llm_client, __MODULE__, "Codex run failed", reason)}
      end
    else
      {:error, reason} ->
        {:error, Error.new(:llm_client, __MODULE__, "Codex options invalid", reason)}
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
