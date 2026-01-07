defmodule CrucibleKitchen.RL.MathEnv do
  @moduledoc """
  Problem environment for math datasets (GSM8K-style).

  Uses \\boxed{} formatted answers and delegates equivalence checking
  to a verifier module (defaults to Snakebridge math_verify).
  """

  use CrucibleTrain.RL.ProblemEnv

  alias CrucibleTrain.Renderers.Types

  @default_suffix " Write your answer in \\boxed{} format."

  defstruct [
    :problem,
    :answer,
    :renderer_module,
    :renderer_state,
    :question_suffix,
    :verifier,
    convo_prefix: nil,
    format_coef: 0.1,
    verifier_opts: []
  ]

  @impl true
  def get_question(%__MODULE__{problem: problem} = env) do
    problem <> question_suffix(env)
  end

  @impl true
  def check_format(_env, sample_str) do
    match?({:ok, _}, extract_boxed(sample_str))
  end

  @impl true
  def check_answer(%__MODULE__{} = env, sample_str) do
    case extract_boxed(sample_str) do
      {:ok, boxed} -> verify_answer(env, boxed)
      _ -> false
    end
  end

  @impl true
  def get_reference_answer(%__MODULE__{answer: answer}), do: answer

  @doc """
  Standard few-shot prefix used in math RL.
  """
  @spec standard_fewshot_prefix() :: [Types.Message.t()]
  def standard_fewshot_prefix do
    [
      Types.message("user", "How many r's are in strawberry?" <> @default_suffix),
      Types.message(
        "assistant",
        "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}"
      )
    ]
  end

  @doc """
  Extract the content inside \\boxed{...}.
  """
  @spec extract_boxed(String.t()) :: {:ok, String.t()} | :error
  def extract_boxed(text) when is_binary(text) do
    case Regex.run(~r/\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/, text) do
      [_, answer] -> {:ok, String.trim(answer)}
      nil -> :error
    end
  end

  def extract_boxed(_), do: :error

  defp question_suffix(%__MODULE__{question_suffix: suffix}) when is_binary(suffix), do: suffix
  defp question_suffix(_env), do: @default_suffix

  defp verify_answer(%__MODULE__{verifier: nil, answer: answer}, given) do
    normalize_result(given == answer)
  end

  defp verify_answer(%__MODULE__{verifier: verifier, verifier_opts: opts, answer: answer}, given)
       when is_atom(verifier) do
    result =
      cond do
        function_exported?(verifier, :verify, 3) -> verifier.verify(given, answer, opts)
        function_exported?(verifier, :verify, 2) -> verifier.verify(given, answer)
        true -> false
      end

    normalize_result(result)
  end

  defp verify_answer(%__MODULE__{verifier: verifier, answer: answer}, given)
       when is_function(verifier, 2) do
    verifier.(given, answer)
    |> normalize_result()
  end

  defp normalize_result({:ok, value}) when is_boolean(value), do: value
  defp normalize_result(value) when is_boolean(value), do: value
  defp normalize_result(_), do: false
end

defmodule CrucibleKitchen.RL.MathEnvBuilder do
  @moduledoc """
  Builds math problem environments from dataset samples.
  """

  alias CrucibleKitchen.Context
  alias CrucibleKitchen.RL.MathEnv
  alias CrucibleTrain.Ports.DatasetStore
  alias CrucibleTrain.Renderers.Types
  alias CrucibleTrain.RL.ProblemGroupBuilder
  alias CrucibleTrain.Utils.PRNG.PCG64

  require Logger

  @spec build(Context.t()) :: ProblemGroupBuilder.t()
  def build(%Context{} = context) do
    dataset_handle = Context.get_state(context, :dataset_handle)
    tokenizer = Context.get_state(context, :tokenizer)
    group_size = Context.get_config(context, :group_size, 4)
    seed = Context.get_config(context, :seed, 0)
    dataset_label = Context.get_config(context, :dataset, :math) |> to_string()
    format_coef = Context.get_config(context, :format_coef, 0.1)
    question_suffix = Context.get_config(context, :question_suffix)

    verifier =
      Context.get_config(
        context,
        :math_verify_module,
        CrucibleKitchen.Adapters.Snakebridge.MathVerify
      )

    verifier_opts = build_verifier_opts(context)

    renderer_module = renderer_for_model(Context.get_config(context, :model))

    ports = Context.get_train_ports(context)

    with {:ok, samples} <- DatasetStore.to_list(ports, dataset_handle),
         {:ok, renderer_state} <- renderer_module.init(tokenizer: tokenizer) do
      builder_ref = make_ref()

      env_thunk = fn ->
        sample = pick_sample(samples, seed, builder_ref)
        {problem, answer} = extract_problem(sample)

        %MathEnv{
          problem: problem,
          answer: answer,
          renderer_module: renderer_module,
          renderer_state: renderer_state,
          convo_prefix: build_convo_prefix(context),
          format_coef: format_coef,
          question_suffix: question_suffix,
          verifier: verifier,
          verifier_opts: verifier_opts
        }
      end

      %ProblemGroupBuilder{
        env_thunk: env_thunk,
        num_envs: group_size,
        dataset_name: dataset_label
      }
    else
      {:error, reason} ->
        Logger.error("[MathEnvBuilder] Failed to build: #{inspect(reason)}")
        raise "MathEnvBuilder failed: #{inspect(reason)}"
    end
  end

  defp build_convo_prefix(context) do
    case Context.get_config(context, :convo_prefix, :standard) do
      :standard ->
        MathEnv.standard_fewshot_prefix()

      messages when is_list(messages) ->
        Enum.map(messages, &normalize_message/1)

      _ ->
        nil
    end
  end

  defp normalize_message(%Types.Message{} = message), do: message
  defp normalize_message(%{role: role, content: content}), do: Types.message(role, content)

  defp normalize_message(%{"role" => role, "content" => content}),
    do: Types.message(role, content)

  defp normalize_message(message), do: message

  defp pick_sample(samples, seed, ref) do
    if samples == [] do
      raise "MathEnvBuilder received empty dataset"
    end

    key = {__MODULE__, ref}
    state = Process.get(key) || PCG64.seed(seed)
    {idx, next_state} = PCG64.random_interval(state, length(samples) - 1)
    Process.put(key, next_state)

    Enum.at(samples, idx)
  end

  defp extract_problem(sample) do
    question =
      first_string([
        get_in(sample, ["question"]),
        get_in(sample, [:question]),
        get_in(sample, ["input", "question"]),
        get_in(sample, [:input, :question]),
        get_in(sample, ["input", "problem"]),
        get_in(sample, [:input, :problem]),
        get_in(sample, ["prompt"]),
        get_in(sample, [:prompt]),
        get_in(sample, ["instruction"]),
        get_in(sample, [:instruction])
      ]) || ""

    answer = extract_answer(sample)
    {question, answer}
  end

  defp extract_answer(sample) do
    answer =
      Enum.find_value(
        [
          get_in(sample, ["answer"]),
          get_in(sample, [:answer]),
          get_in(sample, ["expected", "answer"]),
          get_in(sample, [:expected, :answer]),
          get_in(sample, ["solution"]),
          get_in(sample, [:solution]),
          get_in(sample, ["expected", "reasoning"]),
          get_in(sample, [:expected, :reasoning])
        ],
        & &1
      )

    cond do
      is_number(answer) -> format_number(answer)
      is_binary(answer) -> extract_final_answer(answer) || answer
      true -> ""
    end
  end

  defp first_string(values) do
    Enum.find(values, &is_binary/1)
  end

  defp extract_final_answer(text) do
    lines = String.split(text, "\n", trim: true)

    line =
      Enum.find(Enum.reverse(lines), fn entry ->
        String.trim_leading(entry) |> String.starts_with?("####")
      end)

    case line do
      nil ->
        nil

      value ->
        value
        |> String.trim()
        |> String.trim_leading("####")
        |> String.trim()
        |> String.trim_leading(":")
        |> String.trim()
        |> String.replace(",", "")
    end
  end

  defp format_number(value) when is_integer(value), do: Integer.to_string(value)

  defp format_number(value) when is_float(value) do
    if Float.floor(value) == value do
      value |> trunc() |> Integer.to_string()
    else
      Float.to_string(value)
    end
  end

  defp build_verifier_opts(context) do
    timeout_ms = Context.get_config(context, :math_verify_timeout_ms)
    if timeout_ms, do: [timeout_ms: timeout_ms], else: []
  end

  defp renderer_for_model(model_name) when is_binary(model_name) do
    cond do
      String.contains?(model_name, "Llama-3") -> CrucibleTrain.Renderers.Llama3
      String.contains?(model_name, "Qwen") -> CrucibleTrain.Renderers.Qwen3
      String.contains?(model_name, "DeepSeek") -> CrucibleTrain.Renderers.DeepSeekV3
      true -> CrucibleTrain.Renderers.RoleColon
    end
  end

  defp renderer_for_model(_), do: CrucibleTrain.Renderers.RoleColon
end
