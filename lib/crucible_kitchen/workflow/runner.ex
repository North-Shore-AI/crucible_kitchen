defmodule CrucibleKitchen.Workflow.Runner do
  @moduledoc """
  Executes compiled workflows.

  The runner takes a workflow module and initial context, then executes
  each stage in order, handling control flow constructs like loops,
  conditionals, and parallel blocks.
  """

  require Logger

  alias CrucibleKitchen.Context

  @doc """
  Run a workflow with the given initial context.
  """
  @spec run(module(), Context.t()) :: {:ok, Context.t()} | {:error, term()}
  def run(workflow_module, initial_context) do
    workflow_ir = build_workflow(workflow_module)

    Logger.debug("Starting workflow: #{inspect(workflow_module)}")

    execute_nodes(workflow_ir, initial_context)
  end

  # Build workflow IR from module
  defp build_workflow(workflow_module) do
    raw_stages = workflow_module.__workflow__()
    parse_dsl(raw_stages, [])
  end

  # Parse DSL into IR
  defp parse_dsl([], acc), do: Enum.reverse(acc)

  defp parse_dsl([{:stage, name, module, opts} | rest], acc) do
    node = {:stage, name, module, opts}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:loop_start, name, opts} | rest], acc) do
    {body, rest} = collect_until(:loop_end, name, rest, [])
    iterator = Keyword.fetch!(opts, :over)
    node = {:loop, name, iterator, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:conditional_start, predicate} | rest], acc) do
    {body, rest} = collect_until(:conditional_end, nil, rest, [])
    node = {:conditional, predicate, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([{:parallel_start, opts} | rest], acc) do
    {body, rest} = collect_until(:parallel_end, nil, rest, [])
    node = {:parallel, opts, parse_dsl(body, [])}
    parse_dsl(rest, [node | acc])
  end

  defp parse_dsl([_unknown | rest], acc) do
    # Skip unknown nodes
    parse_dsl(rest, acc)
  end

  # Collect nodes until end marker
  defp collect_until(end_type, name, [{end_type, name} | rest], acc) do
    {Enum.reverse(acc), rest}
  end

  defp collect_until(end_type, nil, [{end_type} | rest], acc) do
    {Enum.reverse(acc), rest}
  end

  defp collect_until(end_type, name, [node | rest], acc) do
    collect_until(end_type, name, rest, [node | acc])
  end

  defp collect_until(_end_type, _name, [], acc) do
    # Unclosed block - return what we have
    {Enum.reverse(acc), []}
  end

  # Execute IR nodes
  defp execute_nodes([], context), do: {:ok, context}

  defp execute_nodes([node | rest], context) do
    case execute_node(node, context) do
      {:ok, new_context} -> execute_nodes(rest, new_context)
      {:error, reason} -> {:error, reason}
    end
  end

  defp execute_node({:stage, name, module, opts}, context) do
    context = %{context | current_stage: name, stage_opts: opts}

    Logger.debug("Executing stage: #{name}")

    :telemetry.span(
      [:crucible_kitchen, :stage, :run],
      %{stage: name, module: module},
      fn ->
        with :ok <- validate_stage(module, context),
             {:ok, new_context} <- execute_stage(module, context) do
          {{:ok, new_context}, %{stage: name, success: true}}
        else
          {:error, reason} ->
            Logger.error("Stage #{name} failed: #{inspect(reason)}")
            {{:error, {name, reason}}, %{stage: name, success: false, error: reason}}
        end
      end
    )
  end

  defp execute_node({:loop, name, iterator, body}, context) do
    items =
      case iterator do
        fun when is_function(fun, 1) -> fun.(context)
        enumerable -> enumerable
      end

    Logger.debug("Starting loop: #{name} with #{Enum.count(items)} items")

    Enum.reduce_while(items, {:ok, context}, fn item, {:ok, ctx} ->
      ctx = Context.put_state(ctx, :"#{name}_current", item)

      case execute_nodes(body, ctx) do
        {:ok, new_ctx} -> {:cont, {:ok, new_ctx}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  defp execute_node({:conditional, predicate, body}, context) do
    should_execute =
      case predicate do
        fun when is_function(fun, 1) -> fun.(context)
        true -> true
        false -> false
      end

    if should_execute do
      Logger.debug("Conditional: executing")
      execute_nodes(body, context)
    else
      Logger.debug("Conditional: skipping")
      {:ok, context}
    end
  end

  defp execute_node({:parallel, opts, body}, context) do
    max_concurrency =
      case Keyword.get(opts, :max_concurrency) do
        nil -> System.schedulers_online()
        fun when is_function(fun, 1) -> fun.(context)
        n when is_integer(n) -> n
      end

    Logger.debug(
      "Parallel: executing #{length(body)} stages with max_concurrency=#{max_concurrency}"
    )

    # For now, execute sequentially (parallel execution is a future enhancement)
    # TODO: Implement actual parallel execution with Task.async_stream
    execute_nodes(body, context)
  end

  defp validate_stage(module, context) do
    if function_exported?(module, :validate, 1) do
      module.validate(context)
    else
      :ok
    end
  end

  defp execute_stage(module, context) do
    module.execute(context)
  rescue
    e ->
      if function_exported?(module, :rollback, 2) do
        context = module.rollback(context, e)
        {:error, {e, context}}
      else
        reraise e, __STACKTRACE__
      end
  end
end
