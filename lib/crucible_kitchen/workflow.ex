defmodule CrucibleKitchen.Workflow do
  @moduledoc """
  DSL for defining training workflows.

  Workflows are compositions of stages with control flow constructs.
  They describe HOW training is orchestrated, not WHAT the stages do.

  ## Example

      defmodule MyWorkflow do
        use CrucibleKitchen.Workflow

        workflow do
          stage :load, LoadStage
          stage :init, InitStage

          loop :epochs, over: fn ctx -> 0..(ctx.config.epochs - 1) end do
            stage :train_epoch, TrainEpochStage

            conditional fn ctx -> should_eval?(ctx) end do
              stage :eval, EvalStage
            end
          end

          stage :save, SaveStage
        end
      end

  ## DSL Reference

  ### stage(name, module, opts \\\\ [])

  Define a stage to execute.

  - `name` - Stage identifier (atom)
  - `module` - Module implementing `CrucibleKitchen.Stage`
  - `opts` - Options passed to the stage

  ### loop(name, opts, do: block)

  Iterate over a collection.

  - `name` - Loop identifier
  - `opts[:over]` - Function `(context) -> Enumerable.t()`

  ### conditional(predicate, do: block)

  Execute block only if predicate returns true.

  - `predicate` - Function `(context) -> boolean()`

  ### parallel(opts \\\\ [], do: block)

  Execute stages in parallel.

  - `opts[:max_concurrency]` - Max concurrent stages (default: schedulers_online)
  """

  @doc false
  defmacro __using__(_opts) do
    quote do
      import CrucibleKitchen.Workflow.DSL

      Module.register_attribute(__MODULE__, :workflow_stages, accumulate: true)

      @before_compile CrucibleKitchen.Workflow
    end
  end

  @doc false
  defmacro __before_compile__(_env) do
    quote do
      def __workflow__ do
        @workflow_stages |> Enum.reverse()
      end
    end
  end
end

defmodule CrucibleKitchen.Workflow.DSL do
  @moduledoc false

  @doc "Define a stage."
  defmacro stage(name, module, opts \\ []) do
    quote do
      @workflow_stages {:stage, unquote(name), unquote(module), unquote(opts)}
    end
  end

  @doc "Define a loop."
  defmacro loop(name, opts, do: block) do
    quote do
      @workflow_stages {:loop_start, unquote(name), unquote(opts)}
      unquote(block)
      @workflow_stages {:loop_end, unquote(name)}
    end
  end

  @doc "Define a conditional block."
  defmacro conditional(predicate, do: block) do
    quote do
      @workflow_stages {:conditional_start, unquote(predicate)}
      unquote(block)
      @workflow_stages {:conditional_end}
    end
  end

  @doc "Define a parallel block."
  defmacro parallel(opts \\ [], do: block) do
    quote do
      @workflow_stages {:parallel_start, unquote(opts)}
      unquote(block)
      @workflow_stages {:parallel_end}
    end
  end

  @doc "Workflow block (required wrapper)."
  defmacro workflow(do: block) do
    quote do
      unquote(block)
    end
  end
end
