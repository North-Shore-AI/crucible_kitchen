defmodule CrucibleKitchen.Stages.DescribeSchemaTest do
  use ExUnit.Case, async: true

  alias Crucible.Stage.Schema

  test "all kitchen stages expose a valid describe/1 schema" do
    CrucibleKitchen.Stages.all()
    |> Enum.each(fn stage ->
      assert Code.ensure_loaded?(stage),
             "expected #{inspect(stage)} to be loadable"

      assert function_exported?(stage, :describe, 1),
             "expected #{inspect(stage)} to implement describe/1"

      schema = stage.describe(%{})

      assert :ok == Schema.validate(schema),
             "invalid schema for #{inspect(stage)}: #{inspect(schema)}"
    end)
  end
end
