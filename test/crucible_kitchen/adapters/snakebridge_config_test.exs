defmodule CrucibleKitchen.Adapters.SnakebridgeConfigTest do
  use ExUnit.Case, async: true

  test "snakebridge libraries include math_verify, sympy, pylatexenc" do
    libraries =
      SnakeBridge.Config.load().libraries
      |> Enum.map(& &1.name)

    assert :math_verify in libraries
    assert :sympy in libraries
    assert :pylatexenc in libraries
  end
end
