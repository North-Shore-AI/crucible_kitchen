defmodule CrucibleKitchen.RL.MathEnvTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.RL.MathEnv

  defmodule VerifierStub do
    def verify(given, expected, _opts \\ []) do
      {:ok, String.trim(given) == String.trim(expected)}
    end
  end

  test "get_question appends the boxed suffix" do
    env = %MathEnv{problem: "2 + 2 = ?", answer: "4"}
    assert MathEnv.get_question(env) == "2 + 2 = ? Write your answer in \\boxed{} format."
  end

  test "check_format requires boxed answer" do
    env = %MathEnv{problem: "Q", answer: "A"}
    assert MathEnv.check_format(env, "Result is \\boxed{4}") == true
    assert MathEnv.check_format(env, "Result is 4") == false
  end

  test "check_answer uses verifier on boxed content" do
    env = %MathEnv{problem: "Q", answer: "4", verifier: VerifierStub}

    assert MathEnv.check_answer(env, "Answer: \\boxed{4}") == true
    assert MathEnv.check_answer(env, "Answer: \\boxed{5}") == false
  end
end
