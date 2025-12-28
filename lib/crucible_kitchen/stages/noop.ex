defmodule CrucibleKitchen.Stages.Noop do
  @moduledoc """
  No-operation stage for testing and placeholders.
  """

  use CrucibleKitchen.Stage

  @impl true
  def name, do: :noop

  @impl true
  def execute(context), do: {:ok, context}
end
