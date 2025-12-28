defmodule CrucibleKitchen.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/crucible_kitchen"

  def project do
    [
      app: :crucible_kitchen,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),

      # Hex
      description: description(),
      package: package(),

      # Docs
      name: "CrucibleKitchen",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {CrucibleKitchen.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Core dependencies
      {:crucible_train, "~> 0.2.0", optional: true},
      {:crucible_ir, "~> 0.2.0", optional: true},
      {:telemetry, "~> 1.2"},
      {:jason, "~> 1.4"},

      # Dev/test
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:mox, "~> 1.1", only: :test}
    ]
  end

  defp aliases do
    [
      quality: ["format --check-formatted", "credo --strict", "dialyzer"]
    ]
  end

  defp description do
    """
    Industrial ML training orchestration - backend-agnostic workflow engine
    for supervised, reinforcement, and preference learning.
    """
  end

  defp package do
    [
      name: "crucible_kitchen",
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md"
      },
      maintainers: ["North-Shore-AI"],
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE CHANGELOG.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "docs/guides/getting_started.md",
        "docs/guides/custom_workflows.md",
        "docs/guides/adapters.md"
      ],
      groups_for_modules: [
        Core: [
          CrucibleKitchen,
          CrucibleKitchen.Context,
          CrucibleKitchen.Recipe,
          CrucibleKitchen.Stage,
          CrucibleKitchen.Workflow
        ],
        Ports: [
          CrucibleKitchen.Ports.TrainingClient,
          CrucibleKitchen.Ports.DatasetStore,
          CrucibleKitchen.Ports.BlobStore,
          CrucibleKitchen.Ports.HubClient,
          CrucibleKitchen.Ports.MetricsStore,
          CrucibleKitchen.Ports.VectorStore
        ],
        "Built-in Workflows": [
          CrucibleKitchen.Workflows.Supervised,
          CrucibleKitchen.Workflows.Reinforcement,
          CrucibleKitchen.Workflows.Preference,
          CrucibleKitchen.Workflows.Distillation
        ],
        Telemetry: [
          CrucibleKitchen.Telemetry,
          CrucibleKitchen.Telemetry.Handlers.Console,
          CrucibleKitchen.Telemetry.Handlers.JSONL
        ]
      ]
    ]
  end
end
