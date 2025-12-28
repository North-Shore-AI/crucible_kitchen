# Crucible Kitchen

Industrial ML training orchestration - backend-agnostic workflow engine for supervised, reinforcement, and preference learning.

[![Hex Version](https://img.shields.io/hexpm/v/crucible_kitchen.svg)](https://hex.pm/packages/crucible_kitchen)
[![Docs](https://img.shields.io/badge/docs-hexpm-blue.svg)](https://hexdocs.pm/crucible_kitchen)

## Overview

Crucible Kitchen is the missing orchestration layer that makes ML cookbooks trivially thin. It provides:

- **Pre-built workflows** for supervised, RL, DPO, and distillation training
- **Declarative workflow DSL** for composing custom pipelines
- **Port/adapter pattern** for backend flexibility (Tinker, Fireworks, Modal, local Nx)
- **Comprehensive telemetry** for observability
- **First-class reproducibility** with deterministic seeding and artifact versioning

```
┌─────────────────────────────────────────────────────────────────┐
│                    COOKBOOK FRONTENDS                            │
│  tinkex_cookbook    fireworks_cookbook    modal_cookbook        │
│  (config + adapters only, <2K LOC each)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CRUCIBLE KITCHEN                             │
│  Recipes → Workflows → Stages → Ports                           │
│  (backend-agnostic orchestration)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND ADAPTERS                              │
│  Tinker, Fireworks, Modal, LocalNx, Noop (testing)             │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

Add `crucible_kitchen` to your dependencies:

```elixir
def deps do
  [
    {:crucible_kitchen, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
# 1. Define adapters for your backend
adapters = %{
  training_client: MyCookbook.Adapters.TrainingClient,
  dataset_store: MyCookbook.Adapters.DatasetStore,
  blob_store: CrucibleKitchen.Adapters.Noop.BlobStore
}

# 2. Configure your training run
config = %{
  model: "meta-llama/Llama-3.1-8B",
  epochs: 3,
  batch_size: 128,
  learning_rate: 2.0e-4
}

# 3. Run!
{:ok, result} = CrucibleKitchen.run(:supervised, config, adapters: adapters)
```

## Key Concepts

### Recipes

Configuration-driven training definitions. Recipes declare WHAT to train, not HOW.

```elixir
defmodule MyCookbook.Recipes.SlBasic do
  use CrucibleKitchen.Recipe

  def name, do: :sl_basic
  def description, do: "Basic supervised fine-tuning"
  def workflow, do: CrucibleKitchen.Workflows.Supervised

  def defaults do
    %{
      model: "meta-llama/Llama-3.1-8B",
      epochs: 1,
      batch_size: 128
    }
  end
end
```

### Workflows

Compositions of stages with control flow. Workflows define HOW training is orchestrated.

```elixir
defmodule MyWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    stage :load, LoadDatasetStage
    stage :init, InitSessionStage

    loop :epochs, over: fn ctx -> 0..(ctx.config.epochs - 1) end do
      loop :batches, over: fn ctx -> ctx.state.dataset end do
        stage :forward_backward, ForwardBackwardStage
        stage :optim_step, OptimStepStage
      end

      conditional fn ctx -> should_checkpoint?(ctx) end do
        stage :checkpoint, CheckpointStage
      end
    end

    stage :save, SaveWeightsStage
  end
end
```

### Stages

Atomic units of work with lifecycle hooks.

```elixir
defmodule ForwardBackwardStage do
  use CrucibleKitchen.Stage

  @impl true
  def name, do: :forward_backward

  @impl true
  def execute(context) do
    client = get_adapter(context, :training_client)
    batch = get_state(context, :current_batch)

    {:ok, future} = client.forward_backward(get_state(context, :session), batch)
    {:ok, put_state(context, :fb_future, future)}
  end
end
```

### Ports & Adapters

Behaviour contracts for external integrations. The kitchen defines ports; cookbooks provide adapters.

```elixir
# Port (in crucible_kitchen)
defmodule CrucibleKitchen.Ports.TrainingClient do
  @callback start_session(config :: map()) :: {:ok, session} | {:error, term()}
  @callback forward_backward(session, datums) :: {:ok, future} | {:error, term()}
  @callback optim_step(session, lr :: float()) :: {:ok, future} | {:error, term()}
  # ...
end

# Adapter (in your cookbook)
defmodule MyCookbook.Adapters.TrainingClient do
  @behaviour CrucibleKitchen.Ports.TrainingClient

  @impl true
  def start_session(config) do
    # Backend-specific implementation
  end
  # ...
end
```

## Built-in Workflows

| Workflow | Description |
|----------|-------------|
| `:supervised` | Standard supervised fine-tuning (SFT) |
| `:reinforcement` | RL with rollouts and PPO (GRPO, etc.) |
| `:preference` | Direct Preference Optimization (DPO) |
| `:distillation` | Knowledge distillation |

## Telemetry

All operations emit telemetry events for observability:

```elixir
# Attach console handler
CrucibleKitchen.attach_telemetry(:console)

# Or JSONL for file logging
CrucibleKitchen.attach_telemetry(:jsonl, path: "/var/log/training.jsonl")
```

Events:
- `[:crucible_kitchen, :workflow, :run, :start/:stop/:exception]`
- `[:crucible_kitchen, :stage, :run, :start/:stop/:exception]`
- `[:crucible_kitchen, :training, :step]`
- `[:crucible_kitchen, :training, :epoch]`
- `[:crucible_kitchen, :training, :checkpoint]`

## Testing

Use noop adapters for testing without a real backend:

```elixir
{:ok, result} = CrucibleKitchen.run(:supervised, config,
  adapters: CrucibleKitchen.Adapters.noop())
```

## Documentation

- [Getting Started](docs/guides/getting_started.md)
- [Custom Workflows](docs/guides/custom_workflows.md)
- [Writing Adapters](docs/guides/adapters.md)
- [Telemetry](docs/guides/telemetry.md)
- [API Reference](https://hexdocs.pm/crucible_kitchen)

## Design Documents

The architecture design is documented in:
- [Architecture Overview](docs/architecture/00_EXECUTIVE_SUMMARY.md)
- [Architecture Patterns](docs/architecture/01_ARCHITECTURE_PATTERNS.md)
- [Component Design](docs/architecture/02_COMPONENT_DESIGN.md)
- [Workflow Engine](docs/architecture/03_WORKFLOW_ENGINE.md)
- [API Surface](docs/architecture/04_API_SURFACE.md)

## Part of the Crucible Ecosystem

Crucible Kitchen integrates with:
- `crucible_train` - Types, renderers, training primitives
- `crucible_ir` - Experiment specifications
- `crucible_framework` - Pipeline orchestration
- `crucible_telemetry` - Research-grade instrumentation
- `crucible_harness` - Batch experiment management

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
