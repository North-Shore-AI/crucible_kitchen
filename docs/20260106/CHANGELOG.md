# Changelog (2026-01-06)

- Added SnakeBridge adapter wrapper and math_verify helper with deterministic tests.
- Configured SnakeBridge libraries (sympy, pylatexenc, math_verify) and compiler; skip generation in tests.
- Migrated LLM/embedding/vector noops and SDK adapters into crucible_kitchen with contract tests.
- Aligned TrainingClient adapters with loss_fn_config/forward_backward_custom and added noop adapter tests.
- Added SamplingClient adapters (noop/Tinkex) plus contract and noop tests.
- Refactored RL stages to use CrucibleTrain rollouts/advantages and added NoopEnv fallback.
- Refactored DPO stages to use SamplingClient reference logprobs and forward_backward_custom DPO loss.
- Added DPO stage test adapters and updated preference workflow await to store DPO metrics.
- Implemented distillation stages (teacher init/inference, distill datums, forward/backward, metrics, cleanup) with tests.
- Added MathEnv + MathEnvBuilder for GSM8K-style RL with math_verify integration.
- Extended BuildEnvGroup to support context-aware builder functions and fixed validation.
- Expanded distillation dataset prompt extraction to handle nested input.problem/question fields.
- Added prod config stub to avoid missing config import during dependency compile.
