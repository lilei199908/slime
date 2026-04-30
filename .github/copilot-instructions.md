# Copilot Instructions for slime

## What is slime

slime is an LLM post-training framework for RL scaling. It connects **Megatron** (training) with **SGLang** (rollout/inference) via a **data buffer**, orchestrated by **Ray**. It supports GRPO, PPO, SFT, and on-policy distillation across models like Qwen3, DeepSeek V3, GLM, and Llama 3.

## Build & Install

```bash
pip install -e . --no-deps          # editable install (no-deps because heavy deps like megatron/sglang are managed separately)
pip install -r requirements.txt     # install Python dependencies
```

## Lint & Format

Pre-commit handles all formatting (black, isort, ruff, autoflake):

```bash
pre-commit run --all-files --show-diff-on-failure --color=always
```

Toolchain: **black** (line-length 119), **isort** (black profile), **ruff** (E/F/B/UP rules, line-length 320), **autoflake** (remove unused imports).

## Tests

Tests are GPU-dependent end-to-end tests that require multi-GPU hardware, model weights, and datasets. They are not standard unit tests.

```bash
# Run a single test (requires GPUs; uses gpu_lock_exec to claim N GPUs)
python tests/ci/gpu_lock_exec.py --count 4 -- python tests/test_qwen2.5_0.5B_gsm8k_short.py

# Run pytest (for any pytest-compatible tests)
pytest tests/ -m "unit"           # run only unit-marked tests
pytest tests/test_chunked_gae.py  # run a specific test file
```

CI is triggered on PRs via GitHub labels (`run-ci-short`, `run-ci-fsdp`, `run-ci-long`). PR test workflows are auto-generated from `pr-test.yml.j2` via `generate_github_workflows.py`.

## Architecture

### Three-module design

1. **Training** (`slime/backends/`): Megatron-based (`megatron_utils/`) or FSDP-based (`fsdp_utils/`) training backends. Handles forward/backward, loss computation, checkpointing, and weight updates.
2. **Rollout** (`slime/rollout/`): SGLang-based inference engine that generates responses, computes rewards, and applies filters. The `sglang_rollout.py` is the main rollout driver.
3. **Data Buffer** (`slime/rollout/data_source.py`): Manages prompt datasets and generated samples flowing between rollout and training.

### Orchestration layer

- **Ray** (`slime/ray/`): Manages GPU placement groups, actor groups, and the `RolloutManager`. `placement_group.py` allocates GPUs; `train_actor.py` wraps training workers.
- **Router** (`slime/router/`): A FastAPI-based request router (`SlimeRouter`) that load-balances across SGLang engine instances with health checking and failure quarantine.

### Plugin system (`slime_plugins/`)

- `models/`: Model-specific adapters for SGLang (e.g., `glm4.py`, `qwen3_next.py`, `deepseek_v32.py`).
- `mbridge/`: Model-specific adapters for the Megatron training backend.
- `rollout_buffer/`: Standalone data generation component that can be used independently from training.

### Entry points

- `train.py`: Standard synchronous RL training loop.
- `train_async.py`: Asynchronous training variant.
- `scripts/`: Ready-to-run shell scripts for various model configurations.

## Key Conventions

### Arguments are in three categories
1. **Megatron args**: Standard Megatron flags (e.g., `--tensor-model-parallel-size 2`).
2. **SGLang args**: Must be prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static`).
3. **slime args**: Defined in `slime/utils/arguments.py`.

### Dynamic function loading
Custom generate functions, reward models, and filters are specified as dotted Python paths (e.g., `my_module.my_reward_fn`) and loaded at runtime via `slime.utils.misc.load_function`. This is the primary extension mechanism.

### Reward models (`slime/rollout/rm_hub/`)
Built-in reward types: `math`, `dapo`, `deepscaler`, `f1`, `gpqa`, `ifbench`, `remote_rm`, `random`. Custom RMs are loaded via `--custom-rm-path`. All RM functions are async.

### The `Sample` dataclass (`slime/utils/types.py`)
Central data type flowing through the entire pipeline. Contains prompt, response, reward, loss_mask, tokens, metadata, and status tracking. Samples are grouped as `list[list[Sample]]` (groups of samples per prompt).

### Training backends are swappable
Megatron (`slime/backends/megatron_utils/`) and FSDP (`slime/backends/fsdp_utils/`) are interchangeable backends. Both implement actor, checkpoint, and weight update interfaces.

### Distributed debugging
When debugging multi-GPU/multi-node issues, see `.claude/skills/SKILL.md` for patterns around asymmetric keys, startup race conditions, and the modify-sync-restart-verify workflow.

## Contribution Scope

Per CONTRIBUTING.md, the project accepts **bug fixes** and **general-purpose large-scale RL optimizations** with clear benchmarks. Large refactors, abstraction proposals, and features that can't be verified through CI are out of scope.
