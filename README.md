# SSA: Skill-Based Labour Market Simulation

This repository contains the SSA market simulator, agent implementations, and config-driven experiment runner.

## Setup (recommended)

Python 3.9+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quickstart (smoke test, no LLM calls)

Runs a tiny market simulation and writes logs under `logs/`:

```bash
python -m ssa.run_experiment --config configs/market_change.yaml --no-model --quiet --steps 2 --replicates 1
```

## Canonical entrypoint (experiments)

All experiments are run via the YAML runner:

```bash
python -m ssa.run_experiment --config configs/baseline.yaml
python -m ssa.run_experiment --config configs/ablation.yaml
python -m ssa.run_experiment --config configs/market_change.yaml
python -m ssa.run_experiment --config configs/market_recession.yaml
python -m ssa.run_experiment --config configs/moral.yaml
python -m ssa.run_experiment --config configs/price.yaml
```

Notes:
- The root-level `exp_*.py` files are legacy wrappers that call `ssa.run_experiment` with a config.
- Use `--steps` / `--replicates` to override config values for quick smoke tests.
- Outputs are written under `logs/<study>/<variant>/<run_name>_<replicate_id>.log`.
- Payments are performance-adjusted by default: `adjusted_reward = bid_price * performance` (toggle via `market.performance_pay` in config).
- Open bidding (optional): set `market.open_bidding: true` to reveal winning bid prices in the agent-visible market history.

## Demo notebook

The interactive entrypoint is `demo.ipynb`:

```bash
pip install -e .
jupyter lab demo.ipynb
```

## LLM / model configuration (optional)

Configs can specify a `model` block per agent (see `configs/*.yaml`). The runner supports a small set of model types
implemented in `ssa/utils.py` (e.g. `OpenAIClient`, `OpenRouterClient`).

The codebase currently defaults to reading credentials from `ssa/assets/secrets.yaml`. For local use, prefer creating an
untracked file (e.g. `ssa/assets/secrets.local.yaml`) and pointing configs at it via `secrets_path`, or passing `api_key`
directly in the config.

### Agent prompt variants

- `CoTAgent` and `ReActAgent` are two baseline prompt variants in this repo (not a full "ReAct tools" implementation).
- Both output the same structured `AgentActionResponse`, but differ in how much prior context is shown (e.g. CoT includes a
  “previous reasoning” snippet).
- Rewards are performance-adjusted by default (reward = performance_ratio * bid_price); the market logs show realized
  per-job rewards.

## Reproducibility / seeding

The config runner seeds randomness at the start of each replicate:
- Python `random` and NumPy `np.random` are seeded.
- If `torch` is available, it is also seeded.

Seed formula (see `ssa/run_experiment.py`):
- `suite.seed` comes from the YAML top-level `seed` (or defaults to 0).
- Each run gets a distinct base seed (`suite.seed + run_idx * 10_000`).
- Effective replicate seed is `(run.seed or base_seed) + replicate_id`.

Each exported log includes `suite_seed` and `effective_seed` under `config_extra`.

## Tests

```bash
pytest
```
