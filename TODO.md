# ICML 2026 Rebuttal Engineering TODO (Code + Experiments)

We are not cooked; tight engineering, clean analysis, and disciplined evidence can still move this.

Scope: this TODO is execution-focused (code + runs + analysis artifacts). Paper-writing text lives elsewhere.

## Working Agreement

- We do one execution task at a time.
- Exactly one checkbox can be marked `(in progress)`.
- Experiments must write under `logs/` (no ad-hoc output dirs for final artifacts).
- Every completed task gets a 1-3 line entry in **Done Log** with exact verify command.

## Current Task

- [ ] **[P0] Stand up rebuttal-grade experiment logging + analysis pipeline** *(in progress)*.

## Archive Reuse Inventory (already reviewed)

Primary reusable sources:
- `.archive/scripts/analysis.py`
- `.archive/scripts/analysis2.py`
- `.archive/scripts/trace_analysis.py`
- `.archive/scripts/action_analysis.py`
- `.archive/notebooks/analysis.ipynb`
- Current `ssa/analysis.py`

Reusable functions/ideas to adopt (and clean up):
- `get_summary_df` (agent-level summary table)
- `compute_trace_means`, `interp_trace` (trace normalization)
- `recovery_score` (rank mobility)
- `gini` (inequality / concentration)
- trace utilities: `format_trace`, `format_trace_history`
- trace scoring scaffold + batch async pattern from `trace_analysis.py`
- ablation main/interaction effect analysis pattern from `analysis2.py`

## P0 — Logging + Analysis First (blocker for all rebuttal experiments)

### P0.1 Logging contract and run layout

- [ ] **[P0][Code] Define a strict run layout under `logs/`**:
  - `logs/<study>/<variant>/<run_name>_<replicate_id>.log`
  - `logs/<study>/<variant>/analysis/*.csv|*.md|*.png`
- [ ] **[P0][Code] Add run metadata fields** into exported config/log:
  - `study`, `variant`, `reviewer_target`, `hypothesis_id`, `git_commit` (if available), `effective_seed`, `scoring_mode`, `rep_update_mode`, `agent_mix`.
- [ ] **[P0][Code] Add one `run_index.csv` builder** that scans `logs/**` and registers all runs + metadata.
- [ ] **[P0][Test] Add a logging schema smoke test** validating required metadata keys exist in exported logs.

### P0.2 Analysis module split (adopt + clean existing functions)

- [ ] **[P0][Code] Create `ssa/analysis/io.py`**
  - `discover_logs()`, `load_experiment_logs()`, `build_run_index()`.
- [ ] **[P0][Code] Create `ssa/analysis/metrics.py`**
  - Move/refactor: `get_summary_df`, `recovery_score`, `compute_trace_means`, `interp_trace`, `gini`.
  - Add robust handling for missing/partial traces and zero-denominator cases.
- [ ] **[P0][Code] Create `ssa/analysis/traces.py`**
  - Move/refactor: `format_trace`, `format_trace_history`, batch trace extraction.
  - Add trace-to-dataframe converter: one row per `(run, agent, round)`.
- [ ] **[P0][Code] Create `ssa/analysis/rebuttal.py`**
  - Directional checks for rebuttal claims (pass/fail + effect size):
    - open bidding lowers prices,
    - performance pay increases training,
    - SSA > controls,
    - trends persist under scale/robustness variants.
- [ ] **[P0][Test] Add unit tests for `metrics.py` and `io.py`** with tiny synthetic logs.

### P0.3 CLI scripts (so analysis is one-command reproducible)

- [ ] **[P0][Code] Add `scripts/analysis/build_run_index.py`**
  - Input: `--root logs`
  - Output: `logs/run_index.csv`
- [ ] **[P0][Code] Add `scripts/analysis/summarize_runs.py`**
  - Input: run selector (glob/tag/study)
  - Output: agent-level + variant-level summary CSV/MD tables.
- [ ] **[P0][Code] Add `scripts/analysis/plot_traces.py`**
  - Output standardized trace plots for bids, training rate, reward trajectories.
- [ ] **[P0][Code] Add `scripts/analysis/rebuttal_bundle.py`**
  - Produces reviewer-ready compact tables in `logs/rebuttal_bundle/`.
- [ ] **[P0][Code] Add `scripts/analysis/trace_score.py`**
  - Wrapper around trace scoring prompt scaffold; writes per-trace scores to CSV.

### P0.4 Small integration smoke script (requested)

- [ ] **[P0][Code] Add `scripts/smoke/smoke_words_task.py`** with:
  - 2 agents,
  - 5 rounds,
  - 2 tasks total,
  - includes a real words-based task (`CipherTask`) plus one lightweight second task,
  - writes to `logs/smoke_words/smoke_words_0.log`.
- [ ] **[P0][Code] Add `scripts/smoke/run_smoke.sh`** for one-command local sanity run.
- [ ] **[P0][Test] Add `tests/test_smoke_words_task.py`** to verify log file shape/keys (short mode, no long run).

## P1 — Rebuttal Experiment Surface (after P0 is stable)

- [ ] **[P1][Code] Configurable client scoring mode** (`cobb_douglas`, `linear`, optional CES rho).
- [ ] **[P1][Code] Reputation update mode for unmatched agents** (`full_benchmark`, `reduced_benchmark`, `none`).
- [ ] **[P1][Code] SSA length/info-matched control agent prompt** (no explicit M/C/P scaffold).
- [ ] **[P1][Code] Scale-ready config generation** for N sweeps (N=32, N=64).
- [ ] **[P1][Code] Stochastic upskilling option** (probabilistic training success).

## P1 — Tier-A Runs (highest ROI for score movement)

- [ ] **[P1][Run] Alternative scoring robustness** (`cobb_douglas` vs `linear`, CES if cheap).
- [ ] **[P1][Run] Reduced/no benchmark recalibration robustness**.
- [ ] **[P1][Run] SSA prompt control run** (same backbone).
- [ ] **[P1][Run] Scale N=32**.
- [ ] **[P1][Artifact] One compact rebuttal table** with effect sizes + directional pass/fail.

## P2 — If compute/time allows

- [ ] **[P2][Run] Scale N=64**.
- [ ] **[P2][Run] Reputation forgetting/window sweep**.
- [ ] **[P2][Run] Client-side sensitivity sweep (`w_q`/scoring params)**.
- [ ] **[P2][Run] LLM-as-judge bias stress check** (order/anonymization).

## Reviewer Mapping (evidence bundles)

- [ ] **GrSA**: scoring robustness + rep-update robustness + SSA control + N=32.
- [ ] **5M91**: at least one robustness result + stochastic upskilling if done.
- [ ] **TW1h**: stochastic upskilling + trace-scoring limitation honesty (+ bias check if done).
- [ ] **t6s6**: concise shared robustness summary.

## Run Order (strict)

- [ ] 1. Logging/analysis infra and smoke scripts.
- [ ] 2. Smoke all new modes/scripts.
- [ ] 3. Tier-A full runs.
- [ ] 4. Rebuttal bundle generation.
- [ ] 5. Tier-B/Tier-C only if Tier-A complete.

## Done Log

- [ ] Add dated entries here as tasks complete, with result + exact verify command.

