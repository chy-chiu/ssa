## Working Agreement (to avoid context rot)

- We do **one task at a time**.
- Exactly **one** checkbox item may be marked **(in progress)** at any time.
- When a task is finished, mark it `- [x]` and add a 1–3 line “Result / How to verify” note under **Done Log**.

## Context notes (from user)
- `paper/` is reference-only for the agent; not intended for submission from this repo.
- Any `vivabench` references are copy/paste artifacts; safe to remove.

## Priorities
- P0 = required for “submission-ready” (reproducible + claims match code).
- P1 = important for reviewer confidence / polish.
- P2 = nice-to-have cleanup.

## Current Task (do this now)

- [x] **[P1] Rename baseline agents for clarity**: rename `ssa/agents/llm.py`→`ssa/agents/cot_agent.py` and `ssa/agents/llm2.py`→`ssa/agents/react_agent.py`; rename SSA agent modules similarly (`_ssa*.py`→`ssa_agent*.py`); update class names (`LLMAgent`/`LLM2Agent`/`LLMSSA`) to match; keep backwards-compatible aliases if easy.

## Backlog

### P0 (submission-blocking)
- [x] [P0] Write documentation including entry point / demo (single canonical “how to run”).
- [x] [P0] Include seeding: pin RNG seeds and document seeding protocol (what is seeded + where).
- [x] [P0] Ensure performance-based pay is enabled in code and documented (paper references this variant).
- [x] [P0] Implement open-bid variant as a toggle-able feature flag when initializing market (paper references this variant).
- [x] [P0] Open-bid variant described but not cleanly implemented; re-implement by modifying / injecting it in the market history.
- [ ] [P0] Fix demand-shift experiment script bugs (index range mismatch; undefined export variable) and confirm corrected version generated the figure.
- [ ] [P0] Remove any identifying information (authors/emails/paths/model endpoints/etc.).

### P1 (core completeness / correctness)
- [ ] [P1] Using `CipherTask` as reference, finish the `OrderTask` and `Diagnosis` tasks.
- [ ] [P1] Enforce / document constraint $p_{i,J,t}>0$ to avoid degenerate utilities at $p=0$ (throw error if violated).
- [ ] [P1] Evidence accumulators $(r_{i,k,t}, s_{i,k,t})$ vs recompute from logs.
  - Decision: recompute from logs (ignore as a separate task unless a reviewer-visible doc/code change is needed).
  - [ ] [P1] (Optional) Align baseline prompts with current payment mechanics and document what “CoT” vs “ReAct” means here.

### P2 (maintenance)
- [ ] [P2] Clean up any unused / exploratory code (after P0/P1 so we don’t delete needed parts), tidy up other messy stuff under `./scripts/`.

## Notes for agent
- Goal: make repo runnable end-to-end from a clean environment with 1–2 commands; “paper claims” should be traceable to a script/config in-repo.
- When in doubt: prefer a minimal, deterministic “baseline” run that produces a small artifact in `logs/` and can be cited in docs.

## Notes / answers from user
- Which single command should be the “golden path” for reviewers? Example: `python -m ssa.run_experiment --config configs/baseline.yaml`
- ANSWER: That path looks fine and good
- Do you want the repo to be installable via `pip install -e .` (preferred) or runnable without install?
- ANSWER: Yes pip install -e .
- Which experiments/figures are must-reproduce for submission readiness (top 2–3)?
- ANSWER: All the experiments included in exp_... .py (except for baseline / ablation - we use baseline2, ablation2 instead. however keep baseline / ablation with suffix "_old" as configs)
- Should we remove `paper/` from git entirely, or keep it but clearly exclude it from packaging/docs?
- ANSWER: just exclude is fine, in case you need to reference it again

## Done Log

- **[2026-01-29] Config-driven runner**
  - Result: Added `ssa/run_experiment.py` + YAML configs in `configs/` for all `exp_*.py` experiments (canonical `baseline.yaml`/`ablation.yaml`, plus `_old` variants); updated each `exp_*.py` to be a thin wrapper around the config runner.
  - How to verify (smoke, no LLM calls): `.venv/bin/python -m ssa.run_experiment --config configs/market_change.yaml --no-model --quiet --steps 2 --replicates 1` (should write `logs/market_change/market_change.log`).
- **[2026-01-29] Rename baseline agent modules/classes**
  - Result: Introduced canonical `CoTAgent` (`ssa/agents/cot_agent.py`), `ReActAgent` (`ssa/agents/react_agent.py`), and `SSAAgent` (`ssa/agents/ssa_agent.py`) / `SSAAgentAblation` (`ssa/agents/ssa_agent_ablation.py`); updated runner registry and `configs/*.yaml` to use canonical names; kept backwards-compatible aliases (`LLMAgent`, `LLM2Agent`, `LLMSSA`).
  - How to verify: `.venv/bin/python -c "import ssa.agents as a; print(a.CoTAgent, a.ReActAgent, a.SSAAgent); print(a.LLMAgent, a.LLM2Agent, a.LLMSSA)"` and `.venv/bin/python -m ssa.run_experiment --config configs/market_change.yaml --no-model --quiet --steps 2 --replicates 1`.
- **[2026-01-29] Canonical README / how-to-run**
  - Result: Expanded `README.md` with a single canonical entrypoint (`python -m ssa.run_experiment --config ...`), a no-model smoke test, and a pointer to `demo.ipynb`; fixed an invalid local path in `requirements.txt` and added a minimal `setup.py` to support `pip install -e .` in older tooling.
  - How to verify: create a fresh venv, run `pip install -r requirements.txt`, then run the “Quickstart” command in `README.md` (should write an output under `logs/`).
- **[2026-01-29] Seeding protocol**
  - Result: Documented seeding protocol in `README.md` and ensured `ssa/run_experiment.py` seeds Python `random`, NumPy, and (if available) `torch` per replicate.
  - How to verify: `pip install -r requirements.txt && pytest -q` (includes `tests/test_seeding.py`), and confirm exported logs include `config_extra.effective_seed`.
- **[2026-01-29] Performance-based pay**
  - Result: Added `market.performance_pay` (default on) and compute `adjusted_reward = bid_price * performance` in `ssa/market.py`; documented the default in `README.md`.
  - How to verify: `pip install -r requirements.txt && pytest -q` (includes `tests/test_payments.py`), and run a short `--no-model` experiment to observe `agent_round_rewards` drop below raw bid sums.
- **[2026-01-29] Open bidding toggle**
  - Result: Added `market.open_bidding` (default off) to optionally reveal winning bid prices in the agent-visible market history string; added example config `configs/market_change_open_bid.yaml`.
  - How to verify: `.venv/bin/python -m ssa.run_experiment --config configs/market_change_open_bid.yaml --no-model --quiet --steps 2 --replicates 1` and confirm the prompt trace includes `@$<winning_price>`.
- **[2026-01-29] Open bidding bid-distribution disclosure**
  - Result: When `market.open_bidding: true`, agent-facing market history now also includes a compact per-job bid summary (`BIDS: ...`) for the shown rounds.
  - How to verify: run the same open-bid smoke command above and confirm the prompt trace includes a `BIDS:` line.
