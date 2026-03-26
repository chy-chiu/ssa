# Repository Guidelines

## Project Structure & Module Organization

- `ssa/`: primary Python package.
  - `ssa/market.py`: core market simulation logic.
  - `ssa/agents/`: agent implementations (LLM, oracle, etc.).
  - `ssa/tasks/`: task definitions used by the market/agents.
  - `ssa/utils.py`, `ssa/common.py`: shared utilities and data models.
- `tests/`: lightweight, pytest-style tests (`test_*.py`).
- `assets/`: small data/config files (treat anything named “secrets” as sensitive).
- `logs/`: experiment outputs and run artifacts.
- `ssa_code`: this is the submission version of the code to ensure anonymity etc. IGNORE THIS FOLDER
- Root notebooks/scripts: `demo.ipynb`, `analysis*.ipynb`, `exp_*.py`, `explore*.py` (mostly research/experiments).

## Build, Test, and Development Commands

Common local setup (Python 3.9+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

- Run tests: `pytest` (install `pytest` if your environment doesn’t include it).
- Work in notebooks: `jupyter lab` (see `demo.ipynb` for an entrypoint).

## Coding Style & Naming Conventions

- Formatting: Black with line length `120` (see `pyproject.toml`).
- Indentation: 4 spaces; prefer explicit types for public functions and Pydantic models.
- Naming: `snake_case` for files/functions/vars; `PascalCase` for classes; constants in `UPPER_SNAKE_CASE`.

## Testing Guidelines

- Prefer small unit tests for deterministic logic in `ssa/` (market mechanics, normalization, scoring).
- Conventions: files `tests/test_*.py`, test functions `test_*`, keep tests independent and fast.

## Commit & Pull Request Guidelines

- Commit history is simple and descriptive (e.g., “updated market mechanisms”, “added experiments, tests”).
- Suggested subject format: `area: short change` (e.g., `market: tweak matching rule`, `agents: improve prompt`).
- PRs: include a brief problem/solution summary, how to reproduce/validate (commands or notebook steps), and link related issues/logs (e.g., files under `logs/`).

## Security & Configuration Tips

- Do not commit real API keys or tokens. Prefer environment variables or local-only config files.
- `assets/secrets.yaml` exists in this repo; treat it as a template/sample and keep real credentials out of it.
- For local secrets, prefer a separate untracked file (e.g., `assets/secrets.local.yaml`) or a `.env`.
