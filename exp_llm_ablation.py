"""Legacy experiment entrypoint.

Old ablation (kept as reference). Prefer:
  python -m ssa.run_experiment --config configs/ablation.yaml
"""

import sys

from ssa.run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/ablation_old.yaml", *sys.argv[1:]]))
