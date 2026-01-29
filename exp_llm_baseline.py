"""Legacy experiment entrypoint.

Old baseline (kept as reference). Prefer:
  python -m ssa.run_experiment --config configs/baseline.yaml
"""

import sys

from ssa.run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/baseline_old.yaml", *sys.argv[1:]]))
