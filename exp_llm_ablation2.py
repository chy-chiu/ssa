"""Legacy experiment entrypoint.

Canonical entrypoint:
  python -m ssa.run_experiment --config configs/ablation.yaml
"""

import sys

from ssa.run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/ablation.yaml", *sys.argv[1:]]))
