"""Legacy experiment entrypoint.

This file is kept for backwards compatibility; the canonical entrypoint is:
  python -m ssa.run_experiment --config configs/baseline.yaml
"""

import sys

from ssa.run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/baseline.yaml", *sys.argv[1:]]))
