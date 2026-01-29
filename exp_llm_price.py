"""Legacy experiment entrypoint.

Canonical entrypoint:
  python -m ssa.run_experiment --config configs/price.yaml
"""

import sys

from ssa.run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/price.yaml", *sys.argv[1:]]))
