import random

import numpy as np

from ssa.run_experiment import _set_seed


def test_set_seed_deterministic():
    _set_seed(123)
    a = (random.random(), float(np.random.rand()))

    _set_seed(123)
    b = (random.random(), float(np.random.rand()))

    assert a == b

