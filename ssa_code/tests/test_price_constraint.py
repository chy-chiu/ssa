import pytest

from ssa.common import AgentActionResponse


def test_bid_price_must_be_positive():
    with pytest.raises(ValueError, match="Bid price must be > 0"):
        AgentActionResponse(reasoning="x", action="bid", targets=[("job_1", 0)])

    with pytest.raises(ValueError, match="Bid price must be > 0"):
        AgentActionResponse(reasoning="x", action="bid", targets=[("job_1", -1)])

