from ssa.market import _adjusted_reward


def test_adjusted_reward_with_performance_pay():
    assert _adjusted_reward(bid_price=10.0, performance=0.25, performance_pay=True) == 2.5


def test_adjusted_reward_without_performance_pay():
    assert _adjusted_reward(bid_price=10.0, performance=0.25, performance_pay=False) == 10.0

