from __future__ import annotations

import numpy as np

from ssa.analysis.metrics import compute_trace_means, get_summary_df, gini, interp_trace, recovery_score
from ssa.common import ExperimentLog
from ssa.experiment_config import AgentSpec, JobSpec, MarketSpec, RunSpec, TaskSpec
from ssa.run_experiment import _run_one


def _make_metrics_run() -> RunSpec:
    return RunSpec(
        name="metrics_run",
        study="metrics",
        variant="v1",
        reviewer_target="metrics",
        hypothesis_id="metrics",
        steps=3,
        tasks=[TaskSpec(id="SK-A"), TaskSpec(id="SK-B")],
        jobs=[
            JobSpec(id="JB-A0", task_id="SK-A", base_reward=10.0, job_p=1.0, noise=0.0),
            JobSpec(id="JB-B0", task_id="SK-B", base_reward=8.0, job_p=1.0, noise=0.0),
        ],
        agents=[
            AgentSpec(type="PolicyAgent", ids=["PL-0"], policy={"greedy": True, "underbid_factor": 0.9, "train_p": 0.0}),
            AgentSpec(type="PolicyAgent", ids=["PL-1"], policy={"greedy": True, "underbid_factor": 0.9, "train_p": 0.0}),
        ],
        market=MarketSpec(),
    )


def test_trace_metric_primitives():
    traces = [[(0, 0.5), (1, 0.7)], [(0, 0.4), (1, 0.9)]]
    means = compute_trace_means(traces)
    assert means.shape == (2, 2)
    assert np.allclose(means[:, 0], [0.0, 1.0])
    interp = interp_trace(means, n_steps=5)
    assert interp.shape == (5,)

    rewards = np.array([[0.1, 0.2, 0.5], [0.3, 0.25, 0.2]])
    recovery, jump = recovery_score(rewards)
    assert recovery.shape == (2,)
    assert jump.shape == (2,)
    assert 0 <= gini([1, 2, 3, 4]) <= 1


def test_get_summary_df_smoke(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = _run_one(
        run=_make_metrics_run(),
        suite_seed=333,
        replicate_id=0,
        replicate_idx=0,
        force_no_model=True,
        quiet=True,
    )
    exp_log = ExperimentLog.load(out)
    summary = get_summary_df(exp_log, fp=out)
    assert len(summary) == 2
    required = {"agent_id", "reward", "train_p", "winrate", "recovery", "total_tokens"}
    assert required.issubset(set(summary.columns))
