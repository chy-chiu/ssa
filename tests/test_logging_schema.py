from __future__ import annotations

from ssa.analysis.io import REQUIRED_RUN_METADATA
from ssa.common import ExperimentLog
from ssa.experiment_config import AgentSpec, JobSpec, MarketSpec, RunSpec, TaskSpec
from ssa.run_experiment import _run_one


def _make_run_spec() -> RunSpec:
    return RunSpec(
        name="logging_schema",
        study="test_study",
        variant="test_variant",
        reviewer_target="smoke",
        hypothesis_id="p0_logging",
        steps=1,
        tasks=[TaskSpec(id="SK-A")],
        jobs=[JobSpec(id="JB-A0", task_id="SK-A", base_reward=10.0, job_p=1.0, noise=0.0)],
        agents=[
            AgentSpec(
                type="PolicyAgent",
                ids=["PL-0"],
                policy={"greedy": True, "underbid_factor": 0.9, "train_p": 0.0},
            )
        ],
        market=MarketSpec(),
    )


def test_logging_schema_required_metadata(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run = _make_run_spec()
    out_path = _run_one(
        run=run,
        suite_seed=123,
        replicate_id=0,
        replicate_idx=0,
        force_no_model=True,
        quiet=True,
    )

    exp_log = ExperimentLog.load(out_path)
    for key in REQUIRED_RUN_METADATA:
        assert key in exp_log.config
    assert exp_log.config["study"] == "test_study"
    assert exp_log.config["variant"] == "test_variant"
    assert exp_log.config["effective_seed"] == 123
