from __future__ import annotations

from pathlib import Path

from ssa.analysis.io import REQUIRED_RUN_METADATA, build_run_index, discover_logs, load_experiment_logs
from ssa.experiment_config import AgentSpec, JobSpec, MarketSpec, RunSpec, TaskSpec
from ssa.run_experiment import _run_one


def _make_run(name: str, study: str, variant: str, open_bidding: bool) -> RunSpec:
    return RunSpec(
        name=name,
        study=study,
        variant=variant,
        reviewer_target="io_test",
        hypothesis_id=name,
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
        market=MarketSpec(open_bidding=open_bidding),
    )


def test_io_discovery_load_and_index(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _run_one(
        run=_make_run("run_a", "study_a", "v1", open_bidding=False),
        suite_seed=101,
        replicate_id=0,
        replicate_idx=0,
        force_no_model=True,
        quiet=True,
    )
    _run_one(
        run=_make_run("run_b", "study_a", "v2", open_bidding=True),
        suite_seed=202,
        replicate_id=0,
        replicate_idx=0,
        force_no_model=True,
        quiet=True,
    )

    paths_v1 = discover_logs("logs", study="study_a", variant="v1")
    assert len(paths_v1) == 1
    assert "run_a_0.log" in paths_v1[0].name

    logs = load_experiment_logs(paths_v1)
    assert len(logs) == 1
    assert logs[0].config["study"] == "study_a"

    out_csv = Path("logs/run_index.csv")
    df = build_run_index(root="logs", output_path=out_csv)
    assert len(df) == 2
    assert out_csv.exists()
    for key in REQUIRED_RUN_METADATA:
        assert key in df.columns
    assert set(df["variant"].tolist()) == {"v1", "v2"}
