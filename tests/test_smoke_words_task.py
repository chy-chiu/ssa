from __future__ import annotations

from ssa.analysis.io import REQUIRED_RUN_METADATA
from ssa.common import ExperimentLog
from scripts.smoke.smoke_words_task import run_smoke


def test_smoke_words_task_log_shape_and_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out_path = run_smoke(
        output_path="logs/smoke_words/smoke_words_0.log",
        secrets_path="assets/secrets.yaml",
        model_name="gpt-5.4-cc",
        rounds=2,
        seed=11,
        no_model=True,
    )

    exp_log = ExperimentLog.load(out_path)
    assert len(exp_log.agent_ids) == 2
    assert len(exp_log.history) == 2
    assert set(exp_log.task_ids) == {"CIPHER", "PROXY"}
    for key in REQUIRED_RUN_METADATA:
        assert key in exp_log.config
