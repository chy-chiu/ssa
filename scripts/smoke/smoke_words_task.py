#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssa.agents.policy import PolicyAgent
from ssa.common import Job
from ssa.market import LabourMarket
from ssa.tasks.cipher import CipherAgent, CipherTask
from ssa.tasks.task import ProxyTask
from ssa.utils import OpenAIClient


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {
            "token_usage": {"total_tokens": 0, "completion_tokens": 0, "prompt_tokens": 0},
            "llm_reasoning": "",
        }


class _DummyModel:
    def invoke(self, messages):
        prompt = ""
        if messages:
            prompt = getattr(messages[-1], "content", "") or ""
        encrypted = re.findall(r"\b[A-Z]{3,}\b", prompt)
        # Keep only the final batch from the decryption question.
        answer = encrypted[-3:] if encrypted else ["AAAAA", "BBBBB", "CCCCC"]
        return _DummyResponse(json.dumps({"reasoning": "offline smoke", "answer": answer}))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip() or None
    except Exception:
        return None


def run_smoke(
    *,
    output_path: str,
    secrets_path: str,
    model_name: str,
    rounds: int = 5,
    seed: int = 7,
    no_model: bool = False,
) -> str:
    random.seed(seed)
    np.random.seed(seed)

    tasks = [
        CipherTask(task_id="CIPHER"),
        ProxyTask(task_id="PROXY", noise=0.0),
    ]
    for idx, task in enumerate(tasks):
        try:
            task.generate_ground_truth(seed=seed + idx)
        except TypeError:
            task.generate_ground_truth()
        except Exception:
            pass
    jobs = [
        Job(id="JB-C0", task_id="CIPHER", base_reward=10.0, job_p=1.0, noise=0.0, w_q=0.6),
        Job(id="JB-P0", task_id="PROXY", base_reward=8.0, job_p=1.0, noise=0.0, w_q=0.6),
    ]

    if no_model:
        task_model = _DummyModel()
    else:
        task_model = OpenAIClient(
            model_name=model_name,
            secrets_path=secrets_path,
            effort="none",
            temperature=0.2,
        )

    agents = []
    for i in range(2):
        agent = PolicyAgent(agent_id=f"AG-{i}", jobs=jobs, model=None, verbose=False)
        agent.set_policy(
            job_preferences=["JB-C0", "JB-P0"],
            train_p=0.0,
            underbid_factor=0.95,
            greedy=False,
        )
        agent.subagents["CIPHER"] = CipherAgent(model=task_model, task_id="CIPHER")
        agent.skill_history = [agent.skill_level_by_task]
        agents.append(agent)

    market = LabourMarket(
        jobs=jobs,
        tasks=tasks,
        agents=agents,
        market_limit=1,
        market_pref_limit=2,
        agent_pref_limit=2,
        history_limit=5,
        skill_phi=0.1,
        rep_initial=0.5,
        rep_window=3,
        rep_sensitivity=1.0,
        rep_lambda=0.5,
        gumbel_t=0.01,
        performance_pay=True,
        open_bidding=False,
    )

    for _ in range(rounds):
        market.simulate_timestep()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    market.export(
        str(out_path),
        config_extra={
            "study": "smoke_words",
            "variant": "default",
            "reviewer_target": "smoke",
            "hypothesis_id": "smoke_words_task",
            "git_commit": _git_commit(),
            "suite_seed": seed,
            "effective_seed": seed,
            "replicate_id": 0,
            "replicate_idx": 0,
            "run_name": "smoke_words",
            "scoring_mode": "cobb_douglas",
            "rep_update_mode": "full_benchmark",
            "agent_mix": {"PolicyAgent": 2},
        },
    )
    # Keep legacy smoke path requested in TODO and also mirror to strict layout.
    strict_path = Path("logs") / "smoke_words" / "default" / out_path.name
    if strict_path.as_posix() != out_path.as_posix():
        strict_path.parent.mkdir(parents=True, exist_ok=True)
        strict_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal smoke market with CipherTask + ProxyTask.")
    parser.add_argument("--output", default="logs/smoke_words/smoke_words_0.log")
    parser.add_argument("--secrets-path", default="assets/secrets.yaml")
    parser.add_argument("--model-name", default="gpt-5.4-cc")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    out = run_smoke(
        output_path=args.output,
        secrets_path=args.secrets_path,
        model_name=args.model_name,
        rounds=args.rounds,
        seed=args.seed,
        no_model=bool(args.no_model),
    )
    print(f"Wrote smoke log: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
