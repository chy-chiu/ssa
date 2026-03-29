#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssa.analysis.io import discover_logs
from ssa.analysis.traces import batch_extract_traces
from ssa.common import ExperimentLog
from ssa.utils import OpenAIClient

TRACE_SCORING_PROMPT = """Score one reasoning trace on three dimensions from 0 to 6:
- metacognition
- competitive_awareness
- strategic_planning
Return strict JSON: {"metacognition": int, "competitive_awareness": int, "strategic_planning": int}.
"""


def _heuristic_score(text: str) -> dict:
    text_l = (text or "").lower()
    length_points = min(len(text_l) // 120, 3)
    numeric_points = 1 if any(ch.isdigit() for ch in text_l) else 0
    comp_points = 1 if any(tok in text_l for tok in ["ssa", "llm", "compet", "market"]) else 0
    plan_points = 1 if any(tok in text_l for tok in ["next", "then", "round", "plan", "strategy"]) else 0
    total = length_points + numeric_points + comp_points + plan_points
    total = max(0, min(total, 6))
    return {
        "metacognition": total,
        "competitive_awareness": max(0, min(total - 1 + comp_points, 6)),
        "strategic_planning": max(0, min(total - 1 + plan_points, 6)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score traces and write CSV output.")
    parser.add_argument("--root", default="logs")
    parser.add_argument("--study", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--glob", dest="glob_pat", default="*.log")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model-name", default="gpt-5.4-cc")
    parser.add_argument("--secrets-path", default="assets/secrets.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_logs = {}
    for path in discover_logs(root=args.root, study=args.study, variant=args.variant):
        if args.glob_pat and not fnmatch.fnmatch(path.name, args.glob_pat):
            continue
        if args.tag and args.tag not in path.as_posix():
            continue
        try:
            run_logs[path.as_posix()] = ExperimentLog.load(str(path))
        except Exception:
            continue
    if not run_logs:
        print("No readable logs found.")
        return 0

    trace_df = batch_extract_traces(run_logs)
    if trace_df.empty:
        print("No traces found.")
        return 0

    llm = None
    if args.use_llm:
        llm = OpenAIClient(model_name=args.model_name, secrets_path=args.secrets_path, effort="low", temperature=0.1)

    score_rows = []
    for row in trace_df.itertuples(index=False):
        reasoning = getattr(row, "reasoning", "")
        if llm is None:
            scores = _heuristic_score(reasoning)
        else:
            message = [SystemMessage(TRACE_SCORING_PROMPT), HumanMessage(reasoning)]
            try:
                response = llm.invoke(message)
                scores = json.loads(response.content)
            except Exception:
                scores = _heuristic_score(reasoning)
        score_rows.append(
            {
                "run": row.run,
                "agent_id": row.agent_id,
                "round": row.round,
                **scores,
            }
        )

    scores_df = pd.DataFrame(score_rows)
    if args.output:
        out_path = Path(args.output)
    elif args.study and args.variant:
        out_path = Path(args.root) / args.study / args.variant / "analysis" / "trace_scores.csv"
    else:
        out_path = Path(args.root) / "trace_scores" / "trace_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(out_path, index=False)
    print(f"Wrote {len(scores_df)} rows -> {out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
