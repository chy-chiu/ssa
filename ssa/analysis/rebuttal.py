from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _directional_claim(
    *,
    claim: str,
    baseline: pd.Series,
    treatment: pd.Series,
    expect: str,
) -> Dict:
    baseline_mean = float(baseline.mean()) if len(baseline) else np.nan
    treatment_mean = float(treatment.mean()) if len(treatment) else np.nan
    effect = treatment_mean - baseline_mean
    if expect == "lower":
        passed = bool(effect < 0)
    else:
        passed = bool(effect > 0)
    return {
        "claim": claim,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "effect_size": effect,
        "direction": expect,
        "pass": passed,
    }


def _scale_persistence(df: pd.DataFrame) -> Tuple[bool, float]:
    if "ssa_minus_control_reward" not in df.columns or df.empty:
        return False, np.nan
    score = float((df["ssa_minus_control_reward"] > 0).mean())
    return bool(score >= 0.7), score


def evaluate_rebuttal_claims(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Directional pass/fail checks used for reviewer-facing rebuttal summaries.
    Expected columns (best-effort): open_bidding, performance_pay, mean_winning_bid_ratio,
    train_rate, ssa_minus_control_reward.
    """
    rows = []

    if {"open_bidding", "mean_winning_bid_ratio"}.issubset(run_df.columns):
        baseline = run_df.loc[run_df["open_bidding"] == False, "mean_winning_bid_ratio"]
        treatment = run_df.loc[run_df["open_bidding"] == True, "mean_winning_bid_ratio"]
        if len(baseline) and len(treatment):
            rows.append(
                _directional_claim(
                    claim="open bidding lowers prices",
                    baseline=baseline,
                    treatment=treatment,
                    expect="lower",
                )
            )

    if {"performance_pay", "train_rate"}.issubset(run_df.columns):
        baseline = run_df.loc[run_df["performance_pay"] == False, "train_rate"]
        treatment = run_df.loc[run_df["performance_pay"] == True, "train_rate"]
        if len(baseline) and len(treatment):
            rows.append(
                _directional_claim(
                    claim="performance pay increases training",
                    baseline=baseline,
                    treatment=treatment,
                    expect="higher",
                )
            )

    if "ssa_minus_control_reward" in run_df.columns and len(run_df):
        baseline = pd.Series([0.0] * len(run_df))
        treatment = run_df["ssa_minus_control_reward"]
        rows.append(
            _directional_claim(
                claim="SSA > controls",
                baseline=baseline,
                treatment=treatment,
                expect="higher",
            )
        )

    persist_pass, persist_score = _scale_persistence(run_df)
    rows.append(
        {
            "claim": "trends persist under scale/robustness variants",
            "baseline_mean": np.nan,
            "treatment_mean": np.nan,
            "effect_size": persist_score,
            "direction": "higher",
            "pass": persist_pass,
        }
    )

    return pd.DataFrame(rows)
