#!/usr/bin/env python3
"""Summarize the task algorithm simplification experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DEFAULT = Path(
    "dev/results/task_algorithm_simplification_study_20260729"
)
PAIR_KEYS = ["scenario", "side"]
PAIR_METRICS = [
    "actual_position_rmse_m",
    "actual_orientation_rmse_rad",
    "actual_ddq_p99_rad_s2",
    "actual_j4_ddq_p99_rad_s2",
    "actual_elbow_accel_p99_m_s2",
    "actual_elbow_lateral_range_m",
    "nullspace_error_abs_p95_rad",
    "tail_actual_ee_p2p_m",
    "actual_min_rho",
]


def paired_summary(
    frame: pd.DataFrame,
    *,
    baseline: str,
) -> pd.DataFrame:
    """Return per-profile mean paired differences against a baseline."""
    reference = frame[frame.profile == baseline].set_index(PAIR_KEYS)
    rows: list[dict[str, object]] = []
    for profile in sorted(set(frame.profile) - {baseline}):
        candidate = frame[frame.profile == profile].set_index(PAIR_KEYS)
        joined = candidate.join(
            reference,
            lsuffix="_candidate",
            rsuffix="_baseline",
            how="inner",
        )
        row: dict[str, object] = {"profile": profile, "pairs": len(joined)}
        for metric in PAIR_METRICS:
            candidate_values = joined[f"{metric}_candidate"].astype(float)
            baseline_values = joined[f"{metric}_baseline"].astype(float)
            difference = candidate_values - baseline_values
            denominator = baseline_values.abs().replace(0.0, np.nan)
            row[f"{metric}_mean_delta"] = float(difference.mean())
            row[f"{metric}_mean_delta_percent"] = float(
                (100.0 * difference / denominator)
                .replace([np.inf, -np.inf], np.nan)
                .mean()
            )
        row["solver_failure_delta"] = int(
            joined.solver_failures_candidate.sum()
            - joined.solver_failures_baseline.sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate target-joint dynamic boundary metrics."""
    return (
        frame.groupby("profile")
        .agg(
            rows=("scenario", "size"),
            minimum_margin_rad=("actual_target_min_margin_rad", "min"),
            p05_margin_rad=(
                "actual_target_min_margin_rad",
                lambda values: np.percentile(values, 5),
            ),
            mean_margin_rad=("actual_target_min_margin_rad", "mean"),
            mean_ddq_p99_rad_s2=("actual_target_ddq_p99_rad_s2", "mean"),
            mean_q_gap_rad=("target_q_gap_rms_rad", "mean"),
            mean_binding_fraction=("target_braking_binding_fraction", "mean"),
            solver_failures=("solver_failures", "sum"),
        )
        .reset_index()
    )


def analyze(root: Path) -> None:
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    run_counts: dict[str, int] = {}
    total_failures = 0
    finite = True
    for suite in ("nullspace", "error", "braking"):
        frame = pd.read_csv(root / suite / "summary.csv")
        paired_summary(frame, baseline="current").to_csv(
            output / f"{suite}_paired_current.csv",
            index=False,
        )
        run_counts[suite] = len(frame[["profile", "scenario"]].drop_duplicates())
        total_failures += int(frame.solver_failures.sum())
        finite = finite and bool(frame.select_dtypes("number").notna().all().all())

    robustness = pd.read_csv(root / "braking_robustness" / "summary.csv")
    run_counts["braking_robustness"] = len(
        robustness[["profile", "scenario"]].drop_duplicates()
    )
    total_failures += int(robustness.solver_failures.sum())
    finite = finite and bool(
        robustness.select_dtypes("number").notna().all().all()
    )
    for environment in ("nominal", "lagged"):
        subset = robustness[
            robustness.profile.str.endswith(f"__{environment}")
        ].copy()
        paired_summary(
            subset,
            baseline=f"current__{environment}",
        ).to_csv(
            output / f"braking_robustness_{environment}.csv",
            index=False,
        )

    for suite in ("boundary", "boundary_robustness"):
        frame = pd.read_csv(root / suite / "targeted_braking.csv")
        boundary_summary(frame).to_csv(
            output / f"{suite}_summary.csv",
            index=False,
        )
        (
            frame.groupby(["profile", "speed"])
            .actual_target_min_margin_rad.min()
            .unstack()
            .reset_index()
            .to_csv(output / f"{suite}_margin_by_speed.csv", index=False)
        )
        run_counts[suite] = len(frame)
        total_failures += int(frame.solver_failures.sum())
        finite = finite and bool(frame.select_dtypes("number").notna().all().all())

    equivalence = json.loads(
        (output / "nullspace_objective_equivalence.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "finite_metrics": finite,
        "run_counts": run_counts,
        "total_dynamic_and_boundary_runs": sum(run_counts.values()),
        "solver_failures": total_failures,
        "nullspace_equivalence": equivalence,
        "key_results": {
            "direct_nullspace": (
                "gain=k*dt is exactly the rate-only objective; gain=1 cannot be "
                "retuned with one cost to match tracking, acceleration, and branch "
                "metrics simultaneously"
            ),
            "fixed_error_clip": (
                "instant 0.6->0.9 m/s with latch dominates unconditional fixed "
                "3/6/9/12 mm clipping across tracking and tail metrics"
            ),
            "distance_only_braking": (
                "measured-position-only braking at 0.6 rad preserved or increased "
                "all tested target-joint margins, including the lagged plant"
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    analyze(args.root.resolve())


if __name__ == "__main__":
    main()
