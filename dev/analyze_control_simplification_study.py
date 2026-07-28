#!/usr/bin/env python3
"""Summarize the dedicated control-parameter simplification experiments."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["scenario", "side"]
METRICS = [
    "actual_position_rmse_m",
    "actual_orientation_rmse_rad",
    "actual_ddq_p99_rad_s2",
    "actual_j1_ddq_p99_rad_s2",
    "actual_j4_ddq_p99_rad_s2",
    "actual_elbow_accel_p99_m_s2",
    "actual_elbow_lateral_range_m",
    "nullspace_error_abs_p95_rad",
    "actual_min_rho",
    "actual_min_joint_margin_rad",
    "frame_used_error_p95_m",
    "q_gap_rms_rad",
    "tail_actual_ee_vibration_rms_m",
    "solve_time_mean_ms",
]


def paired_table(
    frame: pd.DataFrame,
    *,
    reference: str = "current",
) -> pd.DataFrame:
    """Return absolute means and paired mean changes from one reference."""
    baseline = frame[frame.profile == reference].set_index(KEYS)
    rows: list[dict[str, float | str]] = []
    for profile in frame.profile.drop_duplicates():
        current = frame[frame.profile == profile].set_index(KEYS).loc[baseline.index]
        row: dict[str, float | str] = {"profile": profile}
        for metric in METRICS:
            base_values = baseline[metric].astype(float)
            values = current[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_delta"] = float((values - base_values).mean())
            denominator = base_values.abs().where(base_values.abs() > 1e-12)
            row[f"{metric}_pct"] = float(
                ((values - base_values) / denominator).mean() * 100.0
            )
        row["solver_failures"] = float(current.solver_failures.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair each compact candidate with current under the same plant model."""
    environments = (
        "nominal",
        "delay24ms",
        "actuator0p7",
        "delay24ms_actuator0p7",
    )
    rows: list[dict[str, float | str]] = []
    for environment in environments:
        reference_name = f"current__{environment}"
        baseline = frame[frame.profile == reference_name].set_index(KEYS)
        for candidate in ("simple", "simple_energy5e5"):
            profile = f"{candidate}__{environment}"
            current = frame[frame.profile == profile].set_index(KEYS).loc[
                baseline.index
            ]
            row: dict[str, float | str] = {
                "environment": environment,
                "candidate": candidate,
            }
            for metric in METRICS:
                base_values = baseline[metric].astype(float)
                values = current[metric].astype(float)
                row[f"{metric}_delta"] = float((values - base_values).mean())
                denominator = base_values.abs().where(base_values.abs() > 1e-12)
                row[f"{metric}_pct"] = float(
                    ((values - base_values) / denominator).mean() * 100.0
                )
            row["solver_failures"] = float(current.solver_failures.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def boundary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize target-joint braking margins without global-margin confounds."""
    metrics = [
        "command_target_min_margin_rad",
        "actual_target_min_margin_rad",
        "dynamic_overshoot_beyond_command_rad",
        "command_target_approach_dq_max_rad_s",
        "actual_target_approach_dq_max_rad_s",
        "actual_target_ddq_p99_rad_s2",
        "target_q_gap_rms_rad",
        "target_braking_active_fraction",
        "target_braking_binding_fraction",
    ]
    rows: list[dict[str, float | str]] = []
    for profile, group in frame.groupby("profile", sort=False):
        worst = group.loc[group.actual_target_min_margin_rad.idxmin()]
        row: dict[str, float | str] = {
            "profile": profile,
            "solver_failures": float(group.solver_failures.sum()),
            "worst_scenario": str(worst.scenario),
            "worst_joint": float(worst.joint),
            "worst_actual_target_margin_rad": float(
                worst.actual_target_min_margin_rad
            ),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def ik_param_fields(source: str) -> list[str]:
    """Extract dataclass field names from IKParams."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "IKParams":
            return [
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            ]
    raise ValueError("IKParams was not found.")


def parameter_surface(repo: Path) -> dict[str, object]:
    """Describe the main-to-current public parameter growth."""
    current_source = (repo / "src/openarm_control/kinematics.py").read_text()
    main_source = subprocess.check_output(
        ["git", "show", "main:src/openarm_control/kinematics.py"],
        cwd=repo,
        text=True,
    )
    main_fields = ik_param_fields(main_source)
    current_fields = ik_param_fields(current_source)
    return {
        "main_count": len(main_fields),
        "current_count": len(current_fields),
        "added_count": len(set(current_fields) - set(main_fields)),
        "main_fields": main_fields,
        "current_fields": current_fields,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("dev/results/control_simplification_study_20260729"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    study_dir = (repo / args.study_dir).resolve()
    output = study_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    components = pd.read_csv(study_dir / "components/summary.csv")
    retune = pd.read_csv(study_dir / "retune/summary.csv")
    robustness = pd.read_csv(study_dir / "robustness/summary.csv")
    boundary = pd.read_csv(study_dir / "boundary/targeted_braking.csv")

    component_pairs = paired_table(components)
    retune_pairs = paired_table(retune)
    robust_pairs = robustness_table(robustness)
    boundary_summary = boundary_table(boundary)

    component_pairs.to_csv(output / "components_paired_current.csv", index=False)
    retune_pairs.to_csv(output / "retune_paired_current.csv", index=False)
    robust_pairs.to_csv(output / "robustness_paired_current.csv", index=False)
    boundary_summary.to_csv(output / "boundary_summary.csv", index=False)

    summary = {
        "row_counts": {
            "components": len(components),
            "retune": len(retune),
            "robustness": len(robustness),
            "boundary": len(boundary),
            "total": len(components) + len(retune) + len(robustness) + len(boundary),
        },
        "solver_failures": {
            "components": int(components.solver_failures.sum()),
            "retune": int(retune.solver_failures.sum()),
            "robustness": int(robustness.solver_failures.sum()),
            "boundary": int(boundary.solver_failures.sum()),
        },
        "parameter_surface": parameter_surface(repo),
        "finite_metrics": all(
            np.isfinite(frame.select_dtypes("number").to_numpy()).all()
            for frame in (components, retune, robustness, boundary)
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
