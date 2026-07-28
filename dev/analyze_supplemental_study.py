#!/usr/bin/env python3
"""Analyze supplemental IK safety sweeps and write compact comparison tables."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("null", "sing", "brake", "error", "energy")
PAIR_KEYS = ["scenario", "side"]
KEY_METRICS = [
    "solver_failures",
    "command_position_rmse_m",
    "actual_position_rmse_m",
    "command_orientation_rmse_rad",
    "actual_orientation_rmse_rad",
    "q_gap_rms_rad",
    "command_dq_p99_rad_s",
    "actual_dq_p99_rad_s",
    "command_ddq_p99_rad_s2",
    "actual_ddq_p99_rad_s2",
    "command_j1_dq_max_rad_s",
    "actual_j1_dq_max_rad_s",
    "command_j1_ddq_p99_rad_s2",
    "actual_j1_ddq_p99_rad_s2",
    "command_j4_dq_max_rad_s",
    "actual_j4_dq_max_rad_s",
    "command_j4_ddq_p99_rad_s2",
    "actual_j4_ddq_p99_rad_s2",
    "command_elbow_accel_p99_m_s2",
    "actual_elbow_accel_p99_m_s2",
    "command_elbow_lateral_range_m",
    "actual_elbow_lateral_range_m",
    "command_min_rho",
    "actual_min_rho",
    "command_min_joint_margin_rad",
    "actual_min_joint_margin_rad",
    "command_velocity_saturation_fraction",
    "actual_velocity_saturation_fraction",
    "torque_saturation_fraction",
    "tail_command_ee_vibration_rms_m",
    "tail_actual_ee_vibration_rms_m",
]


def _load(root: Path, suite: str) -> pd.DataFrame:
    path = root / suite / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in KEY_METRICS if metric in frame]


def profile_means(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = _metric_columns(frame)
    return frame.groupby("profile", as_index=False)[metrics].mean()


def paired_comparison(
    frame: pd.DataFrame,
    *,
    reference: str,
) -> pd.DataFrame:
    metrics = _metric_columns(frame)
    reference_frame = frame[frame.profile == reference][
        [*PAIR_KEYS, *metrics]
    ].copy()
    rows: list[dict[str, object]] = []
    for profile, selected in frame.groupby("profile"):
        merged = selected[[*PAIR_KEYS, *metrics]].merge(
            reference_frame,
            on=PAIR_KEYS,
            suffixes=("", "_reference"),
            validate="one_to_one",
        )
        row: dict[str, object] = {
            "profile": profile,
            "reference": reference,
            "pairs": len(merged),
        }
        for metric in metrics:
            current = merged[metric].to_numpy(dtype=float)
            baseline = merged[f"{metric}_reference"].to_numpy(dtype=float)
            difference = current - baseline
            denominator = np.maximum(np.abs(baseline), 1e-12)
            row[f"{metric}_delta"] = float(np.mean(difference))
            row[f"{metric}_pct"] = float(
                100.0 * np.mean(difference / denominator)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_factor_profile(name: str) -> dict[str, bool]:
    tokens = name.removeprefix("factor_").split("_")
    parsed: dict[str, bool] = {}
    for feature, token in zip(FEATURES, tokens, strict=True):
        parsed[feature] = token == feature
    return parsed


def factorial_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = frame.copy()
    for feature in FEATURES:
        expanded[feature] = expanded.profile.map(
            lambda name, feature=feature: _parse_factor_profile(name)[feature]
        )
    metrics = _metric_columns(expanded)
    main_rows: list[dict[str, object]] = []
    for feature in FEATURES:
        other = [value for value in FEATURES if value != feature]
        group_keys = [*PAIR_KEYS, *other]
        for metric in metrics:
            pivot = expanded.pivot_table(
                index=group_keys,
                columns=feature,
                values=metric,
                aggfunc="first",
            ).dropna()
            difference = pivot[True] - pivot[False]
            denominator = np.maximum(np.abs(pivot[False]), 1e-12)
            main_rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "pairs": len(pivot),
                    "effect_abs": float(difference.mean()),
                    "effect_pct": float(
                        100.0 * np.mean(difference / denominator)
                    ),
                    "effect_pct_median": float(
                        100.0 * np.median(difference / denominator)
                    ),
                }
            )

    interaction_rows: list[dict[str, object]] = []
    for first, second in itertools.combinations(FEATURES, 2):
        other = [
            value
            for value in FEATURES
            if value not in (first, second)
        ]
        group_keys = [*PAIR_KEYS, *other]
        for metric in metrics:
            pivot = expanded.pivot_table(
                index=group_keys,
                columns=[first, second],
                values=metric,
                aggfunc="first",
            ).dropna()
            interaction = (
                pivot[(True, True)]
                - pivot[(True, False)]
                - pivot[(False, True)]
                + pivot[(False, False)]
            )
            denominator = np.maximum(
                np.abs(pivot[(False, False)]),
                1e-12,
            )
            interaction_rows.append(
                {
                    "feature_a": first,
                    "feature_b": second,
                    "metric": metric,
                    "groups": len(pivot),
                    "interaction_abs": float(interaction.mean()),
                    "interaction_pct": float(
                        100.0 * np.mean(interaction / denominator)
                    ),
                }
            )
    return pd.DataFrame(main_rows), pd.DataFrame(interaction_rows)


def command_actual_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("position_rmse", "command_position_rmse_m", "actual_position_rmse_m"),
        (
            "orientation_rmse",
            "command_orientation_rmse_rad",
            "actual_orientation_rmse_rad",
        ),
        ("dq_p99", "command_dq_p99_rad_s", "actual_dq_p99_rad_s"),
        ("ddq_p99", "command_ddq_p99_rad_s2", "actual_ddq_p99_rad_s2"),
        (
            "elbow_accel",
            "command_elbow_accel_p99_m_s2",
            "actual_elbow_accel_p99_m_s2",
        ),
        (
            "elbow_lateral",
            "command_elbow_lateral_range_m",
            "actual_elbow_lateral_range_m",
        ),
        (
            "tail_vibration",
            "tail_command_ee_vibration_rms_m",
            "tail_actual_ee_vibration_rms_m",
        ),
    ]
    rows: list[dict[str, object]] = []
    for (profile, family), selected in frame.groupby(["profile", "family"]):
        row: dict[str, object] = {
            "profile": profile,
            "family": family,
            "arms": len(selected),
            "q_gap_rms_rad": float(selected.q_gap_rms_rad.mean()),
        }
        for label, command_metric, actual_metric in pairs:
            if command_metric not in selected or actual_metric not in selected:
                continue
            command = selected[command_metric].to_numpy(dtype=float)
            actual = selected[actual_metric].to_numpy(dtype=float)
            row[f"{label}_actual_over_command"] = float(
                np.mean(actual / np.maximum(np.abs(command), 1e-12))
            )
            row[f"{label}_actual_minus_command"] = float(
                np.mean(actual - command)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def joint_limit_table(frame: pd.DataFrame) -> pd.DataFrame:
    extracted = frame.scenario.str.extract(
        r"joint(?P<joint>\d+)_(?P<boundary>lower|upper)_"
    )
    expanded = frame.assign(
        joint=extracted.joint.astype(int),
        boundary=extracted.boundary,
    )
    metrics = [
        "solver_failures",
        "command_position_rmse_m",
        "actual_position_rmse_m",
        "command_min_joint_margin_rad",
        "actual_min_joint_margin_rad",
        "command_dq_p99_rad_s",
        "actual_dq_p99_rad_s",
        "command_ddq_p99_rad_s2",
        "actual_ddq_p99_rad_s2",
        "braking_active_fraction",
        "braking_binding_fraction",
        "tail_actual_ee_vibration_rms_m",
    ]
    return expanded.groupby(
        ["profile", "joint", "boundary"],
        as_index=False,
    )[metrics].mean()


def robustness_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    paired = paired_comparison(frame, reference="current")
    score = np.zeros(len(paired), dtype=float)
    lower_is_better = {
        "actual_position_rmse_m": 2.0,
        "actual_orientation_rmse_rad": 1.0,
        "q_gap_rms_rad": 2.0,
        "actual_ddq_p99_rad_s2": 1.0,
        "actual_elbow_accel_p99_m_s2": 1.0,
        "tail_actual_ee_vibration_rms_m": 1.0,
    }
    for metric, weight in lower_is_better.items():
        column = f"{metric}_pct"
        if column in paired:
            score += weight * paired[column].to_numpy(dtype=float)
    paired["degradation_score"] = score / sum(lower_is_better.values())
    return paired.sort_values("degradation_score")


def finalist_tables(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split finalist profile names into controller and plant dimensions."""
    expanded = frame.copy()
    split = expanded.profile.str.split("__", n=1, expand=True)
    expanded["algorithm"] = split[0]
    expanded["plant"] = split[1]
    metrics = _metric_columns(expanded)
    aggregate = expanded.groupby(
        ["plant", "algorithm"],
        as_index=False,
    )[metrics].mean()
    by_family = expanded.groupby(
        ["plant", "family", "algorithm"],
        as_index=False,
    )[metrics].mean()

    paired_rows: list[dict[str, object]] = []
    for plant, plant_frame in expanded.groupby("plant"):
        reference = plant_frame[plant_frame.algorithm == "current"][
            [*PAIR_KEYS, *metrics]
        ]
        for algorithm, selected in plant_frame.groupby("algorithm"):
            merged = selected[[*PAIR_KEYS, *metrics]].merge(
                reference,
                on=PAIR_KEYS,
                suffixes=("", "_reference"),
                validate="one_to_one",
            )
            row: dict[str, object] = {
                "plant": plant,
                "algorithm": algorithm,
                "pairs": len(merged),
            }
            for metric in metrics:
                difference = (
                    merged[metric].to_numpy(dtype=float)
                    - merged[f"{metric}_reference"].to_numpy(dtype=float)
                )
                row[f"{metric}_delta"] = float(np.mean(difference))
            paired_rows.append(row)
    return aggregate, by_family, pd.DataFrame(paired_rows)


def _write_json_summary(
    output: Path,
    frames: dict[str, pd.DataFrame],
) -> None:
    payload: dict[str, object] = {}
    for suite, frame in frames.items():
        payload[suite] = {
            "rows": len(frame),
            "profiles": int(frame.profile.nunique()),
            "scenarios": int(frame.scenario.nunique()),
            "solver_failures": int(frame.solver_failures.sum()),
        }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suites = (
        "base_qp",
        "interactions",
        "joint_limits",
        "braking_dynamics",
        "randomized",
        "robustness",
        "replay",
        "finalists",
    )
    frames = {
        suite: _load(root, suite)
        for suite in suites
        if (root / suite / "summary.csv").exists()
    }
    for suite, frame in frames.items():
        profile_means(frame).to_csv(
            output / f"{suite}_profile_means.csv",
            index=False,
        )
        command_actual_ratios(frame).to_csv(
            output / f"{suite}_command_actual.csv",
            index=False,
        )
        if "current" in set(frame.profile):
            paired_comparison(frame, reference="current").to_csv(
                output / f"{suite}_paired_current.csv",
                index=False,
            )

    if "interactions" in frames:
        main_effects, interactions = factorial_effects(frames["interactions"])
        main_effects.to_csv(output / "factorial_main_effects.csv", index=False)
        interactions.to_csv(
            output / "factorial_pair_interactions.csv",
            index=False,
        )
    if "joint_limits" in frames:
        joint_limit_table(frames["joint_limits"]).to_csv(
            output / "joint_limits_by_joint.csv",
            index=False,
        )
    if "braking_dynamics" in frames:
        joint_limit_table(frames["braking_dynamics"]).to_csv(
            output / "braking_dynamics_by_joint.csv",
            index=False,
        )
    if "robustness" in frames:
        robustness_ranking(frames["robustness"]).to_csv(
            output / "robustness_ranking.csv",
            index=False,
        )
    if "finalists" in frames:
        aggregate, by_family, paired = finalist_tables(frames["finalists"])
        aggregate.to_csv(output / "finalists_by_plant.csv", index=False)
        by_family.to_csv(
            output / "finalists_by_plant_family.csv",
            index=False,
        )
        paired.to_csv(
            output / "finalists_paired_current_by_plant.csv",
            index=False,
        )
    _write_json_summary(output, frames)


if __name__ == "__main__":
    main()
