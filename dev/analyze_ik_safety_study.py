#!/usr/bin/env python3
"""Analyze the reproducible IK safety simulation matrices and traces."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PAIR_KEYS = ["scenario", "side"]
KEY_METRICS = [
    "actual_position_rmse_m",
    "actual_orientation_rmse_rad",
    "q_gap_rms_rad",
    "actual_ddq_p99_rad_s2",
    "actual_j1_ddq_p99_rad_s2",
    "actual_j4_ddq_p99_rad_s2",
    "actual_elbow_accel_p99_m_s2",
    "actual_elbow_lateral_range_m",
    "actual_min_rho",
    "actual_min_joint_margin_rad",
    "tail_actual_ee_vibration_rms_m",
    "nullspace_error_abs_p95_rad",
]
SHORT_METRIC_NAMES = {
    "actual_position_rmse_m": "position RMSE",
    "actual_orientation_rmse_rad": "orientation RMSE",
    "q_gap_rms_rad": "q command-state gap",
    "actual_ddq_p99_rad_s2": "joint accel p99",
    "actual_j1_ddq_p99_rad_s2": "J1 accel p99",
    "actual_j4_ddq_p99_rad_s2": "J4 accel p99",
    "actual_elbow_accel_p99_m_s2": "elbow accel p99",
    "actual_elbow_lateral_range_m": "elbow lateral range",
    "actual_min_rho": "minimum singularity ratio",
    "actual_min_joint_margin_rad": "minimum joint margin",
    "tail_actual_ee_vibration_rms_m": "tail EE vibration",
    "nullspace_error_abs_p95_rad": "nullspace home error",
}


def load_summary(root: Path, suite: str) -> pd.DataFrame | None:
    path = root / suite / "summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["suite"] = suite
    return frame


def paired_percent_changes(
    frame: pd.DataFrame,
    *,
    baseline: str = "current",
    metrics: Iterable[str] = KEY_METRICS,
    by_family: bool = False,
) -> pd.DataFrame:
    """Compute mean paired percentage changes against one profile."""
    baseline_rows = frame[frame["profile"] == baseline]
    output: list[dict[str, object]] = []
    for profile in sorted(set(frame["profile"]) - {baseline}):
        compared = frame[frame["profile"] == profile]
        merged = baseline_rows.merge(
            compared,
            on=PAIR_KEYS,
            suffixes=("_baseline", "_candidate"),
        )
        if merged.empty:
            continue
        groups = (
            merged.groupby("family_baseline", dropna=False)
            if by_family
            else [("all", merged)]
        )
        for family, group in groups:
            row: dict[str, object] = {
                "profile": profile,
                "family": family,
                "paired_rows": len(group),
            }
            for metric in metrics:
                before = group[f"{metric}_baseline"].to_numpy(dtype=float)
                after = group[f"{metric}_candidate"].to_numpy(dtype=float)
                valid = np.isfinite(before) & np.isfinite(after)
                denominator = np.maximum(np.abs(before), 1e-12)
                row[f"{metric}_pct"] = float(
                    np.mean(100.0 * (after[valid] - before[valid]) / denominator[valid])
                )
                row[f"{metric}_abs"] = float(np.mean(after[valid] - before[valid]))
            output.append(row)
    return pd.DataFrame(output)


def profile_means(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "q_gap_rms_rad",
        "actual_ddq_p99_rad_s2",
        "command_ddq_p99_rad_s2",
        "actual_j1_ddq_p99_rad_s2",
        "actual_j4_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "actual_elbow_lateral_range_m",
        "actual_min_rho",
        "actual_min_joint_margin_rad",
        "frame_limit_activation_mean",
        "nullspace_error_abs_p95_rad",
        "command_velocity_saturation_fraction",
        "braking_binding_fraction",
        "solver_failures",
        "solve_time_mean_ms",
    ]
    available = [column for column in columns if column in frame]
    return frame.groupby("profile", sort=True)[available].mean().reset_index()


def save_screening_heatmap(changes: pd.DataFrame, output: Path) -> None:
    selected_profiles = [
        "no_nullspace",
        "no_singularity_limit",
        "no_joint_braking",
        "no_error_limit",
        "no_kinetic_energy",
        "velocity_only",
        "home_posture_0p01",
        "home_posture_0p1",
        "command_only_safety",
    ]
    selected_metrics = [
        "actual_position_rmse_m",
        "actual_ddq_p99_rad_s2",
        "actual_j1_ddq_p99_rad_s2",
        "actual_j4_ddq_p99_rad_s2",
        "actual_elbow_lateral_range_m",
        "actual_min_rho",
        "actual_min_joint_margin_rad",
        "tail_actual_ee_vibration_rms_m",
    ]
    table = (
        changes[changes["profile"].isin(selected_profiles)]
        .set_index("profile")[[f"{metric}_pct" for metric in selected_metrics]]
        .reindex(selected_profiles)
    )
    table.columns = [SHORT_METRIC_NAMES[metric] for metric in selected_metrics]
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        table,
        center=0.0,
        cmap="vlag",
        robust=True,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "paired change from current (%)"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Feature ablations: paired aggregate effects")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_joint_activity_heatmaps(frame: pd.DataFrame, output: Path) -> None:
    current = frame[frame["profile"] == "current"]
    saturation = current.groupby("family")[
        [f"command_j{index}_sat_fraction" for index in range(1, 8)]
    ].mean()
    braking = current.groupby("family")[
        [f"braking_j{index}_binding_fraction" for index in range(1, 8)]
    ].mean()
    saturation.columns = [f"J{index}" for index in range(1, 8)]
    braking.columns = [f"J{index}" for index in range(1, 8)]

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    sns.heatmap(
        saturation,
        cmap="mako",
        vmin=0.0,
        vmax=max(0.01, float(saturation.to_numpy().max())),
        annot=True,
        fmt=".2f",
        ax=axes[0],
        cbar_kws={"label": "fraction"},
    )
    axes[0].set_title("Current controller: hard velocity saturation")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    sns.heatmap(
        braking,
        cmap="rocket",
        vmin=0.0,
        vmax=max(0.01, float(braking.to_numpy().max())),
        annot=True,
        fmt=".2f",
        ax=axes[1],
        cbar_kws={"label": "fraction"},
    )
    axes[1].set_title("Current controller: position-braking constraint binding")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_parameter_tradeoff(frame: pd.DataFrame, output: Path) -> None:
    names = [
        "current",
        "energy_0",
        "energy_3em05",
        "energy_0p0001",
        "energy_0p0003",
        "energy_0p001",
        "velocity_scale_0p75",
        "velocity_scale_1p25",
        "velocity_scale_1p5",
        "brake_distance_0p2",
        "brake_distance_0p35",
        "brake_distance_0p5",
        "brake_distance_0p7",
    ]
    means = profile_means(frame)
    means = means[means["profile"].isin(names)]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(
        1e3 * means["actual_position_rmse_m"],
        means["actual_ddq_p99_rad_s2"],
        c=means["actual_min_joint_margin_rad"],
        cmap="viridis",
        s=70,
    )
    for row in means.itertuples():
        axis.annotate(
            row.profile,
            (1e3 * row.actual_position_rmse_m, row.actual_ddq_p99_rad_s2),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("actual position RMSE (mm)")
    axis.set_ylabel("actual joint acceleration p99 (rad/s2)")
    axis.set_title("Tracking/smoothness tradeoff; color is joint-limit margin")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _find_trace(trace_dir: Path, profile: str, scenario: str) -> Path | None:
    matches = sorted(trace_dir.glob(f"*_{profile}_{scenario}.npz"))
    return matches[-1] if matches else None


def _active_trace_values(
    trace: np.lib.npyio.NpzFile,
    key: str,
) -> np.ndarray:
    active = trace["phase"] == 1
    if not np.any(active):
        active = trace["phase"] > 0
    return np.asarray(trace[key])[active]


def branch_reference_distances(
    root: Path,
    *,
    suite: str = "branch",
    reference_profile: str = "unlimited_reference",
    reference_suite: str | None = None,
) -> pd.DataFrame:
    """Compare each branch profile against the no-joint-rate reference."""
    trace_dir = root / suite / "traces"
    reference_trace_dir = root / (reference_suite or suite) / "traces"
    metadata_path = root / suite / "metadata.json"
    if not metadata_path.exists():
        return pd.DataFrame()
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    profiles = [item["name"] for item in metadata["profiles"]]
    scenarios = [item["name"] for item in metadata["scenarios"]]
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        reference_path = _find_trace(
            reference_trace_dir,
            reference_profile,
            scenario,
        )
        if reference_path is None:
            continue
        with np.load(reference_path) as reference:
            reference_q = _active_trace_values(reference, "right_actual_q")
            reference_elbow = _active_trace_values(
                reference,
                "right_actual_elbow",
            )
            reference_pose = _active_trace_values(
                reference,
                "right_actual_pose",
            )
            for profile in profiles:
                candidate_path = _find_trace(trace_dir, profile, scenario)
                if candidate_path is None:
                    continue
                with np.load(candidate_path) as candidate:
                    candidate_q = _active_trace_values(
                        candidate,
                        "right_actual_q",
                    )
                    candidate_elbow = _active_trace_values(
                        candidate,
                        "right_actual_elbow",
                    )
                    candidate_pose = _active_trace_values(
                        candidate,
                        "right_actual_pose",
                    )
                    count = min(reference_q.shape[0], candidate_q.shape[0])
                    q_delta = candidate_q[:count] - reference_q[:count]
                    elbow_delta = (
                        candidate_elbow[:count] - reference_elbow[:count]
                    )
                    pose_delta = candidate_pose[:count, :3] - reference_pose[
                        :count, :3
                    ]
                    rows.append(
                        {
                            "profile": profile,
                            "scenario": scenario,
                            "q_rms_rad": float(
                                np.sqrt(np.mean(np.square(q_delta)))
                            ),
                            "j1_rms_rad": float(
                                np.sqrt(np.mean(np.square(q_delta[:, 0])))
                            ),
                            "j4_rms_rad": float(
                                np.sqrt(np.mean(np.square(q_delta[:, 3])))
                            ),
                            "elbow_rms_m": float(
                                np.sqrt(np.mean(np.square(elbow_delta)))
                            ),
                            "elbow_lateral_rms_m": float(
                                np.sqrt(np.mean(np.square(elbow_delta[:, 1])))
                            ),
                            "ee_position_rms_m": float(
                                np.sqrt(np.mean(np.square(pose_delta)))
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _resample_motion(
    trace: np.lib.npyio.NpzFile,
    key: str,
    progress_grid: np.ndarray,
) -> np.ndarray:
    active = trace["phase"] == 1
    target = np.asarray(trace["right_target_pose"])[active, :3]
    values = np.asarray(trace[key])[active]
    increments = np.linalg.norm(np.diff(target, axis=0), axis=1)
    progress = np.concatenate([[0.0], np.cumsum(increments)])
    if progress[-1] <= 1e-12:
        progress = np.linspace(0.0, 1.0, values.shape[0])
    else:
        progress /= progress[-1]
    progress, unique_indices = np.unique(progress, return_index=True)
    values = values[unique_indices]
    return np.column_stack(
        [
            np.interp(progress_grid, progress, values[:, index])
            for index in range(values.shape[1])
        ]
    )


def branch_speed_consistency(
    root: Path,
    *,
    suite: str = "branch",
) -> pd.DataFrame:
    """Compare fast retract branches to the 0.1 m/s path at equal progress."""
    trace_dir = root / suite / "traces"
    metadata_path = root / suite / "metadata.json"
    if not metadata_path.exists():
        return pd.DataFrame()
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    profiles = [item["name"] for item in metadata["profiles"]]
    grid = np.linspace(0.0, 1.0, 301)
    rows: list[dict[str, object]] = []
    for lateral in ("m0p10", "p0p10"):
        slow_scenario = f"retract_diag_{lateral}_v0p10"
        for profile in profiles:
            slow_path = _find_trace(trace_dir, profile, slow_scenario)
            if slow_path is None:
                continue
            with np.load(slow_path) as slow:
                slow_q = _resample_motion(
                    slow,
                    "right_actual_q",
                    grid,
                )
                slow_elbow = _resample_motion(
                    slow,
                    "right_actual_elbow",
                    grid,
                )
                for speed in ("0p40", "0p80"):
                    scenario = f"retract_diag_{lateral}_v{speed}"
                    fast_path = _find_trace(trace_dir, profile, scenario)
                    if fast_path is None:
                        continue
                    with np.load(fast_path) as fast:
                        fast_q = _resample_motion(
                            fast,
                            "right_actual_q",
                            grid,
                        )
                        fast_elbow = _resample_motion(
                            fast,
                            "right_actual_elbow",
                            grid,
                        )
                    q_delta = fast_q - slow_q
                    elbow_delta = fast_elbow - slow_elbow
                    rows.append(
                        {
                            "profile": profile,
                            "lateral": lateral,
                            "speed": float(speed.replace("p", ".")),
                            "q_path_rms_rad": float(
                                np.sqrt(np.mean(np.square(q_delta)))
                            ),
                            "j1_path_rms_rad": float(
                                np.sqrt(np.mean(np.square(q_delta[:, 0])))
                            ),
                            "j4_path_rms_rad": float(
                                np.sqrt(np.mean(np.square(q_delta[:, 3])))
                            ),
                            "elbow_path_rms_m": float(
                                np.sqrt(np.mean(np.square(elbow_delta)))
                            ),
                            "elbow_lateral_path_rms_m": float(
                                np.sqrt(np.mean(np.square(elbow_delta[:, 1])))
                            ),
                            "elbow_lateral_path_max_m": float(
                                np.max(np.abs(elbow_delta[:, 1]))
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def save_branch_comparison(
    root: Path,
    reference: pd.DataFrame,
    consistency: pd.DataFrame,
    output: Path,
) -> None:
    profiles = [
        "current",
        "no_joint_braking_current_tasks",
        "velocity_scale_1p25_current_tasks",
        "brake_distance_0p10",
        "brake_per_joint_compact",
        "candidate_compact",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    retract_reference = reference[
        reference["scenario"].str.startswith("retract_diag")
        & reference["scenario"].str.endswith("v0p80")
        & reference["profile"].isin(profiles)
    ]
    sns.barplot(
        data=retract_reference,
        x="profile",
        y="elbow_lateral_rms_m",
        ax=axes[0],
        color="#4C78A8",
    )
    axes[0].set_title("Fast retract vs unlimited branch reference")
    axes[0].set_ylabel("elbow lateral path RMS (m)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)

    retract_consistency = consistency[
        (consistency["speed"] == 0.8)
        & consistency["profile"].isin(profiles)
    ]
    sns.barplot(
        data=retract_consistency,
        x="profile",
        y="elbow_lateral_path_rms_m",
        ax=axes[1],
        color="#F58518",
    )
    axes[1].set_title("0.8 m/s retract vs same-profile 0.1 m/s branch")
    axes[1].set_ylabel("elbow lateral path RMS (m)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_candidate_tradeoff(
    summary: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
) -> None:
    means = profile_means(summary)
    retract_reference = reference[
        reference["scenario"].str.startswith("retract_diag")
        & reference["scenario"].str.endswith("v0p80")
    ]
    branch_means = (
        retract_reference.groupby("profile", as_index=False)
        .mean(numeric_only=True)
        .loc[:, ["profile", "q_rms_rad", "elbow_lateral_rms_m"]]
    )
    merged = means.merge(branch_means, on="profile", how="inner")
    merged = merged[merged["profile"] != "no_velocity_current_tasks"]
    if merged.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(
        data=merged,
        x="q_rms_rad",
        y="actual_ddq_p99_rad_s2",
        hue="actual_position_rmse_m",
        palette="viridis_r",
        s=90,
        ax=axes[0],
    )
    sns.scatterplot(
        data=merged,
        x="elbow_lateral_rms_m",
        y="actual_j1_ddq_p99_rad_s2",
        hue="actual_position_rmse_m",
        palette="viridis_r",
        s=90,
        legend=False,
        ax=axes[1],
    )
    for row in merged.itertuples():
        axes[0].annotate(
            row.profile,
            (row.q_rms_rad, row.actual_ddq_p99_rad_s2),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
        axes[1].annotate(
            row.profile,
            (row.elbow_lateral_rms_m, row.actual_j1_ddq_p99_rad_s2),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].set_title("Candidate smoothness vs unlimited-velocity branch")
    axes[0].set_xlabel("joint-path RMS from unlimited reference (rad)")
    axes[0].set_ylabel("actual joint acceleration p99 (rad/s2)")
    axes[1].set_title("Elbow branch vs J1 acceleration")
    axes[1].set_xlabel("elbow lateral RMS from unlimited reference (m)")
    axes[1].set_ylabel("actual J1 acceleration p99 (rad/s2)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_iteration_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    means = profile_means(summary)
    rows: list[dict[str, object]] = []
    for row in means.itertuples():
        name = str(row.profile)
        if name == "current":
            family = "native scheduled"
            iterations = 5
        elif "_iters_" in name:
            prefix, value = name.rsplit("_iters_", 1)
            family = {
                "native_3mm": "native scheduled",
                "always_3mm": "fixed 3 mm/substep",
                "fixed_15mm_total": "fixed 15 mm/cycle",
                "no_error_limit": "unlimited error",
            }.get(prefix, prefix)
            iterations = int(value)
        else:
            continue
        rows.append(
            {
                "family": family,
                "iterations": iterations,
                "position_rmse_mm": 1e3 * row.actual_position_rmse_m,
                "actual_ddq_p99_rad_s2": row.actual_ddq_p99_rad_s2,
                "command_ddq_p99_rad_s2": row.command_ddq_p99_rad_s2,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame = frame.drop_duplicates(["family", "iterations"], keep="first")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = (
        ("position_rmse_mm", "position RMSE (mm)"),
        ("actual_ddq_p99_rad_s2", "actual acceleration p99 (rad/s2)"),
        ("command_ddq_p99_rad_s2", "command acceleration p99 (rad/s2)"),
    )
    for axis, (key, label) in zip(axes, keys, strict=True):
        sns.lineplot(
            data=frame,
            x="iterations",
            y=key,
            hue="family",
            marker="o",
            ax=axis,
        )
        axis.set_xticks([1, 2, 5, 10], labels=["1", "2", "5", "10"])
        axis.set_ylabel(label)
        axis.set_xlabel("nonlinear IK iterations")
        axis.grid(alpha=0.25)
    axes[0].set_title("Error-limit iteration sensitivity")
    axes[1].legend_.remove()
    axes[2].legend_.remove()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def symmetry_pair_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare mirrored left/right trajectories and exact bimanual pairs."""
    metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_ddq_p99_rad_s2",
        "actual_j1_ddq_p99_rad_s2",
        "actual_j4_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "command_velocity_saturation_fraction",
        "braking_binding_fraction",
        "nullspace_error_abs_p95_rad",
        "actual_min_rho",
        "q_gap_rms_rad",
    ]

    def canonical_name(name: str) -> str:
        canonical = name.replace("_right_", "_SIDE_").replace(
            "_left_",
            "_SIDE_",
        )
        canonical = canonical.replace("normal_right_", "normal_SIDE_")
        canonical = canonical.replace("normal_left_", "normal_SIDE_")
        if canonical.startswith("reach_SIDE_m0p10"):
            canonical = canonical.replace("m0p10", "lat0p10")
        if canonical.startswith("reach_SIDE_p0p10"):
            canonical = canonical.replace("p0p10", "lat0p10")
        return canonical

    frame = summary.copy()
    frame["canonical_scenario"] = frame["scenario"].map(canonical_name)
    rows: list[dict[str, object]] = []
    for (profile, scenario), group in frame.groupby(
        ["profile", "canonical_scenario"],
    ):
        right = group[group["side"] == "right"]
        left = group[group["side"] == "left"]
        if right.empty or left.empty:
            continue
        row: dict[str, object] = {
            "profile": profile,
            "scenario": scenario,
        }
        for metric in metrics:
            right_value = float(right.iloc[0][metric])
            left_value = float(left.iloc[0][metric])
            row[f"{metric}_right"] = right_value
            row[f"{metric}_left"] = left_value
            row[f"{metric}_left_over_right"] = left_value / max(
                abs(right_value),
                1e-12,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_reach_trace_comparison(root: Path, output: Path) -> None:
    trace_dir = root / "screening" / "traces"
    scenario = "reach_right_p0p00_v0p80"
    profiles = ["current", "no_singularity_limit", "velocity_only"]
    traces: dict[str, np.lib.npyio.NpzFile] = {}
    for profile in profiles:
        path = _find_trace(trace_dir, profile, scenario)
        if path is not None:
            traces[profile] = np.load(path)
    if len(traces) < 2:
        return

    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for profile, trace in traces.items():
        time_values = trace["times"]
        axes[0].plot(time_values, trace["right_actual_q"][:, 0], label=profile)
        axes[1].plot(time_values, trace["right_actual_q"][:, 3], label=profile)
        axes[2].plot(time_values, trace["right_actual_rho"], label=profile)
        error = np.linalg.norm(
            trace["right_target_pose"][:, :3]
            - trace["right_actual_pose"][:, :3],
            axis=1,
        )
        axes[3].plot(time_values, 1e3 * error, label=profile)
    axes[0].set_ylabel("actual J1 (rad)")
    axes[1].set_ylabel("actual J4 (rad)")
    axes[2].set_ylabel("rho")
    axes[3].set_ylabel("EE error (mm)")
    axes[3].set_xlabel("time (s)")
    axes[0].legend(ncol=3)
    axes[0].set_title("Straight reach toward and beyond the workspace boundary")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    for trace in traces.values():
        trace.close()


def save_retract_trace_comparison(root: Path, output: Path) -> None:
    trace_dir = root / "screening" / "traces"
    scenario = "retract_diag_p0p10_v0p80"
    profiles = ["current", "no_nullspace", "no_error_limit", "velocity_only"]
    traces: dict[str, np.lib.npyio.NpzFile] = {}
    for profile in profiles:
        path = _find_trace(trace_dir, profile, scenario)
        if path is not None:
            traces[profile] = np.load(path)
    if len(traces) < 2:
        return

    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for profile, trace in traces.items():
        time_values = trace["times"]
        axes[0].plot(time_values, trace["right_actual_q"][:, 0], label=profile)
        axes[1].plot(time_values, trace["right_actual_q"][:, 3], label=profile)
        axes[2].plot(
            time_values,
            trace["right_actual_elbow"][:, 1],
            label=profile,
        )
        axes[3].plot(
            time_values,
            trace["right_frame_limit_activation"],
            label=profile,
        )
    axes[0].set_ylabel("actual J1 (rad)")
    axes[1].set_ylabel("actual J4 (rad)")
    axes[2].set_ylabel("elbow lateral (m)")
    axes[3].set_ylabel("error-limit alpha")
    axes[3].set_xlabel("time (s)")
    axes[0].legend(ncol=2)
    axes[0].set_title("Fast diagonal retract and branch selection")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    for trace in traces.values():
        trace.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dev/results/ik_safety_study_20260728"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    suites = [
        "screening",
        "parameters",
        "simplified",
        "targeted",
        "refined",
        "branch",
        "candidates",
        "finalists",
        "braking_knee",
        "iterations",
        "symmetry",
        "dynamics",
        "history_024e029",
        "history_cf259dd",
        "history_controlled_024",
        "history_controlled_cf",
    ]
    loaded = {
        suite: frame
        for suite in suites
        if (frame := load_summary(root, suite)) is not None
    }
    for suite, frame in loaded.items():
        profile_means(frame).to_csv(
            output / f"{suite}_profile_means.csv",
            index=False,
        )
        if "current" in set(frame["profile"]):
            paired_percent_changes(frame).to_csv(
                output / f"{suite}_paired_overall.csv",
                index=False,
            )
            paired_percent_changes(frame, by_family=True).to_csv(
                output / f"{suite}_paired_by_family.csv",
                index=False,
            )

    screening = loaded.get("screening")
    if screening is not None:
        overall = paired_percent_changes(screening)
        save_screening_heatmap(overall, output / "feature_ablation_heatmap.png")
        save_joint_activity_heatmaps(
            screening,
            output / "joint_saturation_and_braking.png",
        )
        save_reach_trace_comparison(root, output / "reach_trace_comparison.png")
        save_retract_trace_comparison(root, output / "retract_trace_comparison.png")

    parameters = loaded.get("parameters")
    if parameters is not None:
        save_parameter_tradeoff(
            parameters,
            output / "parameter_tradeoff.png",
        )

    if loaded.get("branch") is not None:
        reference = branch_reference_distances(root)
        consistency = branch_speed_consistency(root)
        reference.to_csv(output / "branch_reference_distances.csv", index=False)
        consistency.to_csv(output / "branch_speed_consistency.csv", index=False)
        if not reference.empty and not consistency.empty:
            save_branch_comparison(
                root,
                reference,
                consistency,
                output / "branch_consistency.png",
            )
    if loaded.get("candidates") is not None:
        candidate_reference = branch_reference_distances(
            root,
            suite="candidates",
            reference_profile="no_velocity_current_tasks",
        )
        candidate_consistency = branch_speed_consistency(
            root,
            suite="candidates",
        )
        candidate_reference.to_csv(
            output / "candidate_reference_distances.csv",
            index=False,
        )
        candidate_consistency.to_csv(
            output / "candidate_speed_consistency.csv",
            index=False,
        )
    if loaded.get("braking_knee") is not None:
        braking_reference = branch_reference_distances(
            root,
            suite="braking_knee",
            reference_profile="unlimited_reference",
            reference_suite="branch",
        )
        braking_consistency = branch_speed_consistency(
            root,
            suite="braking_knee",
        )
        braking_reference.to_csv(
            output / "braking_knee_reference_distances.csv",
            index=False,
        )
        braking_consistency.to_csv(
            output / "braking_knee_speed_consistency.csv",
            index=False,
        )
        if not candidate_reference.empty:
            save_candidate_tradeoff(
                loaded["candidates"],
                candidate_reference,
                output / "candidate_tradeoff.png",
            )
    if loaded.get("finalists") is not None:
        finalist_reference = branch_reference_distances(
            root,
            suite="finalists",
            reference_profile="no_velocity_current_tasks",
        )
        finalist_consistency = branch_speed_consistency(
            root,
            suite="finalists",
        )
        finalist_reference.to_csv(
            output / "finalist_reference_distances.csv",
            index=False,
        )
        finalist_consistency.to_csv(
            output / "finalist_speed_consistency.csv",
            index=False,
        )
        if not finalist_reference.empty:
            save_candidate_tradeoff(
                loaded["finalists"],
                finalist_reference,
                output / "finalist_tradeoff.png",
            )
    if loaded.get("iterations") is not None:
        save_iteration_sensitivity(
            loaded["iterations"],
            output / "iteration_sensitivity.png",
        )
    if loaded.get("symmetry") is not None:
        symmetry_pair_metrics(loaded["symmetry"]).to_csv(
            output / "symmetry_pairs.csv",
            index=False,
        )
    print(f"Wrote analysis tables and plots to {output}")


if __name__ == "__main__":
    main()
