#!/usr/bin/env python3
"""Measure command/actual lag and motion quality in recorded interventions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ik_safety_study as study
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

DEFAULT_RECORD_ROOT = Path(
    "/hdd_data/rollout/"
    "pillow_0702_history_with_visual_from_flatten_hist_base_30k_0711_window_20_80k"
    "/dataset/episodes"
)
DEFAULT_EPISODES = (187, 188, 189, 194, 196, 198)
DT = study.CONTROL_DT


def _load_stream(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    columns = ["timestamp", "qpos"]
    schema = pq.read_schema(path)
    if "qvel" in schema.names:
        columns.append("qvel")
    table = pq.read_table(path, columns=columns)
    timestamp = (
        table["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        * 1e-9
    )
    qpos = np.asarray(table["qpos"].to_pylist(), dtype=np.float64)[:, :7]
    qvel = (
        np.asarray(table["qvel"].to_pylist(), dtype=np.float64)[:, :7]
        if "qvel" in columns
        else None
    )
    keep = np.concatenate([[True], np.diff(timestamp) > 1e-6])
    return timestamp[keep], qpos[keep], None if qvel is None else qvel[keep]


def _interpolate(
    timestamp: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_time, timestamp, values[:, index])
            for index in range(values.shape[1])
        ]
    )


def _smooth_derivative(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = values.shape[0]
    window = min(21, count - (1 - count % 2))
    if window < 5:
        velocity = np.zeros_like(values)
        velocity[1:] = np.diff(values, axis=0) / DT
        acceleration = np.zeros_like(values)
        acceleration[1:] = np.diff(velocity, axis=0) / DT
        return velocity, acceleration
    velocity = savgol_filter(
        values,
        window_length=window,
        polyorder=3,
        deriv=1,
        delta=DT,
        axis=0,
        mode="interp",
    )
    acceleration = savgol_filter(
        values,
        window_length=window,
        polyorder=3,
        deriv=2,
        delta=DT,
        axis=0,
        mode="interp",
    )
    return velocity, acceleration


def _best_lag_ms(
    command_q: np.ndarray,
    actual_q: np.ndarray,
    command_dq: np.ndarray,
    joint_index: int,
) -> tuple[float, float]:
    movement = np.abs(command_dq[:, joint_index]) >= 0.08
    best_lag = 0
    best_rmse = float("inf")
    for lag in range(round(0.24 / DT) + 1):
        if lag == 0:
            command = command_q[:, joint_index]
            actual = actual_q[:, joint_index]
            mask = movement
        else:
            command = command_q[:-lag, joint_index]
            actual = actual_q[lag:, joint_index]
            mask = movement[:-lag]
        if np.count_nonzero(mask) < 100:
            continue
        rmse = float(np.sqrt(np.mean(np.square(command[mask] - actual[mask]))))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = lag
    return 1e3 * best_lag * DT, best_rmse


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_arm(
    record_root: Path,
    episode: int,
    side: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    action_time, action_q, _ = _load_stream(
        record_root / str(episode) / "action" / "arms" / side / "state.parquet"
    )
    obs_time, obs_q, obs_dq = _load_stream(
        record_root / str(episode) / "obs" / "arms" / side / "state.parquet"
    )
    start = max(action_time[0], obs_time[0])
    stop = min(action_time[-1], obs_time[-1])
    uniform_time = np.arange(start, stop, DT)
    command_q = _interpolate(action_time, action_q, uniform_time)
    actual_q = _interpolate(obs_time, obs_q, uniform_time)
    command_dq, command_ddq = _smooth_derivative(command_q)
    if obs_dq is None:
        actual_dq, actual_ddq = _smooth_derivative(actual_q)
    else:
        actual_dq_raw = _interpolate(obs_time, obs_dq, uniform_time)
        window = min(21, actual_dq_raw.shape[0] - (1 - actual_dq_raw.shape[0] % 2))
        actual_dq = (
            savgol_filter(
                actual_dq_raw,
                window_length=window,
                polyorder=3,
                axis=0,
                mode="interp",
            )
            if window >= 5
            else actual_dq_raw
        )
        actual_ddq = np.zeros_like(actual_dq)
        actual_ddq[1:] = np.diff(actual_dq, axis=0) / DT

    q_gap = command_q - actual_q
    caps = np.asarray(study.CURRENT_CAPS)
    joint_rows: list[dict[str, object]] = []
    frozen_joints: list[int] = []
    lag_values: list[float] = []
    for joint_index in range(7):
        lag_ms, aligned_rmse = _best_lag_ms(
            command_q,
            actual_q,
            command_dq,
            joint_index,
        )
        command_range = float(np.ptp(command_q[:, joint_index]))
        actual_range = float(np.ptp(actual_q[:, joint_index]))
        frozen = command_range > 0.05 and actual_range < 0.01
        if frozen:
            frozen_joints.append(joint_index + 1)
        if command_range > 0.05 and not frozen:
            lag_values.append(lag_ms)
        joint_rows.append(
            {
                "episode": episode,
                "side": side,
                "joint": joint_index + 1,
                "duration_s": float(stop - start),
                "command_range_rad": command_range,
                "actual_range_rad": actual_range,
                "frozen": frozen,
                "best_lag_ms": lag_ms,
                "lag_aligned_rmse_rad": aligned_rmse,
                "q_gap_rms_rad": float(
                    np.sqrt(np.mean(np.square(q_gap[:, joint_index])))
                ),
                "q_gap_p99_rad": float(
                    np.percentile(np.abs(q_gap[:, joint_index]), 99)
                ),
                "command_dq_p99_rad_s": float(
                    np.percentile(np.abs(command_dq[:, joint_index]), 99)
                ),
                "actual_dq_p99_rad_s": float(
                    np.percentile(np.abs(actual_dq[:, joint_index]), 99)
                ),
                "command_ddq_p99_rad_s2": float(
                    np.percentile(np.abs(command_ddq[:, joint_index]), 99)
                ),
                "actual_ddq_p99_rad_s2": float(
                    np.percentile(np.abs(actual_ddq[:, joint_index]), 99)
                ),
                "command_speed_over_cap_fraction": float(
                    np.mean(np.abs(command_dq[:, joint_index]) > caps[joint_index])
                ),
                "actual_speed_over_cap_fraction": float(
                    np.mean(np.abs(actual_dq[:, joint_index]) > caps[joint_index])
                ),
            }
        )

    arm_row: dict[str, object] = {
        "episode": episode,
        "side": side,
        "duration_s": float(stop - start),
        "sample_count": int(uniform_time.size),
        "q_gap_rms_rad": float(np.sqrt(np.mean(np.square(q_gap)))),
        "q_gap_p99_rad": float(np.percentile(np.abs(q_gap), 99)),
        "command_dq_p99_rad_s": float(np.percentile(np.abs(command_dq), 99)),
        "actual_dq_p99_rad_s": float(np.percentile(np.abs(actual_dq), 99)),
        "command_ddq_p99_rad_s2": float(
            np.percentile(np.abs(command_ddq), 99)
        ),
        "actual_ddq_p99_rad_s2": float(np.percentile(np.abs(actual_ddq), 99)),
        "median_active_joint_lag_ms": (
            float(np.median(lag_values)) if lag_values else float("nan")
        ),
        "frozen_joints": ",".join(str(value) for value in frozen_joints),
        "has_frozen_joint": bool(frozen_joints),
    }
    return arm_row, joint_rows


def _markdown_summary(arm_rows: list[dict[str, object]]) -> str:
    valid = [row for row in arm_rows if not bool(row["has_frozen_joint"])]
    frozen = [row for row in arm_rows if bool(row["has_frozen_joint"])]
    lines = [
        "# Recorded intervention command/actual audit",
        "",
        (
            "The action stream is a position command, not the original VR target. "
            "All lag values minimize command-to-observation joint RMSE on moving "
            "samples."
        ),
        "",
        "## Valid arms",
        "",
        "| episode | side | lag ms | q gap rms rad | command dq p99 | actual dq p99 | actual ddq p99 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in valid:
        lines.append(
            f"| {row['episode']} | {row['side']} | "
            f"{float(row['median_active_joint_lag_ms']):.1f} | "
            f"{float(row['q_gap_rms_rad']):.4f} | "
            f"{float(row['command_dq_p99_rad_s']):.2f} | "
            f"{float(row['actual_dq_p99_rad_s']):.2f} | "
            f"{float(row['actual_ddq_p99_rad_s2']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen/reconnect arms",
            "",
            "| episode | side | frozen joints | q gap rms rad |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in frozen:
        lines.append(
            f"| {row['episode']} | {row['side']} | {row['frozen_joints']} | "
            f"{float(row['q_gap_rms_rad']):.4f} |"
        )
    if valid:
        lags = np.asarray(
            [float(row["median_active_joint_lag_ms"]) for row in valid]
        )
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- Median valid arm lag: `{np.nanmedian(lags):.1f} ms`.",
                f"- Valid lag p90: `{np.nanpercentile(lags, 90):.1f} ms`.",
                "- Episodes with frozen joints are excluded from controller tuning.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPISODES),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    for episode in args.episodes:
        for side in study.SIDES:
            arm_row, rows = analyze_arm(
                args.record_root.resolve(),
                episode,
                side,
            )
            arm_rows.append(arm_row)
            joint_rows.extend(rows)
            print(
                episode,
                side,
                "lag_ms=",
                arm_row["median_active_joint_lag_ms"],
                "frozen=",
                arm_row["frozen_joints"],
                flush=True,
            )
    _write_csv(output_dir / "arm_summary.csv", arm_rows)
    _write_csv(output_dir / "joint_summary.csv", joint_rows)
    (output_dir / "summary.md").write_text(
        _markdown_summary(arm_rows),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "record_root": str(args.record_root.resolve()),
                "episodes": args.episodes,
                "sample_dt": DT,
                "lag_search_max_s": 0.24,
                "savgol_window": 21,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
