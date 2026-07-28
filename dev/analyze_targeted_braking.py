#!/usr/bin/env python3
"""Measure targeted-joint braking margins in dynamic boundary trajectories."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
from pathlib import Path

import ik_safety_study as study
import numpy as np
import pandas as pd
import run_supplemental_study as supplemental

SCENARIO_PATTERN = re.compile(
    r"joint(?P<joint>\d+)_(?P<boundary>lower|upper)_v"
)


def _target_margin(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    boundary: str,
) -> np.ndarray:
    if boundary == "lower":
        return values - lower
    return upper - values


def _analyze_job(
    profile: study.Profile,
    scenario: study.Scenario,
    joint_ranges: tuple[tuple[float, float], ...],
) -> dict[str, object]:
    match = SCENARIO_PATTERN.fullmatch(scenario.name.rsplit("_v", 1)[0] + "_v")
    if match is None:
        match = SCENARIO_PATTERN.match(scenario.name)
    if match is None:
        raise ValueError(f"Cannot parse joint scenario {scenario.name!r}.")
    joint_index = int(match.group("joint")) - 1
    boundary = match.group("boundary")
    lower, upper = joint_ranges[joint_index]

    trace = study.simulate(profile, scenario)
    side = trace.sides["right"]
    approach = np.isin(trace.phase, (1, 2))
    command_q = side.command_q[approach, joint_index]
    actual_q = side.actual_q[approach, joint_index]
    command_dq = side.command_dq[approach, joint_index]
    actual_dq = side.actual_dq[approach, joint_index]
    actual_ddq = side.actual_ddq[approach, joint_index]
    command_margin = _target_margin(
        command_q,
        lower=lower,
        upper=upper,
        boundary=boundary,
    )
    actual_margin = _target_margin(
        actual_q,
        lower=lower,
        upper=upper,
        boundary=boundary,
    )
    direction = -1.0 if boundary == "lower" else 1.0
    command_approach = direction * command_dq
    actual_approach = direction * actual_dq
    if boundary == "lower":
        dynamic_overshoot = float(np.min(command_q) - np.min(actual_q))
    else:
        dynamic_overshoot = float(np.max(actual_q) - np.max(command_q))

    braking_fraction = side.braking_fraction_by_joint[approach, joint_index]
    braking_utilization = side.braking_utilization_by_joint[
        approach, joint_index
    ]
    return {
        "profile": profile.name,
        "scenario": scenario.name,
        "joint": joint_index + 1,
        "boundary": boundary,
        "speed": scenario.speed,
        "solver_failures": int(np.count_nonzero(trace.solver_failed)),
        "command_target_min_margin_rad": float(np.min(command_margin)),
        "actual_target_min_margin_rad": float(np.min(actual_margin)),
        "dynamic_overshoot_beyond_command_rad": dynamic_overshoot,
        "command_target_approach_dq_max_rad_s": float(
            np.max(command_approach)
        ),
        "actual_target_approach_dq_max_rad_s": float(np.max(actual_approach)),
        "actual_target_ddq_p99_rad_s2": float(
            np.quantile(np.abs(actual_ddq), 0.99)
        ),
        "target_q_gap_rms_rad": float(
            np.sqrt(np.mean(np.square(command_q - actual_q)))
        ),
        "target_braking_active_fraction": float(
            np.mean(braking_fraction < 1.0 - 1e-9)
        ),
        "target_braking_binding_fraction": float(
            np.mean(braking_utilization >= 0.995)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    factory = study.PoseFactory()
    joint_ranges = tuple(
        supplemental._joint_range(factory, "right", joint_index)
        for joint_index in range(7)
    )
    jobs = [
        (profile, scenario, joint_ranges)
        for profile in supplemental.braking_dynamics_profiles()
        for scenario in supplemental.braking_dynamics_scenarios()
    ]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(_analyze_job, profile, scenario, joint_ranges)
            for profile, scenario, joint_ranges in jobs
        ]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            rows.append(future.result())
            if index % 20 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}]", flush=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["joint", "boundary", "speed", "profile"]
    ).to_csv(output, index=False)


if __name__ == "__main__":
    main()
