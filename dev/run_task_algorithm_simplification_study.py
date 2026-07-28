#!/usr/bin/env python3
"""Test algorithm-level task simplifications without changing production code."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import analyze_targeted_braking as targeted
import ik_safety_study as study
import numpy as np
import pandas as pd
import run_control_simplification_study as simplification
import run_supplemental_study as supplemental


OUTPUT_DEFAULT = Path(
    "dev/results/task_algorithm_simplification_study_20260729"
)


def nullspace_profiles() -> list[study.Profile]:
    """Compare rate-form and direct-angle nullspace objectives."""
    direct_rate = 1.0 / (study.CONTROL_DT / 5)
    profiles = [
        study.current_profile("current"),
        study.current_profile(
            "null_rate_only",
            nullspace_max_speed=1e6,
        ),
        study.current_profile(
            "null_rate_only_no_activation",
            nullspace_max_speed=1e6,
            nullspace_singularity_low=0.0,
            nullspace_singularity_high=1e-6,
        ),
    ]
    for cost in (0.03, 0.1, 0.3, 1.0, 3.0, 12.0):
        profiles.append(
            study.current_profile(
                f"null_direct_cost_{str(cost).replace('.', 'p')}",
                nullspace_cost=cost,
                nullspace_return_rate=direct_rate,
                nullspace_max_speed=1e6,
            )
        )
    return profiles


def error_profiles() -> list[study.Profile]:
    """Compare speed scheduling with unconditional direct clipping."""
    profiles = [
        study.current_profile("current"),
        study.current_profile(
            "error_instant_latched",
            **simplification._instant_error_overrides(),
        ),
    ]
    for cap in (0.003, 0.006, 0.009, 0.012):
        profiles.append(
            study.current_profile(
                f"error_fixed_{int(round(cap * 1000))}mm",
                frame_position_error_limit=cap,
                error_schedule_override="always",
            )
        )
    return profiles


def _distance_only(
    name: str,
    distance: float,
    **metadata: object,
) -> study.Profile:
    return study.current_profile(
        name,
        joint_limit_braking_slowdown_distance=distance,
        joint_limit_braking_reaction_time=0.0,
        joint_limit_braking_distance_buffer=0.0,
        **metadata,
    )


def braking_profiles() -> list[study.Profile]:
    """Retune one distance after removing measured-velocity prediction."""
    return [
        study.current_profile("current"),
        *[
            _distance_only(
                f"brake_distance_only_{str(distance).replace('.', 'p')}",
                distance,
            )
            for distance in (0.5, 0.6, 0.7, 0.8, 0.9)
        ],
        study.current_profile(
            "brake_reaction_only_0p5",
            joint_limit_braking_distance_buffer=0.0,
        ),
        study.current_profile(
            "brake_buffer_only_0p5",
            joint_limit_braking_reaction_time=0.0,
        ),
    ]


def braking_robustness_profiles() -> list[study.Profile]:
    """Cross promising distance-only variants with representative lag."""
    environments = (
        ("nominal", {}),
        (
            "lagged",
            {
                "command_delay_s": 0.024,
                "state_delay_s": 0.024,
                "actuator_kp_scale": 0.7,
                "actuator_kv_scale": 0.7,
            },
        ),
    )
    profiles: list[study.Profile] = []
    for suffix, metadata in environments:
        profiles.append(study.current_profile(f"current__{suffix}", **metadata))
        for distance in (0.6, 0.7, 0.8, 0.9):
            profiles.append(
                _distance_only(
                    (
                        f"brake_distance_only_"
                        f"{str(distance).replace('.', 'p')}__{suffix}"
                    ),
                    distance,
                    **metadata,
                )
            )
    return profiles


def braking_boundary_robustness_profiles() -> list[study.Profile]:
    """Stress selected distance-only candidates under delayed weak actuation."""
    lagged = {
        "command_delay_s": 0.024,
        "state_delay_s": 0.024,
        "actuator_kp_scale": 0.7,
        "actuator_kv_scale": 0.7,
    }
    return [
        study.current_profile("current__nominal"),
        _distance_only("brake_distance_only_0p6__nominal", 0.6),
        _distance_only("brake_distance_only_0p7__nominal", 0.7),
        study.current_profile("current__lagged", **lagged),
        _distance_only("brake_distance_only_0p6__lagged", 0.6, **lagged),
        _distance_only("brake_distance_only_0p7__lagged", 0.7, **lagged),
    ]


def representative_scenarios() -> list[study.Scenario]:
    """Use the established VR-like trajectory set."""
    return simplification.component_scenarios()


def braking_control_scenarios() -> list[study.Scenario]:
    """Focus plant robustness on trajectories that stress shoulder and elbow."""
    factory = study.PoseFactory()
    return [
        study.make_reach_scenario(factory, speed=0.4),
        study.make_reach_scenario(factory, speed=0.8),
        study.make_reach_scenario(factory, speed=0.8, lateral=-0.10),
        study.make_reach_scenario(factory, speed=0.8, lateral=0.10),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.4,
            lateral=-0.10,
        ),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=-0.10,
        ),
        study.make_extended_circle_scenario(factory, speed=0.8),
        study.make_normal_lissajous_scenario(factory, speed=0.8, mode="right"),
    ]


def run_boundary(
    profiles: list[study.Profile],
    output: Path,
    *,
    workers: int,
) -> None:
    """Measure target-joint margins for each braking simplification."""
    factory = study.PoseFactory()
    joint_ranges = tuple(
        supplemental._joint_range(factory, "right", joint_index)
        for joint_index in range(7)
    )
    jobs = [
        (profile, scenario, joint_ranges)
        for profile in profiles
        for scenario in supplemental.braking_dynamics_scenarios()
    ]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(targeted._analyze_job, *job)
            for job in jobs
        ]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            rows.append(future.result())
            if index % 20 == 0 or index == len(futures):
                print(f"[boundary {index}/{len(futures)}]", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["joint", "boundary", "speed", "profile"]
    ).to_csv(output / "targeted_braking.csv", index=False)


def write_nullspace_equivalence(output: Path) -> None:
    """Record the algebraic equivalence of rate-only and scaled direct error."""
    dt_sub = study.CONTROL_DT / 5
    return_rate = 1.6
    rng = np.random.default_rng(20260729)
    max_h_error = 0.0
    max_c_error = 0.0
    samples = []
    for _ in range(10_000):
        direction = rng.normal(size=7)
        direction /= np.linalg.norm(direction)
        posture_error = float(rng.normal(scale=0.5))
        activation = float(rng.uniform())
        cost = np.sqrt(activation) * 12.0

        displacement = -return_rate * posture_error * dt_sub
        task_error = -displacement
        weighted_error_rate = cost * (-task_error)
        weighted_jacobian = cost * direction
        h_rate = np.outer(weighted_jacobian, weighted_jacobian)
        c_rate = -weighted_error_rate * weighted_jacobian

        gain = return_rate * dt_sub
        weighted_error_direct = cost * (-gain * posture_error)
        h_direct = np.outer(weighted_jacobian, weighted_jacobian)
        c_direct = -weighted_error_direct * weighted_jacobian

        max_h_error = max(max_h_error, float(np.max(np.abs(h_rate - h_direct))))
        max_c_error = max(max_c_error, float(np.max(np.abs(c_rate - c_direct))))
        samples.append(abs(displacement))

    payload = {
        "samples": 10_000,
        "dt_sub_s": dt_sub,
        "return_rate_s_inv": return_rate,
        "equivalent_direct_gain": return_rate * dt_sub,
        "max_abs_H_difference": max_h_error,
        "max_abs_c_difference": max_c_error,
        "max_reference_displacement_rad": float(np.max(samples)),
        "interpretation": (
            "Without speed clipping, direct angle error with gain=k*dt_sub "
            "is exactly the same QP objective as the current rate target."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "nullspace_objective_equivalence.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "nullspace",
            "error",
            "braking",
            "braking-robustness",
            "boundary",
            "boundary-robustness",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    write_nullspace_equivalence(output / "analysis")
    scenarios = representative_scenarios()
    if args.suite in {"nullspace", "all"}:
        study.run_matrix(
            nullspace_profiles(),
            scenarios,
            output / "nullspace",
            workers=args.workers,
        )
    if args.suite in {"error", "all"}:
        study.run_matrix(
            error_profiles(),
            scenarios,
            output / "error",
            workers=args.workers,
        )
    if args.suite in {"braking", "all"}:
        study.run_matrix(
            braking_profiles(),
            braking_control_scenarios(),
            output / "braking",
            workers=args.workers,
        )
    if args.suite in {"braking-robustness", "all"}:
        study.run_matrix(
            braking_robustness_profiles(),
            braking_control_scenarios(),
            output / "braking_robustness",
            workers=args.workers,
        )
    if args.suite in {"boundary", "all"}:
        run_boundary(
            braking_profiles(),
            output / "boundary",
            workers=args.workers,
        )
    if args.suite in {"boundary-robustness", "all"}:
        run_boundary(
            braking_boundary_robustness_profiles(),
            output / "boundary_robustness",
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
