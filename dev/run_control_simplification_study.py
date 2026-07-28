#!/usr/bin/env python3
"""Run controller-parameter simplification studies without changing production code."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path

import analyze_targeted_braking as targeted
import ik_safety_study as study
import pandas as pd
import run_supplemental_study as supplemental


def _instant_error_overrides() -> dict[str, float]:
    return {
        "frame_error_limit_linear_slow": 0.6,
        "frame_error_limit_linear_fast": 0.9,
        "frame_error_limit_activation_rise_rate": 1e6,
        "frame_error_limit_activation_fall_rate": 1e6,
    }


def component_profiles() -> list[study.Profile]:
    """Return one-factor candidates that remove or tie public parameters."""
    instant = _instant_error_overrides()
    return [
        study.current_profile("current"),
        study.current_profile(
            "shared_rho_window",
            nullspace_singularity_high=0.08,
        ),
        study.current_profile("error_instant_latched", **instant),
        study.current_profile(
            "error_instant_unlatched",
            error_schedule_override="instant_unlatched",
            **instant,
        ),
        study.current_profile(
            "error_always",
            error_schedule_override="always",
        ),
        study.current_profile(
            "brake_command_only",
            use_measured_state=False,
        ),
        study.current_profile(
            "brake_measured_position_only",
            joint_limit_braking_reaction_time=0.0,
            joint_limit_braking_distance_buffer=0.0,
        ),
        study.current_profile(
            "brake_reaction_0p04_no_buffer",
            joint_limit_braking_distance_buffer=0.0,
        ),
        study.current_profile(
            "brake_reaction_0p08_no_buffer",
            joint_limit_braking_reaction_time=0.08,
            joint_limit_braking_distance_buffer=0.0,
        ),
        study.current_profile(
            "brake_reaction_0p08_buffer_0p01",
            joint_limit_braking_reaction_time=0.08,
        ),
        study.current_profile(
            "null_rate_only_1p6",
            nullspace_max_speed=100.0,
        ),
        study.current_profile(
            "null_rate_only_1p0",
            nullspace_return_rate=1.0,
            nullspace_max_speed=100.0,
        ),
        study.current_profile(
            "null_speed_only_1p0",
            nullspace_return_rate=1e6,
        ),
        study.current_profile("no_energy", kinetic_energy_cost=0.0),
        study.current_profile(
            "no_energy_damping_0p3",
            kinetic_energy_cost=0.0,
            damping=0.3,
        ),
        study.current_profile(
            "no_energy_damping_1p0",
            kinetic_energy_cost=0.0,
            damping=1.0,
        ),
        study.current_profile(
            "simple_shared",
            nullspace_singularity_high=0.08,
            **instant,
        ),
        study.current_profile(
            "simple_shared_unlatched",
            error_schedule_override="instant_unlatched",
            nullspace_singularity_high=0.08,
            **instant,
        ),
        study.current_profile(
            "simple_shared_command_only",
            use_measured_state=False,
            nullspace_singularity_high=0.08,
            **instant,
        ),
        study.current_profile(
            "simple_shared_reaction_0p08",
            nullspace_singularity_high=0.08,
            joint_limit_braking_reaction_time=0.08,
            **instant,
        ),
    ]


def component_scenarios() -> list[study.Scenario]:
    """Cover slow/fast reach, branch changes, wrists, and normal motion."""
    factory = study.PoseFactory()
    return [
        study.make_reach_scenario(factory, speed=speed)
        for speed in (0.1, 0.4, 0.8)
    ] + [
        study.make_reach_scenario(factory, speed=0.8, lateral=lateral)
        for lateral in (-0.10, 0.10)
    ] + [
        study.make_diagonal_retract_scenario(
            factory,
            speed=speed,
            lateral=-0.10,
        )
        for speed in (0.1, 0.4, 0.8)
    ] + [
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=0.10,
        ),
        study.make_extended_circle_scenario(factory, speed=0.4),
        study.make_extended_circle_scenario(factory, speed=0.8),
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=6.0,
            extended=True,
        ),
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=6.0,
            extended=False,
        ),
        study.make_normal_lissajous_scenario(factory, speed=0.5, mode="right"),
        study.make_normal_lissajous_scenario(factory, speed=0.8, mode="right"),
        study.make_normal_lissajous_scenario(
            factory,
            speed=0.6,
            mode="bimanual",
        ),
        supplemental.make_random_cartesian_scenario(
            factory,
            seed=101,
            speed=0.5,
            mode="right",
            extended=False,
        ),
        supplemental.make_random_cartesian_scenario(
            factory,
            seed=202,
            speed=0.8,
            mode="right",
            extended=False,
        ),
    ]


def retune_profiles() -> list[study.Profile]:
    """Retune the remaining knobs after removing redundant parameters."""
    profiles = [study.current_profile("current")]

    for slow, fast in (
        (0.5, 0.8),
        (0.5, 0.9),
        (0.6, 0.9),
        (0.6, 1.0),
        (0.7, 1.0),
    ):
        profiles.append(
            study.current_profile(
                f"error_instant_{slow:g}_{fast:g}".replace(".", "p"),
                frame_error_limit_linear_slow=slow,
                frame_error_limit_linear_fast=fast,
                frame_error_limit_activation_rise_rate=1e6,
                frame_error_limit_activation_fall_rate=1e6,
            )
        )

    profiles.append(
        study.current_profile(
            "error_single_threshold_0p75",
            frame_error_limit_linear_slow=0.75,
            frame_error_limit_linear_fast=0.750001,
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
        )
    )
    for cap in (0.002, 0.004, 0.006):
        profiles.append(
            study.current_profile(
                f"error_instant_cap_{cap:g}".replace(".", "p").replace("-", "m"),
                frame_position_error_limit=cap,
                frame_error_limit_linear_slow=0.6,
                frame_error_limit_linear_fast=0.9,
                frame_error_limit_activation_rise_rate=1e6,
                frame_error_limit_activation_fall_rate=1e6,
            )
        )

    profiles.append(
        study.current_profile(
            "null_rate_only",
            nullspace_max_speed=100.0,
        )
    )
    for high in (0.06, 0.07, 0.08):
        profiles.append(
            study.current_profile(
                f"null_high_{high:g}".replace(".", "p"),
                nullspace_singularity_high=high,
                nullspace_max_speed=100.0,
            )
        )
    for cost in (18.0, 24.0):
        profiles.append(
            study.current_profile(
                f"null_high_0p08_cost_{cost:g}".replace(".", "p"),
                nullspace_cost=cost,
                nullspace_singularity_high=0.08,
                nullspace_max_speed=100.0,
            )
        )

    profiles.extend(
        [
            study.current_profile(
                "simple_error_rate_only",
                nullspace_max_speed=100.0,
                **_instant_error_overrides(),
            ),
            study.current_profile(
                "simple_error_rate_only_energy_5e5",
                nullspace_max_speed=100.0,
                kinetic_energy_cost=5e-5,
                **_instant_error_overrides(),
            ),
        ]
    )
    return profiles


def robustness_profiles() -> list[study.Profile]:
    """Cross the compact candidates with representative command/plant lag."""
    environments = (
        ("nominal", {}),
        ("delay24ms", {"command_delay_s": 0.024}),
        (
            "actuator0p7",
            {"actuator_kp_scale": 0.7, "actuator_kv_scale": 0.7},
        ),
        (
            "delay24ms_actuator0p7",
            {
                "command_delay_s": 0.024,
                "actuator_kp_scale": 0.7,
                "actuator_kv_scale": 0.7,
            },
        ),
    )
    profiles: list[study.Profile] = []
    for environment, metadata in environments:
        profiles.extend(
            [
                study.current_profile(f"current__{environment}", **metadata),
                study.current_profile(
                    f"simple__{environment}",
                    nullspace_max_speed=100.0,
                    **_instant_error_overrides(),
                    **metadata,
                ),
                study.current_profile(
                    f"simple_energy5e5__{environment}",
                    nullspace_max_speed=100.0,
                    kinetic_energy_cost=5e-5,
                    **_instant_error_overrides(),
                    **metadata,
                ),
            ]
        )
    return profiles


def boundary_profiles() -> list[study.Profile]:
    """Return braking simplifications that need target-joint margin checks."""
    return [
        study.current_profile("current"),
        study.current_profile("no_braking", joint_limit_braking=False),
        study.current_profile(
            "brake_command_only",
            use_measured_state=False,
        ),
        study.current_profile(
            "brake_measured_position_only",
            joint_limit_braking_reaction_time=0.0,
            joint_limit_braking_distance_buffer=0.0,
        ),
        study.current_profile(
            "brake_reaction_0p04_no_buffer",
            joint_limit_braking_distance_buffer=0.0,
        ),
        study.current_profile(
            "brake_reaction_0p08_no_buffer",
            joint_limit_braking_reaction_time=0.08,
            joint_limit_braking_distance_buffer=0.0,
        ),
        study.current_profile(
            "brake_reaction_0p08_buffer_0p01",
            joint_limit_braking_reaction_time=0.08,
        ),
    ]


def run_boundary(output: Path, *, workers: int) -> None:
    """Write target-joint dynamic margins for braking simplifications."""
    factory = study.PoseFactory()
    joint_ranges = tuple(
        supplemental._joint_range(factory, "right", joint_index)
        for joint_index in range(7)
    )
    jobs = [
        (profile, scenario, joint_ranges)
        for profile in boundary_profiles()
        for scenario in supplemental.braking_dynamics_scenarios()
    ]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("components", "retune", "robustness", "boundary", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dev/results/control_simplification_study_20260729"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    if args.suite in {"components", "all"}:
        study.run_matrix(
            component_profiles(),
            component_scenarios(),
            output / "components",
            workers=args.workers,
        )
    if args.suite in {"retune", "all"}:
        study.run_matrix(
            retune_profiles(),
            component_scenarios(),
            output / "retune",
            workers=args.workers,
        )
    if args.suite in {"robustness", "all"}:
        study.run_matrix(
            robustness_profiles(),
            component_scenarios(),
            output / "robustness",
            workers=args.workers,
        )
    if args.suite in {"boundary", "all"}:
        run_boundary(output / "boundary", workers=args.workers)


if __name__ == "__main__":
    main()
