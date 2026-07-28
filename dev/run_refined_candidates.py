#!/usr/bin/env python3
"""Run final compact-parameter candidates after the broad safety sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from ik_safety_study import current_profile, run_matrix, screening_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    keep_primary_braking = (0.50, 0.50, 0.15, 0.50, 0.15, 0.15, 0.15)
    keep_elbow_plane_braking = (0.15, 0.50, 0.15, 0.50, 0.15, 0.15, 0.15)
    instant_schedule = {
        "frame_error_limit_linear_slow": 0.6,
        "frame_error_limit_linear_fast": 0.9,
        "frame_error_limit_activation_rise_rate": 1e6,
        "frame_error_limit_activation_fall_rate": 1e6,
        "joint_limit_braking_distance_buffer": 0.0,
    }
    profiles = [
        current_profile("current"),
        current_profile("simple_instant_no_buffer", **instant_schedule),
        current_profile(
            "simple_instant_energy_1e4",
            **instant_schedule,
            kinetic_energy_cost=1e-4,
        ),
        current_profile(
            "brake_keep_j124",
            braking_distances=keep_primary_braking,
        ),
        current_profile(
            "brake_keep_j24",
            braking_distances=keep_elbow_plane_braking,
        ),
        current_profile(
            "null_stronger",
            nullspace_cost=16.0,
            nullspace_return_rate=2.0,
        ),
    ]
    run_matrix(
        profiles,
        screening_scenarios(),
        args.output_dir.resolve(),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
