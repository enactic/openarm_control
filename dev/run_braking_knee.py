#!/usr/bin/env python3
"""Sweep the J2 braking-distance knee with an optional J1 cap increase."""

from __future__ import annotations

import argparse
from pathlib import Path

from ik_safety_study import (
    CURRENT_CAPS,
    branch_scenarios,
    current_profile,
    run_matrix,
)


def braking_distances(j2_distance: float) -> tuple[float, ...]:
    return (0.50, j2_distance, 0.50, 0.50, 0.50, 0.50, 0.50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profiles = [current_profile("current")]
    for distance in (0.25, 0.35, 0.40):
        tag = str(distance).replace(".", "p")
        profiles.append(
            current_profile(
                f"j2_brake_{tag}",
                braking_distances=braking_distances(distance),
            )
        )
        profiles.append(
            current_profile(
                f"j1_2p5_j2_brake_{tag}",
                velocity_caps=(2.5, *CURRENT_CAPS[1:]),
                braking_distances=braking_distances(distance),
            )
        )
    profiles.append(
        current_profile(
            "j1_2p5_j2_brake_0p35_energy5e5",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
            braking_distances=braking_distances(0.35),
            kinetic_energy_cost=5e-5,
        )
    )
    run_matrix(
        profiles,
        branch_scenarios(),
        args.output_dir.resolve(),
        workers=args.workers,
        save_all_traces=True,
    )


if __name__ == "__main__":
    main()
