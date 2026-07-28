#!/usr/bin/env python3
"""Run controlled historical IK comparisons with the common study harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from ik_safety_study import (
    CURRENT_CAPS,
    Profile,
    current_profile,
    run_matrix,
    screening_scenarios,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("old_caps_current", "no_error_limit"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.variant == "old_caps_current":
        profile = Profile(
            name="history_024_caps_current",
            velocity_caps=CURRENT_CAPS,
            description=(
                "Historical controller with current velocity caps, isolating "
                "algorithm changes from the old 1.57 rad/s J1/J2 defaults."
            ),
        )
    else:
        profile = current_profile(
            "history_cf_no_error_limit",
            frame_position_error_limit=0.0,
            description=(
                "cf259dd controller without bounded FrameTask position error."
            ),
        )
    run_matrix(
        [profile],
        screening_scenarios(),
        args.output_dir.resolve(),
        workers=1,
    )


if __name__ == "__main__":
    main()
