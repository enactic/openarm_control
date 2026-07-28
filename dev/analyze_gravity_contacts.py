#!/usr/bin/env python3
"""Compare gravity compensation with contact-free joint-limit dynamics."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import ClassVar

import ik_safety_study as study
import mujoco
import numpy as np
import pandas as pd
import run_supplemental_study as supplemental


def _contact_plant(*, zero_gravity: bool) -> type[study.DynamicPlant]:
    class ContactPlant(study.DynamicPlant):
        contact_counts: ClassVar[Counter[tuple[str, str]]] = Counter()

        def __init__(self, *args: object, **kwargs: object) -> None:
            if zero_gravity:
                kwargs["gravity_compensation"] = False
            super().__init__(*args, **kwargs)
            if zero_gravity:
                self.model.opt.gravity[:] = 0.0
                mujoco.mj_forward(self.model, self.data)

        def _physics_step(self) -> None:
            super()._physics_step()
            for contact in self.data.contact[: self.data.ncon]:
                first = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom1,
                )
                second = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom2,
                )
                type(self).contact_counts[(first or "?", second or "?")] += 1

    return ContactPlant


def _run(
    profile: study.Profile,
    scenario: study.Scenario,
    *,
    mode: str,
    joint_upper: float,
) -> dict[str, object]:
    original = study.DynamicPlant
    plant_type = _contact_plant(zero_gravity=mode == "zero_gravity")
    study.DynamicPlant = plant_type
    try:
        trace = study.simulate(profile, scenario)
    finally:
        study.DynamicPlant = original

    side = trace.sides["right"]
    approach = np.isin(trace.phase, (1, 2))
    command = side.command_q[approach, 0]
    actual = side.actual_q[approach, 0]
    actual_dq = side.actual_dq[approach, 0]
    contacts = {
        f"{first}|{second}": count
        for (first, second), count in plant_type.contact_counts.items()
    }
    roof_contacts = sum(
        count
        for pair, count in contacts.items()
        if "cell_roof_col" in pair
    )
    return {
        "scenario": scenario.name,
        "mode": mode,
        "solver_failures": int(np.count_nonzero(trace.solver_failed)),
        "command_j1_upper_margin_rad": float(joint_upper - np.max(command)),
        "actual_j1_upper_margin_rad": float(joint_upper - np.max(actual)),
        "actual_j1_approach_dq_max_rad_s": float(np.max(actual_dq)),
        "j1_q_gap_rms_rad": float(
            np.sqrt(np.mean(np.square(command - actual)))
        ),
        "roof_contact_count": roof_contacts,
        "contact_pairs": ";".join(
            f"{pair}:{count}" for pair, count in sorted(contacts.items())
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    factory = study.PoseFactory()
    _, joint_upper = supplemental._joint_range(factory, "right", 0)
    scenarios = [
        scenario
        for scenario in supplemental.braking_dynamics_scenarios()
        if scenario.name in {"joint1_upper_v1", "joint1_upper_v2"}
    ]
    profiles = {
        profile.name: profile
        for profile in supplemental.braking_dynamics_profiles()
        if profile.name in {"current", "gravity_compensation"}
    }
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        rows.append(
            _run(
                profiles["current"],
                scenario,
                mode="current",
                joint_upper=joint_upper,
            )
        )
        rows.append(
            _run(
                profiles["gravity_compensation"],
                scenario,
                mode="gravity_compensation",
                joint_upper=joint_upper,
            )
        )
        rows.append(
            _run(
                profiles["current"],
                scenario,
                mode="zero_gravity",
                joint_upper=joint_upper,
            )
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
