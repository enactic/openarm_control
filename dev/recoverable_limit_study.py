#!/usr/bin/env python3
"""Compare native and recoverable joint limits after encoder overshoot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ik_safety_study as study
import mink
import numpy as np

OVERSHOOTS = (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)


def profile(name: str, limit_style: str) -> study.Profile:
    return study.current_profile(
        name,
        limit_style=limit_style,
        posture_cost=0.0,
        nullspace_cost=0.0,
        joint_limit_braking=False,
        singularity_approach_limit=False,
        frame_position_error_limit=0.0,
        kinetic_energy_cost=0.0,
    )


def solve_once(
    *,
    limit_style: str,
    side: str,
    joint_index: int,
    boundary: str,
    overshoot: float,
) -> dict[str, object]:
    kinematics = study.make_kinematics(
        profile(limit_style, limit_style),
        side,
    )
    solver = kinematics._ik
    assert solver is not None
    qpos_index = int(solver._arm_qpos_by_side[side][joint_index])
    dof_index = int(solver._arm_dofs_by_side[side][joint_index])
    joint_ids = np.flatnonzero(solver._model.jnt_qposadr == qpos_index)
    joint_id = int(joint_ids[0])
    lower, upper = solver._model.jnt_range[joint_id]
    q = solver._config.q.copy()
    q[qpos_index] = (
        float(lower - overshoot)
        if boundary == "lower"
        else float(upper + overshoot)
    )
    solver._config.update(q=q)
    for task in solver._tasks.values():
        task.set_target_from_configuration(solver._config)

    tasks = list(solver._tasks.values())
    constraints = [solver._freeze_task] if solver._freeze_task else []
    try:
        velocity = mink.solve_ik(
            solver._config,
            tasks,
            solver._substep_dt,
            solver._solver_name,
            limits=solver._limits,
            constraints=constraints,
            safety_break=False,
            **solver._solver_params,
        )
        feasible = True
        recovery_velocity = float(velocity[dof_index])
    except mink.exceptions.NoSolutionFound:
        feasible = False
        recovery_velocity = float("nan")

    cap = float(study.CURRENT_CAPS[joint_index])
    native_feasible_threshold = cap * solver._substep_dt / 0.95
    expected_sign = 1.0 if boundary == "lower" else -1.0
    return {
        "limit_style": limit_style,
        "side": side,
        "joint": joint_index + 1,
        "boundary": boundary,
        "overshoot_rad": overshoot,
        "velocity_cap_rad_s": cap,
        "substep_dt_s": solver._substep_dt,
        "native_feasible_threshold_rad": native_feasible_threshold,
        "feasible": feasible,
        "recovery_velocity_rad_s": recovery_velocity,
        "recovery_direction_correct": bool(
            feasible and expected_sign * recovery_velocity > 0.0
        ),
    }


def main() -> None:
    output_dir = Path("dev/results/ik_safety_study_20260728/recoverable_limit")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        solve_once(
            limit_style=limit_style,
            side=side,
            joint_index=joint_index,
            boundary=boundary,
            overshoot=overshoot,
        )
        for limit_style in ("standard", "recoverable")
        for side in ("right", "left")
        for joint_index in range(7)
        for boundary in ("lower", "upper")
        for overshoot in OVERSHOOTS
    ]
    with (output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, object] = {}
    for limit_style in ("standard", "recoverable"):
        selected = [row for row in rows if row["limit_style"] == limit_style]
        aggregate[limit_style] = {
            "cases": len(selected),
            "feasible": sum(bool(row["feasible"]) for row in selected),
            "correct_recovery_direction": sum(
                bool(row["recovery_direction_correct"]) for row in selected
            ),
        }
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
