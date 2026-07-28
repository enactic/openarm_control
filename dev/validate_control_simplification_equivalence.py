#!/usr/bin/env python3
"""Validate source-level IK safety simplifications without editing production code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mink
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco

from openarm_control import (
    ArmSetup,
    ErrorLimitedFrameTask,
    ErrorLimitedRelativeFrameTask,
)
from openarm_control.joint_braking_limit import JointBrakingLimit
from openarm_control.kinematics import (
    _arm_qpos_indices,
    _arm_velocity_limit_mapping,
)
from openarm_control.kinetic_energy_task import (
    KineticEnergyRegularizationTask,
)
from openarm_control.recoverable_configuration_limit import (
    RecoverableConfigurationLimit,
)


def _setup() -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode="bimanual",
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
        origin_frame="arm_origin",
        origin_frame_type="site",
    )


def _active_qpos(setup: ArmSetup) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [_arm_qpos_indices(setup, side) for side in setup.sides],
        )
    )


def _random_configuration(
    setup: ArmSetup,
    home: np.ndarray,
    qpos_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    q = home.copy()
    for qpos_index in qpos_indices:
        joint_ids = np.flatnonzero(setup.model.jnt_qposadr == qpos_index)
        if joint_ids.size != 1:
            raise RuntimeError(f"Cannot resolve qpos index {qpos_index}.")
        joint_id = int(joint_ids[0])
        if setup.model.jnt_limited[joint_id]:
            lower, upper = setup.model.jnt_range[joint_id]
            margin = 0.05 * (upper - lower)
            q[qpos_index] = rng.uniform(lower + margin, upper - margin)
        else:
            q[qpos_index] += rng.uniform(-0.5, 0.5)
    return q


def _max_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    absolute = float(np.max(np.abs(left - right), initial=0.0))
    scale = max(
        float(np.max(np.abs(left), initial=0.0)),
        float(np.max(np.abs(right), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    return absolute, absolute / scale


def _update_maximum(
    result: dict[str, float],
    name: str,
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    absolute, relative = _max_difference(left, right)
    result[f"{name}_max_abs"] = max(result.get(f"{name}_max_abs", 0.0), absolute)
    result[f"{name}_max_rel"] = max(result.get(f"{name}_max_rel", 0.0), relative)


def _validate_kinetic_energy(
    setup: ArmSetup,
    configuration: mink.Configuration,
    home: np.ndarray,
    qpos_indices: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float | int | bool | str]:
    cost = 3e-5
    local = KineticEnergyRegularizationTask(cost=cost)
    native = mink.KineticEnergyRegularizationTask(cost=cost)
    result: dict[str, float | int | bool | str] = {
        "requested_samples": samples,
        "completed_samples": 0,
        "native_compatible": True,
    }

    for _ in range(samples):
        configuration.update(
            q=_random_configuration(setup, home, qpos_indices, rng)
        )
        dt = float(rng.uniform(1e-4, 0.02))
        local.set_dt(dt)
        native.set_dt(dt)
        local_objective = local.compute_qp_objective(configuration)
        try:
            native_objective = native.compute_qp_objective(configuration)
        except (AttributeError, TypeError) as error:
            result["native_compatible"] = False
            result["native_error"] = f"{type(error).__name__}: {error}"
            result["wrapper_required"] = True
            result["equivalent"] = False
            return result
        _update_maximum(result, "H", local_objective.H, native_objective.H)
        _update_maximum(result, "c", local_objective.c, native_objective.c)
        result["completed_samples"] = int(result["completed_samples"]) + 1

    result["wrapper_required"] = False
    result["equivalent"] = bool(
        result["H_max_abs"] <= 1e-10 and result["c_max_abs"] <= 1e-12
    )
    return result


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(output))
    if limit > 0.0 and norm > limit:
        output *= limit / norm
    return output


def _composed_bounded_objective(
    task: mink.Task,
    configuration: mink.Configuration,
    *,
    position_limit: float,
    orientation_limit: float,
    activation: float,
) -> mink.Objective:
    error = task.compute_error(configuration)
    limited = error.copy()
    limited[:3] = _limit_norm(error[:3], position_limit)
    limited[3:] = _limit_norm(error[3:], orientation_limit)
    error += activation * (limited - error)
    jacobian = task.compute_jacobian(configuration)
    return task._assemble_qp(error, jacobian, configuration._eye_nv)


def _perturbed_target(
    base: mink.SE3,
    rng: np.random.Generator,
) -> mink.SE3:
    tangent = np.concatenate(
        [
            rng.uniform(-0.25, 0.25, size=3),
            rng.uniform(-1.0, 1.0, size=3),
        ]
    )
    return base.rplus(tangent)


def _validate_bounded_tasks(
    setup: ArmSetup,
    configuration: mink.Configuration,
    home: np.ndarray,
    qpos_indices: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, dict[str, float | int | bool]]:
    frame_name = "right_ee_control_point"
    frame_type = "site"
    root_name = "arm_origin"
    root_type = "site"
    task_kwargs = {
        "position_cost": 10.0,
        "orientation_cost": 1.0,
        "gain": 0.8,
        "lm_damping": 0.02,
    }

    limited_world = ErrorLimitedFrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        position_error_limit=0.003,
        orientation_error_limit=0.1,
        **task_kwargs,
    )
    native_world = mink.FrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        **task_kwargs,
    )
    limited_relative = ErrorLimitedRelativeFrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        root_name=root_name,
        root_type=root_type,
        position_error_limit=0.003,
        orientation_error_limit=0.1,
        **task_kwargs,
    )
    native_relative = mink.RelativeFrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        root_name=root_name,
        root_type=root_type,
        **task_kwargs,
    )

    results: dict[str, dict[str, float | int | bool]] = {
        "world": {"samples": samples * 4},
        "relative": {"samples": samples * 4},
    }
    activations = (0.0, 0.25, 0.75, 1.0)

    for _ in range(samples):
        configuration.update(
            q=_random_configuration(setup, home, qpos_indices, rng)
        )
        world_base = mink.SE3(
            wxyz_xyz=configuration._get_transform_frame_to_world_wxyz_xyz(
                frame_name,
                frame_type,
            )
        )
        relative_base = configuration.get_transform(
            frame_name,
            frame_type,
            root_name,
            root_type,
        )
        world_target = _perturbed_target(world_base, rng)
        relative_target = _perturbed_target(relative_base, rng)

        limited_world.set_target(world_target)
        native_world.set_target(world_target)
        limited_relative.set_target(relative_target)
        native_relative.set_target(relative_target)

        for activation in activations:
            limited_world.set_limit_activation(activation)
            limited_relative.set_limit_activation(activation)
            world_current = limited_world.compute_qp_objective(configuration)
            world_composed = _composed_bounded_objective(
                native_world,
                configuration,
                position_limit=0.003,
                orientation_limit=0.1,
                activation=activation,
            )
            relative_current = limited_relative.compute_qp_objective(configuration)
            relative_composed = _composed_bounded_objective(
                native_relative,
                configuration,
                position_limit=0.003,
                orientation_limit=0.1,
                activation=activation,
            )
            _update_maximum(
                results["world"],
                "H",
                world_current.H,
                world_composed.H,
            )
            _update_maximum(
                results["world"],
                "c",
                world_current.c,
                world_composed.c,
            )
            _update_maximum(
                results["relative"],
                "H",
                relative_current.H,
                relative_composed.H,
            )
            _update_maximum(
                results["relative"],
                "c",
                relative_current.c,
                relative_composed.c,
            )

    for result in results.values():
        result["equivalent"] = bool(
            result["H_max_abs"] <= 1e-10 and result["c_max_abs"] <= 1e-10
        )
    return results


def _validate_merged_joint_limit(
    setup: ArmSetup,
    configuration: mink.Configuration,
    home: np.ndarray,
    qpos_indices: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float | int | bool]:
    selected = {int(index) for index in qpos_indices}
    velocity_limits = _arm_velocity_limit_mapping(setup, None)
    recoverable = RecoverableConfigurationLimit(
        setup.model,
        selected,
        velocity_limits,
        gain=0.95,
        recovery_velocity_scale=1.0,
    )
    braking = JointBrakingLimit(
        setup.model,
        selected,
        velocity_limits,
        slowdown_distance=0.5,
        exponent=2.0,
        guard_margin=0.0,
        reaction_time=0.04,
        distance_buffer=0.01,
    )
    result: dict[str, float | int | bool] = {
        "configuration_samples": samples,
        "displacement_samples_per_configuration": 200,
        "feasibility_mismatches": 0,
        "projection_max_abs": 0.0,
        "merged_h_rule_max_abs": 0.0,
    }

    for _ in range(samples):
        q = _random_configuration(setup, home, qpos_indices, rng)
        configuration.update(q=q)
        measured_q = q.copy()
        measured_q[qpos_indices] += rng.uniform(-0.03, 0.03, qpos_indices.size)
        measured_dq = rng.uniform(-3.0, 3.0, setup.model.nv)
        braking.update_measured_state(measured_q, measured_dq)

        dt = float(rng.uniform(0.0002, 0.01))
        recoverable_constraint = recoverable.compute_qp_inequalities(
            configuration,
            dt,
        )
        braking_constraint = braking.compute_qp_inequalities(configuration, dt)
        projection_difference, _ = _max_difference(
            recoverable_constraint.G,
            braking_constraint.G,
        )
        result["projection_max_abs"] = max(
            float(result["projection_max_abs"]),
            projection_difference,
        )
        merged_h = np.minimum(recoverable_constraint.h, braking_constraint.h)
        explicit_h = np.minimum(
            recoverable_constraint.h,
            braking_constraint.h,
        )
        merged_h_difference, _ = _max_difference(merged_h, explicit_h)
        result["merged_h_rule_max_abs"] = max(
            float(result["merged_h_rule_max_abs"]),
            merged_h_difference,
        )

        delta_q = rng.uniform(
            -0.05,
            0.05,
            size=(int(result["displacement_samples_per_configuration"]), setup.model.nv),
        )
        recoverable_feasible = np.all(
            delta_q @ recoverable_constraint.G.T
            <= recoverable_constraint.h + 1e-12,
            axis=1,
        )
        braking_feasible = np.all(
            delta_q @ braking_constraint.G.T <= braking_constraint.h + 1e-12,
            axis=1,
        )
        merged_feasible = np.all(
            delta_q @ recoverable_constraint.G.T <= merged_h + 1e-12,
            axis=1,
        )
        result["feasibility_mismatches"] = int(
            result["feasibility_mismatches"]
        ) + int(
            np.count_nonzero(
                merged_feasible
                != (recoverable_feasible & braking_feasible)
            )
        )

    result["total_displacement_samples"] = int(
        samples * int(result["displacement_samples_per_configuration"])
    )
    result["equivalent"] = bool(
        result["projection_max_abs"] == 0.0
        and result["merged_h_rule_max_abs"] == 0.0
        and result["feasibility_mismatches"] == 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "dev/results/control_simplification_study_20260729/"
            "analysis/source_equivalence.json"
        ),
    )
    args = parser.parse_args()

    setup = _setup()
    home = setup.data.qpos.copy()
    qpos_indices = _active_qpos(setup)
    configuration = mink.Configuration(setup.model, q=home)
    rng = np.random.default_rng(20260729)

    results = {
        "kinetic_energy_native_vs_wrapper": _validate_kinetic_energy(
            setup,
            configuration,
            home,
            qpos_indices,
            rng,
            args.samples,
        ),
        "bounded_task_composition": _validate_bounded_tasks(
            setup,
            configuration,
            home,
            qpos_indices,
            rng,
            args.samples,
        ),
        "merged_joint_limit": _validate_merged_joint_limit(
            setup,
            configuration,
            home,
            qpos_indices,
            rng,
            args.samples,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))

    validations = [
        results["kinetic_energy_native_vs_wrapper"]["equivalent"]
        or results["kinetic_energy_native_vs_wrapper"]["wrapper_required"],
        results["bounded_task_composition"]["world"]["equivalent"],
        results["bounded_task_composition"]["relative"]["equivalent"],
        results["merged_joint_limit"]["equivalent"],
    ]
    if not all(validations):
        raise SystemExit("At least one source-simplification validation failed.")


if __name__ == "__main__":
    main()
