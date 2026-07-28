#!/usr/bin/env python3
"""Validate OpenArm IK frame conventions and single/bimanual mode equivalence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco
from scipy.spatial.transform import Rotation

from openarm_control import ArmSetup, IKParams, Kinematics
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S, WORLD_FRAME
from openarm_control.poses import read_ee_pose, relative_pose

START_Q_RIGHT = np.array(
    [-0.24633651, 0.12070639, 0.30554437, 2.1537668, 0.44992149, -0.02483132, 0.68082729]
)
START_Q_LEFT = np.array(
    [0.31404758, -0.30474089, -0.33635407, 2.195766, -0.39094594, 0.1, -0.685755]
)


def setup(mode: str, origin: str) -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
        origin_frame=origin,
        origin_frame_type="site",
    )


def velocity_mapping() -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": float(value)
        for side in ("right", "left")
        for index, value in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    }


def params(*, kinetic_energy_cost: float) -> IKParams:
    return IKParams(
        position_cost=10.0,
        orientation_cost=1.0,
        frame_position_error_limit=0.003,
        frame_orientation_error_limit=0.0,
        frame_error_limit_linear_slow=0.5,
        frame_error_limit_linear_fast=1.0,
        lm_damping=0.02,
        damping=0.1,
        posture_cost=0.0,
        dt=0.004,
        max_iters=5,
        velocity_limits=velocity_mapping(),
        joint_limit_recovery_velocity_scale=1.0,
        nullspace_cost=12.0,
        nullspace_return_rate=1.6,
        nullspace_max_speed=1.0,
        nullspace_singularity_low=0.02,
        nullspace_singularity_high=0.05,
        nullspace_characteristic_length=0.3,
        joint_limit_braking=True,
        joint_limit_braking_slowdown_distance=0.5,
        joint_limit_braking_exponent=2.0,
        joint_limit_braking_reaction_time=0.04,
        joint_limit_braking_distance_buffer=0.01,
        singularity_approach_limit=True,
        singularity_ratio_stop=0.02,
        singularity_ratio_slow=0.08,
        singularity_max_approach_rate=0.25,
        singularity_braking_exponent=2.0,
        kinetic_energy_cost=kinetic_energy_cost,
    )


def driver_state(arm_setup: ArmSetup) -> np.ndarray:
    arm_setup.joint_resolver.set_qpos(
        arm_setup.data.qpos,
        np.append(START_Q_RIGHT, 0.0),
        "right",
    )
    arm_setup.joint_resolver.set_qpos(
        arm_setup.data.qpos,
        np.append(START_Q_LEFT, 0.0),
        "left",
    )
    mujoco.mj_forward(arm_setup.model, arm_setup.data)
    right, right_gripper = arm_setup.joint_resolver.get_driver(
        arm_setup.data.qpos,
        "right",
    )
    left, left_gripper = arm_setup.joint_resolver.get_driver(
        arm_setup.data.qpos,
        "left",
    )
    return np.concatenate(
        [
            np.append(right, right_gripper),
            np.append(left, left_gripper),
        ]
    ).astype(np.float32)


def quat_to_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    )


def rotation_to_quat(rotation: Rotation) -> np.ndarray:
    quaternion = rotation.as_quat()
    return np.array(
        [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    )


def compose_pose(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    parent_rotation = quat_to_rotation(parent[3:])
    output = np.empty(7)
    output[:3] = parent[:3] + parent_rotation.apply(child[:3])
    output[3:] = rotation_to_quat(parent_rotation * quat_to_rotation(child[3:]))
    return output


def pose_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(first[:3] - second[:3]))
    angle = float(
        np.linalg.norm(
            (quat_to_rotation(first[3:]).inv() * quat_to_rotation(second[3:])).as_rotvec()
        )
    )
    return position, angle


def shifted_target(pose: np.ndarray, side: str) -> np.ndarray:
    target = pose.copy()
    mirror = -1.0 if side == "left" else 1.0
    target[:3] += np.array([0.012, mirror * 0.006, 0.004])
    rotation = quat_to_rotation(target[3:]) * Rotation.from_rotvec(
        [0.04, mirror * -0.03, 0.02]
    )
    target[3:] = rotation_to_quat(rotation)
    return target


def frame_validation() -> dict[str, object]:
    relative_setup = setup("bimanual", "arm_origin")
    world_setup = setup("bimanual", WORLD_FRAME)
    relative_state = driver_state(relative_setup)
    driver_state(world_setup)
    origin_id = mujoco.mj_name2id(
        world_setup.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "arm_origin",
    )
    origin_world = read_ee_pose(world_setup.data, origin_id, "site")

    sides: dict[str, object] = {}
    for side in ("right", "left"):
        relative_fk = relative_setup.read_ee_pose(side)
        world_fk = world_setup.read_ee_pose(side)
        composed = compose_pose(origin_world, relative_fk)
        roundtrip = relative_pose(origin_world, world_fk)
        compose_position, compose_angle = pose_difference(composed, world_fk)
        roundtrip_position, roundtrip_angle = pose_difference(roundtrip, relative_fk)
        sides[side] = {
            "compose_position_error_m": compose_position,
            "compose_orientation_error_rad": compose_angle,
            "roundtrip_position_error_m": roundtrip_position,
            "roundtrip_orientation_error_rad": roundtrip_angle,
        }

    return {
        "origin_site_id": origin_id,
        "origin_world_pose": origin_world.tolist(),
        "driver_state_shape": list(relative_state.shape),
        "sides": sides,
    }


def world_relative_solve_validation() -> dict[str, object]:
    relative_setup = setup("bimanual", "arm_origin")
    world_setup = setup("bimanual", WORLD_FRAME)
    relative_state = driver_state(relative_setup)
    world_state = driver_state(world_setup)
    relative_kinematics = Kinematics(relative_setup, params(kinetic_energy_cost=3e-5))
    world_kinematics = Kinematics(world_setup, params(kinetic_energy_cost=3e-5))
    relative_kinematics.sync(relative_state)
    world_kinematics.sync(world_state)

    origin_id = mujoco.mj_name2id(
        world_setup.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "arm_origin",
    )
    origin_world = read_ee_pose(world_setup.data, origin_id, "site")
    for side in ("right", "left"):
        relative_target = shifted_target(relative_setup.read_ee_pose(side), side)
        world_target = compose_pose(origin_world, relative_target)
        relative_kinematics.set_target(side, relative_target)
        world_kinematics.set_target(side, world_target)
    relative_result = relative_kinematics.solve()
    world_result = world_kinematics.solve()
    if relative_result is None or world_result is None:
        raise RuntimeError("World/relative equivalence solve unexpectedly failed.")
    difference = relative_result.astype(np.float64) - world_result.astype(np.float64)
    return {
        "max_joint_difference_rad": float(
            np.max(np.abs(np.concatenate([difference[:7], difference[8:15]])))
        ),
        "rms_joint_difference_rad": float(
            np.sqrt(
                np.mean(
                    np.square(np.concatenate([difference[:7], difference[8:15]]))
                )
            )
        ),
    }


def mode_validation(kinetic_energy_cost: float) -> dict[str, object]:
    setups = {
        mode: setup(mode, "arm_origin")
        for mode in ("right", "left", "bimanual")
    }
    states = {mode: driver_state(value) for mode, value in setups.items()}
    kinematics = {
        mode: Kinematics(value, params(kinetic_energy_cost=kinetic_energy_cost))
        for mode, value in setups.items()
    }
    for mode, value in kinematics.items():
        value.sync(states[mode])

    right_target = shifted_target(setups["right"].read_ee_pose("right"), "right")
    left_target = shifted_target(setups["left"].read_ee_pose("left"), "left")
    kinematics["right"].set_target("right", right_target)
    kinematics["left"].set_target("left", left_target)
    kinematics["bimanual"].set_target("right", right_target)
    kinematics["bimanual"].set_target("left", left_target)
    results = {mode: value.solve() for mode, value in kinematics.items()}
    if any(result is None for result in results.values()):
        raise RuntimeError("Single/bimanual mode equivalence solve unexpectedly failed.")
    right = results["right"]
    left = results["left"]
    bimanual = results["bimanual"]
    assert right is not None and left is not None and bimanual is not None
    return {
        "kinetic_energy_cost": kinetic_energy_cost,
        "right_max_difference_rad": float(
            np.max(np.abs(right[:7] - bimanual[:7]))
        ),
        "left_max_difference_rad": float(
            np.max(np.abs(left[8:15] - bimanual[8:15]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/results/ik_safety_study_20260728/compatibility.json"),
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(openarm_mujoco.openarm_cell_xml())
    origin_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "arm_origin",
    )
    result = {
        "openarm_mujoco_version": importlib.metadata.version("openarm-mujoco"),
        "model_xml": str(openarm_mujoco.openarm_cell_xml()),
        "arm_origin_present": origin_id >= 0,
        "frame_validation": frame_validation(),
        "world_relative_solve": world_relative_solve_validation(),
        "mode_without_energy": mode_validation(0.0),
        "mode_with_current_energy": mode_validation(3e-5),
    }
    result["passed"] = bool(
        result["arm_origin_present"]
        and result["world_relative_solve"]["max_joint_difference_rad"] < 5e-6
        and result["mode_without_energy"]["right_max_difference_rad"] < 5e-6
        and result["mode_without_energy"]["left_max_difference_rad"] < 5e-6
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
