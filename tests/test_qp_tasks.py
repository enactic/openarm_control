# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for Mink frame and nullspace tasks."""

from __future__ import annotations

from unittest import mock

import mink
import mujoco
import numpy as np
import pytest

from openarm_control import IKParams, Kinematics, pose_to_se3
from openarm_control.geometry.jacobian import normalized_arm_jacobian
from openarm_control.qp.bounded_frame_task import BoundedFrameTask
from openarm_control.qp.nullspace_posture_task import (
    NullspacePostureTask,
    smoothstep_activation,
    structural_nullspace_direction,
)
from tests._support import make_setup


def _make_task(
    *,
    orientation_error_limit: float = 0.0,
    substeps: int = 5,
) -> tuple[mink.Configuration, mink.FrameTask, BoundedFrameTask, np.ndarray]:
    setup = make_setup("right")
    configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
    native = mink.FrameTask(
        "right_ee_control_point",
        "site",
        position_cost=10.0,
        orientation_cost=1.0,
        lm_damping=0.01,
    )
    task = BoundedFrameTask(
        native,
        position_error_limit=0.015,
        orientation_error_limit=orientation_error_limit,
        control_dt=0.004,
        substeps=substeps,
        target_linear_speed_slow=0.6,
        target_linear_speed_fast=0.9,
        position_error_latch_threshold=0.006,
    )
    return (
        configuration,
        native,
        task,
        setup.read_ee_pose("right").astype(np.float64),
    )


def test_limits_request_but_preserves_full_error_and_target() -> None:
    configuration, native, task, target = _make_task()
    target[0] += 0.1
    task.set_target(pose_to_se3(target))

    full_error = task.compute_full_error(configuration)
    limited_error = task.compute_limited_error(configuration)

    assert float(np.linalg.norm(full_error[:3])) > 0.09
    assert float(np.linalg.norm(limited_error[:3])) == pytest.approx(0.015 / 5)
    np.testing.assert_array_equal(limited_error[3:], full_error[3:])
    assert native.transform_target_to_world is not None


def test_zero_activation_is_exactly_the_native_objective() -> None:
    configuration, native, task, target = _make_task()
    target[:3] += [0.03, -0.02, 0.01]
    task.set_target(pose_to_se3(target))
    task.set_limit_activation(0.0)

    native_objective = native.compute_qp_objective(configuration)
    wrapped_objective = task.compute_qp_objective(configuration)

    np.testing.assert_array_equal(wrapped_objective.H, native_objective.H)
    np.testing.assert_array_equal(wrapped_objective.c, native_objective.c)


def test_orientation_limit_is_independent_of_position_activation() -> None:
    configuration, _, task, target = _make_task(orientation_error_limit=0.20)
    target[:3] += [0.03, -0.02, 0.01]
    rotation = np.array(
        [np.cos(0.5), np.sin(0.5), 0.0, 0.0],
        dtype=np.float64,
    )
    mujoco.mju_mulQuat(target[3:7], target[3:7].copy(), rotation)
    task.set_target(pose_to_se3(target))
    task.set_limit_activation(0.0)

    full_error = task.compute_full_error(configuration)
    limited_error = task.compute_limited_error(configuration)
    assert float(np.linalg.norm(full_error[3:])) > 0.9
    assert float(np.linalg.norm(limited_error[3:])) == pytest.approx(0.20 / 5)

    expected_error = full_error.copy()
    expected_error[3:] = limited_error[3:]
    expected = task._assemble_qp(
        expected_error,
        task.compute_jacobian(configuration),
        np.eye(configuration.model.nv),
    )
    actual = task.compute_qp_objective(configuration)
    np.testing.assert_allclose(actual.H, expected.H)
    np.testing.assert_allclose(actual.c, expected.c)


def test_objective_computes_native_error_once() -> None:
    configuration, native, task, target = _make_task(orientation_error_limit=0.20)
    target[:3] += [0.03, -0.02, 0.01]
    task.set_target(pose_to_se3(target))

    with mock.patch.object(
        native,
        "compute_error",
        wraps=native.compute_error,
    ) as compute_error:
        task.compute_qp_objective(configuration)

    compute_error.assert_called_once_with(configuration)


@pytest.mark.parametrize("substeps", (1, 5, 10))
def test_total_error_budgets_are_independent_of_substep_count(
    substeps: int,
) -> None:
    configuration, _, task, target = _make_task(
        orientation_error_limit=0.20,
        substeps=substeps,
    )
    target[0] += 0.1
    rotation = np.array(
        [np.cos(0.5), np.sin(0.5), 0.0, 0.0],
        dtype=np.float64,
    )
    mujoco.mju_mulQuat(target[3:7], target[3:7].copy(), rotation)
    task.set_target(pose_to_se3(target))

    limited_error = task.compute_limited_error(configuration)
    assert float(np.linalg.norm(limited_error[:3])) * substeps == pytest.approx(0.015)
    assert float(np.linalg.norm(limited_error[3:])) * substeps == pytest.approx(0.20)


def test_latch_uses_fixed_outer_position_error_threshold() -> None:
    configuration, _, latched, pose = _make_task(substeps=10)
    above_threshold = pose.copy()
    above_threshold[0] += 0.010
    latched.set_target_and_update_schedule(
        pose_to_se3(above_threshold),
        configuration,
    )
    assert latched.limit_activation == 1.0

    configuration, _, released, pose = _make_task(substeps=1)
    below_threshold = pose.copy()
    below_threshold[0] += 0.005
    released.set_target_and_update_schedule(
        pose_to_se3(below_threshold),
        configuration,
    )
    assert released.limit_activation == 0.0


def test_speed_schedule_is_instant_and_latches_accumulated_error() -> None:
    setup = make_setup("right")
    kinematics = Kinematics(
        setup,
        IKParams(
            dt=0.004,
            max_iters=5,
            posture_cost=0.0,
        ),
    )
    pose = setup.read_ee_pose("right").astype(np.float64)
    kinematics.set_target("right", pose)
    medium = pose.copy()
    medium[0] += 0.003  # 0.75 m/s, midpoint of the 0.6 -> 0.9 window.
    kinematics.set_target("right", medium)

    solver = kinematics._ik
    assert solver is not None
    task = solver._tasks["right"]
    assert isinstance(task, BoundedFrameTask)
    assert task.limit_activation == pytest.approx(0.5)

    far = medium.copy()
    far[0] += 0.02
    kinematics.set_target("right", far)
    assert task.limit_activation == 1.0
    kinematics.set_target("right", far)
    assert task.limit_activation == 1.0


def test_structural_direction_is_null_and_sign_continuous() -> None:
    rng = np.random.default_rng(7)
    jacobian = rng.normal(size=(6, 7))
    direction, singular_values = structural_nullspace_direction(jacobian)
    aligned, _ = structural_nullspace_direction(jacobian, previous=-direction)

    assert singular_values.shape == (6,)
    assert float(np.linalg.norm(direction)) == pytest.approx(1.0)
    assert float(np.linalg.norm(jacobian @ direction)) < 1e-12
    assert float(aligned @ (-direction)) > 1.0 - 1e-12


def test_smooth_activation_has_flat_clamped_endpoints() -> None:
    assert smoothstep_activation(0.02, 0.02, 0.05) == 0.0
    assert smoothstep_activation(0.035, 0.02, 0.05) == pytest.approx(0.5)
    assert smoothstep_activation(0.05, 0.02, 0.05) == 1.0


def test_task_is_nullspace_only_and_caps_return_speed() -> None:
    setup = make_setup("right")
    configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
    frame_task = mink.FrameTask(
        "right_ee_control_point",
        "site",
        position_cost=10.0,
        orientation_cost=1.0,
    )
    dofs = setup.joint_resolver.arm_dof_indices("right")
    task = NullspacePostureTask(
        model=setup.model,
        frame_task=frame_task,
        dof_indices=dofs,
        home_qpos=configuration.q,
        cost=12.0,
        dt=0.0008,
        return_rate=100.0,
        max_speed=1.0,
        singularity_low=0.0,
        singularity_high=1e-9,
        characteristic_length=0.3,
    )
    _, initial_jacobian = task._compute_terms(configuration)
    initial_direction = initial_jacobian[0, dofs].copy()

    q = configuration.q
    tangent = np.zeros(setup.model.nv)
    tangent[dofs] = initial_direction
    mujoco.mj_integratePos(setup.model, q, tangent, 0.5)
    configuration.update(q=q)
    error, jacobian = task._compute_terms(configuration)
    direction = jacobian[0, dofs]
    geometric_jacobian = normalized_arm_jacobian(
        frame_task,
        configuration,
        dofs,
        0.3,
    )
    return_speed = -float(error[0]) / 0.0008

    assert float(np.linalg.norm(geometric_jacobian @ direction)) < 1e-10
    assert abs(return_speed) <= 1.0
    assert abs(float(error[0])) == pytest.approx(0.0008)


def test_sync_does_not_move_home_reference() -> None:
    setup = make_setup()
    kinematics = Kinematics(
        setup,
        IKParams(
            dt=0.004,
            max_iters=5,
            posture_cost=0.0,
        ),
    )
    solver = kinematics._ik
    assert solver is not None
    homes = {
        side: task._home_qpos.copy() for side, task in solver._nullspace_tasks.items()
    }
    kinematics.sync(np.linspace(-0.2, 0.2, 16, dtype=np.float32))
    for side, task in solver._nullspace_tasks.items():
        np.testing.assert_array_equal(task._home_qpos, homes[side])
