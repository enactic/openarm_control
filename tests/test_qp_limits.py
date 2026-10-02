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

"""Regression tests for Mink position, velocity, and singularity limits."""

from __future__ import annotations

import contextlib
import io
import xml.etree.ElementTree as ET
from unittest import mock

import mink
import mujoco
import numpy as np
import pytest

from openarm_control import ArmSetup, pose_to_se3
from openarm_control.geometry.jacobian import (
    normalized_arm_jacobian,
    singularity_ratio,
)
from openarm_control.qp.arm_joint_limit import (
    ArmConfigurationLimit,
    ArmJointLimit,
)
from openarm_control.qp.bounded_frame_task import BoundedFrameTask
from openarm_control.qp.singularity_approach_limit import (
    SingularityApproachLimit,
)
from _support import make_setup, velocity_mapping


def _make_joint_limit(
    *,
    braking_distance: float | None = 0.5,
) -> tuple[ArmJointLimit, mink.Configuration]:
    setup = make_setup("right")
    limit = ArmJointLimit(
        setup.model,
        setup.joint_resolver.arm_qpos_indices("right"),
        velocity_mapping("right"),
        position_gain=0.95,
        braking_distance=braking_distance,
        braking_exponent=2.0,
        braking_distance_buffer=0.01,
    )
    configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
    return limit, configuration


def test_center_uses_physical_velocity_cap() -> None:
    limit, configuration = _make_joint_limit(braking_distance=None)
    row = 0
    q = configuration.q
    q[limit.qpos_indices[row]] = 0.5 * (limit.lower[row] + limit.upper[row])
    configuration.update(q=q)
    constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
    assert constraint.h is not None
    assert constraint.G is limit.compute_qp_inequalities(configuration, dt=0.01).G

    assert constraint.h[row] == pytest.approx(limit.max_velocity[row] * 0.01)
    assert constraint.h[limit.indices.size + row] == pytest.approx(
        limit.max_velocity[row] * 0.01
    )


def test_configuration_limit_selects_scalar_joints_and_applies_gain() -> None:
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody>
          <body><joint name="arm_a" range="-1 1"/>
            <geom type="sphere" size="0.1" mass="1"/></body>
          <body pos="1 0 0"><joint name="object" type="ball" range="0 0.2"/>
            <geom type="sphere" size="0.1" mass="1"/></body>
          <body pos="2 0 0"><joint name="arm_b" range="-2 2"/>
            <geom type="sphere" size="0.1" mass="1"/></body>
        </worldbody></mujoco>
    """)
    configuration = mink.Configuration(model)
    configuration.update(q=np.array([0.25, np.cos(0.3), np.sin(0.3), 0, 0, -0.5]))
    limit = ArmConfigurationLimit(model, [5, 0], gain=0.8)
    constraint = limit.compute_qp_inequalities(configuration, dt=0.004)

    np.testing.assert_array_equal(limit.indices, [0, 4])
    np.testing.assert_array_equal(limit.qpos_indices, [0, 5])
    np.testing.assert_array_equal(
        constraint.G,
        [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [-1, 0, 0, 0, 0], [0, 0, 0, 0, -1]],
    )
    np.testing.assert_allclose(constraint.h, [0.6, 2.0, 1.0, 1.2])
    assert constraint.G is limit.compute_qp_inequalities(configuration, dt=0.004).G
    empty = ArmConfigurationLimit(model, [], gain=0.8)
    assert empty.compute_qp_inequalities(configuration, dt=0.004).inactive


def test_half_braking_distance_allows_quarter_velocity() -> None:
    limit, configuration = _make_joint_limit()
    row = 0
    q = configuration.q
    q[limit.qpos_indices[row]] = limit.lower[row] + 0.25
    configuration.update(q=q)
    constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
    assert constraint.h is not None

    lower_h = constraint.h[limit.indices.size + row]
    assert lower_h == pytest.approx(0.25 * limit.max_velocity[row] * 0.01)


def test_measured_q_reduces_effective_distance() -> None:
    limit, configuration = _make_joint_limit()
    row = 0
    qpos_index = limit.qpos_indices[row]
    measured_q = configuration.q
    measured_q[qpos_index] = limit.lower[row] + limit.braking_distance_buffer
    limit.update_measured_state(measured_q)
    constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
    assert constraint.h is not None

    assert constraint.h[limit.indices.size + row] == pytest.approx(0.0, abs=1e-7)


def test_overshoot_has_one_feasible_recovery_step() -> None:
    limit, configuration = _make_joint_limit()
    row = 0
    q = configuration.q
    q[limit.qpos_indices[row]] = limit.lower[row] - 0.1
    configuration.update(q=q)
    constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
    assert constraint.h is not None
    upper_step = constraint.h[row]
    lower_step = -constraint.h[limit.indices.size + row]

    assert lower_step > 0.0
    assert lower_step == pytest.approx(upper_step)
    assert upper_step <= limit.max_velocity[row] * 0.01


def test_unlimited_selected_joint_prints_warning_and_is_skipped() -> None:
    setup = make_setup("right")
    selected_qpos = setup.joint_resolver.arm_qpos_indices("right")
    skipped_qpos = int(selected_qpos[0])
    joint_id = next(
        joint_id
        for joint_id in range(setup.model.njnt)
        if int(setup.model.jnt_qposadr[joint_id]) == skipped_qpos
    )
    name = mujoco.mj_id2name(
        setup.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    )
    setup.model.jnt_limited[joint_id] = 0

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        limit = ArmJointLimit(
            setup.model,
            selected_qpos,
            velocity_mapping("right"),
            position_gain=0.95,
            braking_distance=None,
            braking_exponent=2.0,
            braking_distance_buffer=0.01,
        )

    assert f"selected arm joint {name!r}" in output.getvalue()
    assert skipped_qpos not in limit.qpos_indices


def _make_singularity_limit(
    root: str | None = None,
) -> tuple[ArmSetup, SingularityApproachLimit, mink.Configuration]:
    setup = make_setup("right")
    configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
    task_type = mink.FrameTask if root is None else mink.RelativeFrameTask
    task = task_type(
        frame_name="right_ee_control_point",
        frame_type="site",
        position_cost=10.0,
        orientation_cost=1.0,
        **({"root_name": root, "root_type": "body"} if root is not None else {}),
    )
    limit = SingularityApproachLimit(
        setup.model,
        task,
        setup.joint_resolver.arm_dof_indices("right"),
        characteristic_length=0.3,
        ratio_stop=0.02,
        ratio_slow=0.08,
        max_approach_rate=0.25,
        exponent=2.0,
    )
    return setup, limit, configuration


def _scalar_singularity_gradient(
    limit: SingularityApproachLimit, configuration: mink.Configuration
) -> np.ndarray:
    model = configuration.model
    q0 = configuration.q
    reference = mink.Configuration(model)
    gradient = np.empty(limit.dof_indices.size)
    tangent = np.zeros(model.nv)
    for index, dof in enumerate(limit.dof_indices):
        tangent[dof] = 1.0
        ratios = []
        for sign in (1.0, -1.0):
            q = q0.copy()
            mujoco.mj_integratePos(model, q, tangent, sign * limit.gradient_epsilon)
            reference.update(q=q)
            jacobian = normalized_arm_jacobian(
                limit.frame_task,
                reference,
                limit.dof_indices,
                limit.characteristic_length,
            )
            ratios.append(singularity_ratio(jacobian)[0])
        gradient[index] = (ratios[0] - ratios[1]) / (2.0 * limit.gradient_epsilon)
        tangent[dof] = 0.0
    return gradient


def test_only_approaching_gradient_component_is_bounded() -> None:
    _, limit, configuration = _make_singularity_limit()
    gradient = _scalar_singularity_gradient(limit, configuration)
    limit.prepare(configuration)
    constraint = limit.compute_qp_inequalities(configuration, dt=0.004)
    assert constraint.G is not None
    approach = np.zeros(configuration.model.nv)
    approach[limit.dof_indices] = -gradient
    assert float((constraint.G @ approach)[0]) > 0.0
    assert float((constraint.G @ (-approach))[0]) < 0.0


@pytest.mark.parametrize("straight", (False, True))
@pytest.mark.parametrize("root", (None, "world", "openarm_right_link4"))
def test_batched_gradient_matches_scalar_differences(
    straight: bool, root: str | None
) -> None:
    setup, limit, configuration = _make_singularity_limit(root)
    q0 = configuration.q
    if straight:
        q0[setup.joint_resolver.arm_qpos_indices("right")] = 0.0
    configuration.update(q=q0)
    actual = limit._finite_difference_gradient(configuration).copy()
    expected = _scalar_singularity_gradient(limit, configuration)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_array_equal(configuration.q, q0)


def test_scratch_update_preserves_measured_state_constraint() -> None:
    setup, limit, configuration = _make_singularity_limit()
    q = configuration.q
    q[setup.joint_resolver.arm_qpos_indices("right")] += np.linspace(-0.1, 0.1, 7)
    configuration.update(q=q)
    limit.update_measured_configuration(setup.data.qpos)

    with mock.patch.object(mujoco, "mj_makeConstraint") as make_constraint:
        limit.prepare(configuration)
    make_constraint.assert_not_called()
    actual = limit.compute_qp_inequalities(configuration, dt=0.004)

    with mock.patch.object(limit, "_update_scratch", limit._scratch.update):
        limit.prepare(configuration)
    reference = limit.compute_qp_inequalities(configuration, dt=0.004)
    np.testing.assert_allclose(actual.G, reference.G, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.h, reference.h, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(configuration.q, q)


def test_target_does_not_change_geometric_ratio() -> None:
    setup, limit, configuration = _make_singularity_limit()
    limit.prepare(configuration)
    initial = limit.compute_qp_inequalities(configuration, dt=0.004)
    assert initial.G is not None
    assert initial.h is not None

    wrapped = BoundedFrameTask(
        mink.FrameTask(
            "right_ee_control_point",
            "site",
            position_cost=10.0,
            orientation_cost=1.0,
        ),
        position_error_limit=0.015,
        orientation_error_limit=0.0,
        control_dt=0.004,
        substeps=5,
        target_linear_speed_slow=0.6,
        target_linear_speed_fast=0.9,
        position_error_latch_threshold=0.006,
    )
    target = setup.read_ee_pose("right").astype(np.float64)
    target[:3] += [0.2, -0.1, 0.15]
    wrapped.set_target(pose_to_se3(target))
    second = SingularityApproachLimit(
        setup.model,
        wrapped,
        limit.dof_indices,
        characteristic_length=0.3,
        ratio_stop=0.02,
        ratio_slow=0.08,
        max_approach_rate=0.25,
    )
    second.prepare(configuration)
    shifted = second.compute_qp_inequalities(configuration, dt=0.004)
    assert shifted.G is not None
    assert shifted.h is not None

    np.testing.assert_allclose(shifted.G, initial.G, atol=1e-12)
    np.testing.assert_allclose(shifted.h, initial.h, atol=1e-12)


@pytest.mark.parametrize("reverse", (False, True))
def test_jacobian_out_preserves_values_and_owned_results(reverse: bool) -> None:
    setup, limit, configuration = _make_singularity_limit()
    indices = limit.dof_indices.copy()
    if reverse:
        indices = indices[::-1]
    limit = SingularityApproachLimit(
        setup.model,
        limit.frame_task,
        indices,
        characteristic_length=0.3,
        ratio_stop=0.02,
        ratio_slow=0.08,
        max_approach_rate=0.25,
    )
    owned = limit._jacobian(configuration)
    saved = owned.copy()
    indices[:] = 0
    output = np.empty((6, 7))
    assert limit._jacobian(configuration, out=output) is output
    np.testing.assert_array_equal(output, owned)
    assert not np.shares_memory(owned, limit._world_jacobian)
    q = configuration.q
    q[setup.joint_resolver.arm_qpos_indices("right")] += 0.1
    configuration.update(q=q)
    limit._jacobian(configuration, out=output)
    np.testing.assert_array_equal(owned, saved)


def _make_articulated_pair() -> tuple[
    SingularityApproachLimit, SingularityApproachLimit, mink.Configuration
]:
    xml = ET.Element("mujoco")
    yaw = ET.SubElement(ET.SubElement(xml, "worldbody"), "body")
    ET.SubElement(yaw, "joint", name="yaw", axis="0 0 1")
    ET.SubElement(yaw, "geom", type="sphere", size="0.02", mass="1")
    base = ET.SubElement(yaw, "body")
    ET.SubElement(base, "joint", name="pitch", axis="0 1 0")
    ET.SubElement(base, "geom", type="sphere", size="0.02", mass="1")
    ET.SubElement(base, "site", name="arm_origin")
    for side, offset in (("right", "0 -0.2 0"), ("left", "0 0.2 0")):
        body = ET.SubElement(base, "body", pos=offset)
        for index in range(7):
            body = ET.SubElement(
                body, "body", name=f"{side}_link{index}", pos="0 0 0.1"
            )
            ET.SubElement(
                body,
                "joint",
                name=f"{side}_joint{index}",
                axis=("0 0 1", "0 1 0", "1 0 0")[index % 3],
            )
            ET.SubElement(body, "geom", type="sphere", size="0.02", mass="0.1")
        ET.SubElement(body, "site", name=f"{side}_tip", pos="0 0 0.1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    limits = []
    for side in ("right", "left"):
        task = mink.RelativeFrameTask(
            frame_name=f"{side}_tip",
            frame_type="site",
            root_name="arm_origin",
            root_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        limits.append(
            SingularityApproachLimit(
                model,
                task,
                [model.joint(f"{side}_joint{i}").dofadr[0] for i in range(7)],
                characteristic_length=0.3,
                ratio_stop=0.02,
                ratio_slow=0.08,
                max_approach_rate=0.25,
            )
        )
    return limits[0], limits[1], mink.Configuration(model)


@pytest.mark.parametrize("base_angles", ((0.0, 0.0), (0.6, -0.4)))
def test_paired_gradients_preserve_articulated_base(
    base_angles: tuple[float, float],
) -> None:
    right, left, configuration = _make_articulated_pair()
    q = np.linspace(-0.5, 0.6, configuration.model.nq)
    q[:2] = base_angles
    configuration.update(q=q)
    expected = [
        limit._finite_difference_gradient(configuration).copy()
        for limit in (right, left)
    ]

    assert right.can_share_perturbations(left)
    actual = right._paired_gradients(configuration, left)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_array_equal(configuration.q, q)


@pytest.mark.parametrize("change", ("frame", "root", "dof", "epsilon"))
def test_pairing_rejects_cross_dependencies_or_different_steps(change: str) -> None:
    right, left, _ = _make_articulated_pair()
    if change == "frame":
        left.frame_task.frame_name = "right_tip"
    elif change == "root":
        left.frame_task.root_name = "right_link3"
        left.frame_task.root_type = "body"
    elif change == "dof":
        left.dof_indices[0] = right.dof_indices[0]
    else:
        left.gradient_epsilon *= 2.0
    assert not right.can_share_perturbations(left)
    assert not left.can_share_perturbations(right)
