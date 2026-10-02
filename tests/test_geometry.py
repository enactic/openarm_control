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

"""Regression tests for pose and Jacobian geometry."""

from __future__ import annotations

import mink
import numpy as np
import pytest

from openarm_control import IKParams, Kinematics
from openarm_control.geometry.jacobian import (
    normalized_arm_jacobian,
    relative_root_is_independent_of_dofs,
)
from _support import make_setup


@pytest.mark.parametrize("operation", ("update", "integrate"))
def test_geometry_cache_is_read_only_and_invalidated(operation: str) -> None:
    setup = make_setup("right")
    solver = Kinematics(setup, IKParams())._ik
    assert solver is not None
    configuration = solver._config
    task = solver._nullspace_tasks["right"]
    frame_name, frame_type = "right_ee_control_point", "site"
    dofs = setup.joint_resolver.arm_dof_indices("right")
    jacobian = configuration.get_frame_jacobian(frame_name, frame_type)
    assert configuration.get_frame_jacobian(frame_name, frame_type) is jacobian
    spectrum, direction = task._svd(configuration)
    solver._tasks["right"].set_target(mink.SE3.identity())
    assert task._svd(configuration)[0] is spectrum
    assert not jacobian.flags.writeable
    assert not spectrum.flags.writeable
    assert not direction.flags.writeable

    if operation == "update":
        q = configuration.q
        q[setup.joint_resolver.arm_qpos_indices("right")[3]] += 0.1
        configuration.update(q=q)
    else:
        velocity = np.zeros(setup.model.nv)
        velocity[dofs[3]] = 0.1
        configuration.integrate_inplace(velocity, dt=1.0)
    new_jacobian = configuration.get_frame_jacobian(frame_name, frame_type)
    new_spectrum, _ = task._svd(configuration)
    assert new_jacobian is not jacobian
    assert new_spectrum is not spectrum
    reference = mink.Configuration(setup.model, q=configuration.q)
    np.testing.assert_allclose(
        new_jacobian, reference.get_frame_jacobian(frame_name, frame_type)
    )
    np.testing.assert_allclose(new_spectrum, task._svd(reference)[0])


def test_relative_root_fast_path_matches_full_formula() -> None:
    setup = make_setup("right", origin_frame="arm_origin")
    configuration = mink.Configuration(
        setup.model,
        q=setup.data.qpos.copy(),
    )
    task = mink.RelativeFrameTask(
        frame_name="right_ee_control_point",
        frame_type="site",
        root_name="arm_origin",
        root_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    dofs = setup.joint_resolver.arm_dof_indices("right")

    assert relative_root_is_independent_of_dofs(task, setup.model, dofs)
    fast = normalized_arm_jacobian(
        task,
        configuration,
        dofs,
        0.3,
        root_is_independent_of_dofs=True,
    )
    general = normalized_arm_jacobian(
        task,
        configuration,
        dofs,
        0.3,
    )
    np.testing.assert_array_equal(fast, general)


def test_moving_relative_root_uses_general_formula() -> None:
    setup = make_setup("right")
    configuration = mink.Configuration(
        setup.model,
        q=setup.data.qpos.copy(),
    )
    dofs = setup.joint_resolver.arm_dof_indices("right")
    task = mink.RelativeFrameTask(
        frame_name="right_ee_control_point",
        frame_type="site",
        root_name="openarm_right_link4",
        root_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    assert not relative_root_is_independent_of_dofs(task, setup.model, dofs)
    actual = normalized_arm_jacobian(
        task,
        configuration,
        dofs,
        0.3,
    )
    frame_jacobian = configuration.get_frame_jacobian(
        task.frame_name,
        task.frame_type,
    )[:, dofs]
    root_jacobian = configuration.get_frame_jacobian(
        task.root_name,
        task.root_type,
    )[:, dofs]
    transform_frame_to_root = configuration.get_transform(
        task.frame_name,
        task.frame_type,
        task.root_name,
        task.root_type,
    )
    expected = (
        frame_jacobian - transform_frame_to_root.inverse().adjoint() @ root_jacobian
    )
    expected[:3] /= 0.3
    np.testing.assert_allclose(actual, expected, atol=1e-12)
