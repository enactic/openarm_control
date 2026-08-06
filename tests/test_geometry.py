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

from openarm_control.geometry.jacobian import (
    normalized_arm_jacobian,
    relative_root_is_independent_of_dofs,
)
from _support import make_setup


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
