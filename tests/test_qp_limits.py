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
import unittest

import mink
import mujoco
import numpy as np

from openarm_control import ArmSetup, pose_to_se3
from openarm_control.qp.arm_joint_limit import (
    ArmConfigurationLimit,
    ArmJointLimit,
)
from openarm_control.qp.bounded_frame_task import BoundedFrameTask
from openarm_control.qp.singularity_approach_limit import (
    SingularityApproachLimit,
)
from tests._support import make_setup, velocity_mapping


class ArmJointLimitTest(unittest.TestCase):
    """Exercise the merged position, velocity, and braking inequalities."""

    def _limit(
        self,
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

    def test_center_uses_physical_velocity_cap(self) -> None:
        limit, configuration = self._limit(braking_distance=None)
        row = 0
        q = configuration.q
        q[limit.qpos_indices[row]] = 0.5 * (limit.lower[row] + limit.upper[row])
        configuration.update(q=q)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None

        self.assertAlmostEqual(
            constraint.h[row],
            limit.max_velocity[row] * 0.01,
        )
        self.assertAlmostEqual(
            constraint.h[limit.indices.size + row],
            limit.max_velocity[row] * 0.01,
        )

    def test_configuration_limit_uses_selected_dofs_and_gain(self) -> None:
        setup = make_setup("right")
        limit = ArmConfigurationLimit(
            setup.model,
            setup.joint_resolver.arm_qpos_indices("right"),
            gain=0.8,
        )

        np.testing.assert_array_equal(
            limit.indices,
            setup.joint_resolver.arm_dof_indices("right"),
        )
        self.assertEqual(limit.gain, 0.8)

    def test_half_braking_distance_allows_quarter_velocity(self) -> None:
        limit, configuration = self._limit()
        row = 0
        q = configuration.q
        q[limit.qpos_indices[row]] = limit.lower[row] + 0.25
        configuration.update(q=q)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None

        lower_h = constraint.h[limit.indices.size + row]
        self.assertAlmostEqual(
            lower_h,
            0.25 * limit.max_velocity[row] * 0.01,
        )

    def test_measured_q_reduces_effective_distance(self) -> None:
        limit, configuration = self._limit()
        row = 0
        qpos_index = limit.qpos_indices[row]
        measured_q = configuration.q
        measured_q[qpos_index] = limit.lower[row] + limit.braking_distance_buffer
        limit.update_measured_state(measured_q)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None

        self.assertAlmostEqual(constraint.h[limit.indices.size + row], 0.0)

    def test_overshoot_has_one_feasible_recovery_step(self) -> None:
        limit, configuration = self._limit()
        row = 0
        q = configuration.q
        q[limit.qpos_indices[row]] = limit.lower[row] - 0.1
        configuration.update(q=q)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None
        upper_step = constraint.h[row]
        lower_step = -constraint.h[limit.indices.size + row]

        self.assertGreater(lower_step, 0.0)
        self.assertAlmostEqual(lower_step, upper_step)
        self.assertLessEqual(upper_step, limit.max_velocity[row] * 0.01)

    def test_unlimited_selected_joint_prints_warning_and_is_skipped(self) -> None:
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

        self.assertIn(f"selected arm joint {name!r}", output.getvalue())
        self.assertNotIn(skipped_qpos, limit.qpos_indices)


class SingularityLimitTest(unittest.TestCase):
    """Exercise one-sided geometric singularity approach limiting."""

    def _limit(
        self,
    ) -> tuple[ArmSetup, SingularityApproachLimit, mink.Configuration]:
        setup = make_setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        task = mink.FrameTask(
            "right_ee_control_point",
            "site",
            position_cost=10.0,
            orientation_cost=1.0,
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

    def test_only_approaching_gradient_component_is_bounded(self) -> None:
        _, limit, configuration = self._limit()
        limit.prepare(configuration)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.004)
        assert constraint.G is not None
        gradient = -constraint.G[0, limit.dof_indices]
        approach = np.zeros(configuration.model.nv)
        approach[limit.dof_indices] = -gradient
        self.assertGreater(float((constraint.G @ approach)[0]), 0.0)
        self.assertLess(float((constraint.G @ (-approach))[0]), 0.0)

    def test_target_does_not_change_geometric_ratio(self) -> None:
        setup, limit, configuration = self._limit()
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


if __name__ == "__main__":
    unittest.main()
