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

"""Integration tests for OpenArm differential IK."""

from __future__ import annotations

import unittest
from unittest import mock

import mink
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco

from openarm_control import ArmSetup, IKParams, Kinematics
from openarm_control.qp.arm_joint_limit import ArmJointLimit
from openarm_control.qp.bounded_frame_task import BoundedFrameTask
from tests._support import driver_state, make_setup, velocity_mapping


class ArmSetupTest(unittest.TestCase):
    """Verify driver qpos mapping respects MuJoCo configuration indices."""

    def test_driver_qpos_mapping_updates_only_active_arm(self) -> None:
        setup = make_setup("right")
        base_qpos = setup.data.qpos.copy()
        driver_qpos = driver_state(setup).astype(np.float64)
        driver_qpos[:7] += 0.1
        driver_qpos[8:15] += 0.2

        model_qpos = setup.driver_qpos_to_mujoco(
            driver_qpos,
            base_qpos=base_qpos,
        )

        np.testing.assert_array_equal(
            model_qpos[setup.joint_resolver.arm_qpos_indices("right")],
            driver_qpos[:7],
        )
        np.testing.assert_array_equal(
            model_qpos[setup.joint_resolver.arm_qpos_indices("left")],
            base_qpos[setup.joint_resolver.arm_qpos_indices("left")],
        )


class RelativeFrameTest(unittest.TestCase):
    """Exercise upstream relative-frame handling through the wrapper."""

    def test_latest_model_contains_default_arm_origin(self) -> None:
        setup = ArmSetup.from_args(
            xml=openarm_mujoco.openarm_cell_xml(),
            mode="right",
            frame_right="right_ee_control_point",
            frame_type_right="site",
            frame_left="left_ee_control_point",
            frame_type_left="site",
        )
        self.assertGreaterEqual(setup.origin_id, 0)
        self.assertEqual(setup.read_ee_pose("right").shape, (7,))

    def test_vr_solver_wraps_native_relative_frame_task(self) -> None:
        setup = make_setup(
            "right",
            origin_frame="openarm_right_base_link",
            origin_frame_type="body",
        )
        kinematics = Kinematics(
            setup,
            IKParams(
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=0.01,
                damping=0.1,
                posture_cost=0.0,
                dt=0.004,
                max_iters=5,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        task = solver._tasks["right"]
        self.assertIsInstance(task, BoundedFrameTask)
        assert isinstance(task, BoundedFrameTask)
        self.assertIsInstance(task.frame_task, mink.RelativeFrameTask)
        self.assertEqual(task.frame_task.root_name, "openarm_right_base_link")

        kinematics.set_target("right", setup.read_ee_pose("right"))
        self.assertIsNotNone(kinematics.solve())


class SolverTest(unittest.TestCase):
    """Exercise solver wiring, timing, state handling, and failure recovery."""

    def test_flat_params_wire_control_features_and_full_posture(self) -> None:
        setup = make_setup("right")
        params = IKParams(
            posture_cost=0.01,
            dt=0.004,
            velocity_limits=velocity_mapping("right"),
        )
        kinematics = Kinematics(
            setup,
            params,
        )
        solver = kinematics._ik
        assert solver is not None

        self.assertEqual(solver._posture_cost, 0.01)
        assert solver._joint_limit is not None
        self.assertEqual(solver._joint_limit.braking_distance, 0.2)
        self.assertIsInstance(solver._tasks["right"], BoundedFrameTask)
        self.assertIsInstance(solver._joint_limit, ArmJointLimit)
        self.assertEqual(
            solver._joint_limit.braking_distance,
            params.joint_braking_distance,
        )
        self.assertIn("right", solver._nullspace_tasks)
        self.assertIn("right", solver._singularity_limits)
        self.assertIsNotNone(solver._kinetic_energy_task)

        kinematics.set_target("right", setup.read_ee_pose("right"))
        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=np.zeros(setup.model.nv),
        ) as solve:
            self.assertIsNotNone(kinematics.solve())
        self.assertIn(solver._posture_task, solve.call_args.args[1])

    def test_substeps_cover_one_control_period_in_physical_units(self) -> None:
        setup = make_setup()
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(cap)
            for side in ("left", "right")
            for index, cap in enumerate(np.linspace(0.2, 0.8, 7))
        }
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                damping=0.25,
                posture_cost=0.0,
                velocity_limits=velocity_limits,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        self.assertEqual(solver._substep_dt, 0.0004)
        self.assertIsInstance(solver._limits[0], ArmJointLimit)

        velocity = np.zeros(setup.model.nv)
        for side in ("right", "left"):
            velocity[setup.joint_resolver.arm_dof_indices(side)] = np.linspace(
                0.2, 0.8, 7
            )
        q_before = solver._config.q.copy()
        for side in setup.sides:
            kinematics.set_target(side, setup.read_ee_pose(side))

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=velocity,
        ) as solve:
            result = kinematics.solve()

        self.assertIsNotNone(result)
        self.assertEqual(solve.call_count, 10)
        for call in solve.call_args_list:
            self.assertEqual(call.args[2], 0.0004)
            self.assertEqual(call.kwargs["damping"], 0.25)
            self.assertNotIn("diag_reg", call.kwargs)
        np.testing.assert_allclose(
            solver._config.q - q_before,
            velocity * 0.004,
            atol=1e-12,
        )

    def test_measured_state_does_not_sync_command_and_can_be_cleared(self) -> None:
        setup = make_setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                velocity_limits=velocity_mapping("right", "left"),
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        command_before = solver._config.q.copy()
        measured = driver_state(setup)
        measured[0] += 0.1

        kinematics.update_measured_state(measured)

        np.testing.assert_array_equal(solver._config.q, command_before)
        assert solver._joint_limit is not None
        self.assertIsNotNone(solver._joint_limit._measured_qpos)
        kinematics.clear_measured_state()
        self.assertIsNone(solver._joint_limit._measured_qpos)

    def test_sync_does_not_overwrite_gripper_command(self) -> None:
        setup = make_setup()
        kinematics = Kinematics(setup, IKParams())
        kinematics.set_gripper("right", 0.7)
        state = driver_state(setup)
        state[7] = 0.1
        kinematics.sync(state)
        solver = kinematics._ik
        assert solver is not None
        self.assertAlmostEqual(float(solver._gripper[0]), 0.7)

    def test_failure_rolls_back_and_requires_fresh_targets(self) -> None:
        setup = make_setup()
        kinematics = Kinematics(
            setup,
            IKParams(dt=0.004, max_iters=3, posture_cost=0.0),
        )
        solver = kinematics._ik
        assert solver is not None
        for side in setup.sides:
            kinematics.set_target(side, setup.read_ee_pose(side))
        q_before = solver._config.q.copy()
        velocity = np.zeros(setup.model.nv)
        velocity[solver._arm_dofs_by_side["right"][0]] = 0.5

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            side_effect=[velocity, mink.exceptions.NoSolutionFound("daqp")],
        ):
            result = kinematics.solve()

        self.assertIsNone(result)
        np.testing.assert_array_equal(solver._config.q, q_before)
        self.assertFalse(kinematics.ready())
        kinematics.set_target("right", setup.read_ee_pose("right"))
        self.assertFalse(kinematics.ready())
        kinematics.set_target("left", setup.read_ee_pose("left"))
        self.assertTrue(kinematics.ready())

    def test_right_left_and_bimanual_real_qp_solve(self) -> None:
        for mode in ("right", "left", "bimanual"):
            with self.subTest(mode=mode):
                setup = make_setup(mode, origin_frame="arm_origin")
                kinematics = Kinematics(
                    setup,
                    IKParams(
                        position_cost=10.0,
                        orientation_cost=1.0,
                        lm_damping=0.01,
                        damping=0.1,
                        posture_cost=0.0,
                        dt=0.004,
                        max_iters=5,
                        velocity_limits=velocity_mapping(*setup.sides),
                    ),
                )
                for side in setup.sides:
                    kinematics.set_target(side, setup.read_ee_pose(side))
                result = kinematics.solve()
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.shape, (16,))

    def test_single_arm_mode_freezes_inactive_arm_dofs(self) -> None:
        setup = make_setup("right")
        kinematics = Kinematics(setup, IKParams())
        solver = kinematics._ik
        assert solver is not None
        assert solver._freeze_task is not None

        frozen = set(solver._freeze_task.dof_indices)
        right = set(setup.joint_resolver.arm_dof_indices("right"))
        left = set(setup.joint_resolver.arm_dof_indices("left"))
        self.assertTrue(left <= frozen)
        self.assertTrue(right.isdisjoint(frozen))

    def test_kinetic_energy_uses_current_mujoco_inertia_api(self) -> None:
        setup = make_setup("right")
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                kinetic_energy_cost=1e-7,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        task = solver._kinetic_energy_task
        assert task is not None
        objective = task.compute_qp_objective(solver._config)

        self.assertEqual(objective.H.shape, (setup.model.nv, setup.model.nv))
        self.assertTrue(np.all(np.isfinite(objective.H)))
        np.testing.assert_allclose(objective.H, objective.H.T, atol=1e-12)
        np.testing.assert_array_equal(objective.c, np.zeros(setup.model.nv))


if __name__ == "__main__":
    unittest.main()
