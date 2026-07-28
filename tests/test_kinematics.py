"""Regression tests for OpenArm differential IK timing and nullspace control."""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import unittest
from unittest import mock

import mink
import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco

from openarm_control import (
    ArmSetup,
    ErrorLimitedFrameTask,
    ErrorLimitedRelativeFrameTask,
    IKParams,
    JointBrakingLimit,
    Kinematics,
    RecoverableConfigurationLimit,
    SingularityApproachLimit,
    ik_params_from_args,
    pose_to_se3,
    read_ee_pose,
    register_ik_args,
)
from openarm_control.config import (
    ARM_JOINT_VELOCITY_LIMITS_RAD_S,
    WORLD_FRAME,
)
from openarm_control.kinematics import (
    _arm_qpos_indices,
    _dof_indices_for_qpos,
)
from openarm_control.nullspace_posture_task import (
    NullspacePostureTask,
    smoothstep_activation,
    structural_nullspace_direction,
)


def _setup(
    mode: str = "bimanual",
    *,
    origin_frame: str = WORLD_FRAME,
    origin_frame_type: str = "site",
) -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
        origin_frame=origin_frame,
        origin_frame_type=origin_frame_type,
    )


class NullspaceMathTest(unittest.TestCase):
    """Exercise the pure SVD and singularity-activation helpers."""

    def test_structural_direction_is_null_and_sign_continuous(self) -> None:
        rng = np.random.default_rng(7)
        jacobian = rng.normal(size=(6, 7))

        direction, singular_values = structural_nullspace_direction(jacobian)
        aligned, _ = structural_nullspace_direction(jacobian, previous=-direction)

        self.assertEqual(singular_values.shape, (6,))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        self.assertLess(float(np.linalg.norm(jacobian @ direction)), 1e-12)
        self.assertGreater(float(aligned @ (-direction)), 1.0 - 1e-12)

    def test_smoothstep_activation_has_clamped_flat_endpoints(self) -> None:
        self.assertEqual(smoothstep_activation(0.0, 0.02, 0.05), 0.0)
        self.assertEqual(smoothstep_activation(0.02, 0.02, 0.05), 0.0)
        self.assertAlmostEqual(smoothstep_activation(0.035, 0.02, 0.05), 0.5)
        self.assertEqual(smoothstep_activation(0.05, 0.02, 0.05), 1.0)
        self.assertEqual(smoothstep_activation(1.0, 0.02, 0.05), 1.0)


class NullspaceTaskTest(unittest.TestCase):
    """Exercise the task against the actual OpenArm MuJoCo model."""

    def test_task_is_nullspace_only_and_caps_home_return_speed(self) -> None:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        frame_task = mink.FrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
        )
        frame_task.set_target_from_configuration(configuration)
        dof_indices = _dof_indices_for_qpos(
            setup.model, _arm_qpos_indices(setup, "right")
        )
        home_qpos = configuration.q
        task = NullspacePostureTask(
            model=setup.model,
            frame_task=frame_task,
            dof_indices=dof_indices,
            home_qpos=home_qpos,
            cost=0.3,
            dt=0.0004,
            return_rate=100.0,
            max_speed=0.5,
            singularity_low=0.0,
            singularity_high=1e-9,
            characteristic_length=0.3,
        )

        task.compute_qp_objective(configuration)
        initial_state = task.last_state
        self.assertIsNotNone(initial_state)
        assert initial_state is not None

        displaced_qpos = configuration.q
        tangent = np.zeros(setup.model.nv)
        tangent[dof_indices] = initial_state.direction
        mujoco.mj_integratePos(setup.model, displaced_qpos, tangent, 0.5)
        configuration.update(q=displaced_qpos)

        objective = task.compute_qp_objective(configuration)
        state = task.last_state
        self.assertIsNotNone(state)
        assert state is not None

        self.assertLess(state.jacobian_residual, 1e-10)
        self.assertLessEqual(abs(state.return_speed), 0.5)
        self.assertAlmostEqual(abs(state.displacement), 0.5 * 0.0004)
        self.assertAlmostEqual(state.effective_cost, np.sqrt(state.activation) * 0.3)

        z_full = np.zeros(setup.model.nv)
        z_full[dof_indices] = state.direction
        expected_h = 0.3**2 * state.activation * np.outer(z_full, z_full)
        expected_c = -(0.3**2) * state.activation * state.displacement * z_full
        np.testing.assert_allclose(objective.H, expected_h, atol=1e-12)
        np.testing.assert_allclose(objective.c, expected_c, atol=1e-12)

    def test_sync_does_not_change_fixed_home_targets(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                posture_cost=0.01,
                nullspace_cost=0.3,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        posture_home = solver._posture_task.target_q.copy()
        nullspace_homes = {
            side: task._home_qpos.copy()
            for side, task in solver._nullspace_tasks.items()
        }

        measured = np.linspace(-0.2, 0.2, 16, dtype=np.float32)
        kinematics.sync(measured)

        np.testing.assert_array_equal(solver._posture_task.target_q, posture_home)
        for side, task in solver._nullspace_tasks.items():
            np.testing.assert_array_equal(task._home_qpos, nullspace_homes[side])

    def test_target_error_does_not_change_geometric_nullspace(self) -> None:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        frame_task = mink.FrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
        )
        frame_task.set_target_from_configuration(configuration)
        task = NullspacePostureTask(
            model=setup.model,
            frame_task=frame_task,
            dof_indices=_dof_indices_for_qpos(
                setup.model,
                _arm_qpos_indices(setup, "right"),
            ),
            home_qpos=configuration.q,
            cost=0.3,
            dt=0.0004,
            return_rate=0.5,
            max_speed=0.5,
            singularity_low=0.02,
            singularity_high=0.05,
            characteristic_length=0.3,
        )

        task.compute_qp_objective(configuration)
        initial_state = task.last_state
        assert initial_state is not None

        target = setup.read_ee_pose("right").astype(np.float64)
        target[:3] += np.array([0.2, -0.1, 0.15])
        frame_task.set_target(pose_to_se3(target))
        task.compute_qp_objective(configuration)
        shifted_state = task.last_state
        assert shifted_state is not None

        np.testing.assert_allclose(
            shifted_state.singular_values,
            initial_state.singular_values,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            abs(float(shifted_state.direction @ initial_state.direction)),
            1.0,
        )
        self.assertAlmostEqual(
            shifted_state.singularity_ratio,
            initial_state.singularity_ratio,
        )
        self.assertAlmostEqual(shifted_state.activation, initial_state.activation)


class ErrorLimitedFrameTaskTest(unittest.TestCase):
    """Exercise the non-augmented per-substep Cartesian error limit."""

    def test_position_error_is_limited_without_changing_stored_target(self) -> None:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        task = ErrorLimitedFrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            position_error_limit=0.001,
            orientation_error_limit=0.0,
        )
        target = setup.read_ee_pose("right").astype(np.float64)
        target[0] += 0.1
        task.set_target(pose_to_se3(target))

        full_error = mink.FrameTask.compute_error(task, configuration)
        limited_error = task.compute_limited_error(configuration)

        self.assertGreater(float(np.linalg.norm(full_error[:3])), 0.09)
        self.assertAlmostEqual(float(np.linalg.norm(limited_error[:3])), 0.001)
        np.testing.assert_array_equal(limited_error[3:], full_error[3:])
        self.assertIsNotNone(task.transform_target_to_world)

    def test_zero_limits_match_native_frame_task_objective(self) -> None:
        setup = _setup("right")
        configuration = mink.Configuration(setup.model, q=setup.data.qpos.copy())
        native = mink.FrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=0.01,
        )
        limited = ErrorLimitedFrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            position_error_limit=0.0,
            orientation_error_limit=0.0,
            lm_damping=0.01,
        )
        target = setup.read_ee_pose("right").astype(np.float64)
        target[:3] += np.array([0.03, -0.02, 0.01])
        target_se3 = pose_to_se3(target)
        native.set_target(target_se3)
        limited.set_target(target_se3)

        native_objective = native.compute_qp_objective(configuration)
        limited_objective = limited.compute_qp_objective(configuration)

        np.testing.assert_array_equal(limited_objective.H, native_objective.H)
        np.testing.assert_array_equal(limited_objective.c, native_objective.c)

    def test_solver_preserves_relative_frame_support(self) -> None:
        setup = _setup(
            "right",
            origin_frame="openarm_right_base_link",
            origin_frame_type="body",
        )
        kinematics = Kinematics(
            setup,
            IKParams(
                frame_position_error_limit=0.003,
                frame_error_limit_linear_slow=0.2,
                frame_error_limit_linear_fast=0.5,
                frame_error_limit_activation_rise_rate=4.0,
                nullspace_cost=0.3,
                singularity_approach_limit=True,
                dt=0.004,
                max_iters=5,
            ),
        )
        pose = setup.read_ee_pose("right").astype(np.float64)
        kinematics.set_target("right", pose)
        translated = pose.copy()
        translated[0] += 0.004
        kinematics.set_target("right", translated)

        solver = kinematics._ik
        assert solver is not None
        task = solver._tasks["right"]
        self.assertIsInstance(task, ErrorLimitedRelativeFrameTask)
        assert isinstance(task, ErrorLimitedRelativeFrameTask)
        self.assertEqual(task.root_name, "openarm_right_base_link")
        self.assertEqual(task.root_type, "body")
        self.assertGreater(task.limit_activation, 0.0)
        self.assertIn("right", solver._nullspace_tasks)
        self.assertIn("right", solver._singularity_limits)
        self.assertIsNotNone(kinematics.solve())

    def test_solver_schedule_ignores_pure_target_rotation(self) -> None:
        setup = _setup("right")
        kinematics = Kinematics(
            setup,
            IKParams(
                frame_position_error_limit=0.003,
                frame_error_limit_linear_slow=0.2,
                frame_error_limit_linear_fast=0.5,
                dt=0.004,
                max_iters=5,
            ),
        )
        pose = setup.read_ee_pose("right").astype(np.float64)
        kinematics.set_target("right", pose)
        rotated = pose.copy()
        rotated[3:] = np.array([0.0, 1.0, 0.0, 0.0])
        kinematics.set_target("right", rotated)

        solver = kinematics._ik
        assert solver is not None
        task = solver._tasks["right"]
        self.assertIsInstance(task, ErrorLimitedFrameTask)
        assert isinstance(task, ErrorLimitedFrameTask)
        self.assertEqual(task.limit_activation, 0.0)

    def test_solver_schedule_rises_on_fast_target_translation(self) -> None:
        setup = _setup("right")
        kinematics = Kinematics(
            setup,
            IKParams(
                frame_position_error_limit=0.003,
                frame_error_limit_linear_slow=0.2,
                frame_error_limit_linear_fast=0.5,
                frame_error_limit_activation_rise_rate=4.0,
                dt=0.004,
                max_iters=5,
            ),
        )
        pose = setup.read_ee_pose("right").astype(np.float64)
        kinematics.set_target("right", pose)
        translated = pose.copy()
        translated[0] += 0.004
        kinematics.set_target("right", translated)

        solver = kinematics._ik
        assert solver is not None
        task = solver._tasks["right"]
        assert isinstance(task, ErrorLimitedFrameTask)
        self.assertAlmostEqual(task.limit_activation, 4.0 * 0.004)


class JointBrakingLimitTest(unittest.TestCase):
    """Exercise bidirectional braking for every active arm joint."""

    def _limit_and_configuration(
        self,
        *,
        reaction_time: float = 0.0,
        distance_buffer: float = 0.0,
    ) -> tuple[JointBrakingLimit, mink.Configuration]:
        setup = _setup("right")
        qpos_indices = _arm_qpos_indices(setup, "right")
        velocities = {
            f"openarm_right_joint{i + 1}": value
            for i, value in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
        }
        limit = JointBrakingLimit(
            setup.model,
            qpos_indices,
            velocities,
            slowdown_distance=0.4,
            exponent=2.0,
            reaction_time=reaction_time,
            distance_buffer=distance_buffer,
        )
        configuration = mink.Configuration(
            setup.model,
            q=setup.data.qpos.copy(),
        )
        return limit, configuration

    def test_half_slowdown_distance_allows_quarter_velocity(self) -> None:
        limit, configuration = self._limit_and_configuration()
        index = limit.joint_names.index("openarm_right_joint1")
        q = configuration.q
        q[limit.qpos_indices[index]] = limit.lower_guard[index] + 0.2
        configuration.update(q=q)

        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None
        count = len(limit.joint_names)
        lower_displacement = constraint.h[count + index]

        self.assertAlmostEqual(
            lower_displacement,
            0.25 * limit.max_velocity[index] * 0.01,
        )

    def test_upper_guard_uses_the_same_envelope_and_leaves_escape_free(self) -> None:
        limit, configuration = self._limit_and_configuration()
        index = limit.joint_names.index("openarm_right_joint1")
        q = configuration.q
        q[limit.qpos_indices[index]] = limit.upper_guard[index] - 0.2
        configuration.update(q=q)

        constraint = limit.compute_qp_inequalities(configuration, dt=0.01)
        assert constraint.h is not None
        count = len(limit.joint_names)
        upper_displacement = constraint.h[index]
        lower_displacement = constraint.h[count + index]

        self.assertAlmostEqual(
            upper_displacement,
            0.25 * limit.max_velocity[index] * 0.01,
        )
        self.assertAlmostEqual(
            lower_displacement,
            limit.max_velocity[index] * 0.01,
        )

    def test_measured_motion_reduces_effective_guard_distance(self) -> None:
        limit, configuration = self._limit_and_configuration(
            reaction_time=0.05,
            distance_buffer=0.01,
        )
        index = limit.joint_names.index("openarm_right_joint1")
        qpos_index = limit.qpos_indices[index]
        dof_index = limit.dof_indices[index]
        q = configuration.q
        q[qpos_index] = limit.lower_guard[index] + 0.3
        configuration.update(q=q)

        measured_q = configuration.q
        measured_q[qpos_index] = limit.lower_guard[index] + 0.1
        measured_dq = np.zeros(configuration.model.nv)
        measured_dq[dof_index] = -1.0
        limit.update_measured_state(measured_q, measured_dq)
        limit.compute_qp_inequalities(configuration, dt=0.01)

        state = limit.last_state
        self.assertIsNotNone(state)
        assert state is not None
        self.assertAlmostEqual(state.lower_distance[index], 0.04)
        self.assertLess(
            state.lower_approach_velocity[index],
            0.01 * limit.max_velocity[index],
        )

    def test_solver_builds_one_limit_for_every_active_arm_joint(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                posture_cost=0.0,
                nullspace_cost=0.0,
                joint_limit_braking=True,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        limit = solver._joint_braking_limit
        self.assertIsNotNone(limit)
        assert limit is not None
        self.assertEqual(len(limit.joint_names), 14)
        for side in ("left", "right"):
            index = limit.joint_names.index(f"openarm_{side}_joint4")
            joint_id = mujoco.mj_name2id(
                setup.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"openarm_{side}_joint4",
            )
            self.assertAlmostEqual(
                limit.lower_guard[index],
                setup.model.jnt_range[joint_id, 0],
            )

    def test_measured_state_does_not_sync_the_command_configuration(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                posture_cost=0.0,
                nullspace_cost=0.0,
                joint_limit_braking=True,
                singularity_approach_limit=True,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        command_before = solver._config.q.copy()
        right, right_gripper = setup.joint_resolver.get_driver(
            setup.data.qpos,
            "right",
        )
        left, left_gripper = setup.joint_resolver.get_driver(
            setup.data.qpos,
            "left",
        )
        measured_qpos = np.concatenate(
            [
                np.append(right, right_gripper),
                np.append(left, left_gripper),
            ]
        )
        measured_qpos[0] += 0.1

        kinematics.update_measured_state(
            measured_qpos,
            np.zeros(16),
            timestamp=10.0,
        )

        np.testing.assert_array_equal(solver._config.q, command_before)
        self.assertIsNotNone(solver._joint_braking_limit)
        assert solver._joint_braking_limit is not None
        self.assertIsNotNone(solver._joint_braking_limit._measured_qpos)


class SingularityApproachLimitTest(unittest.TestCase):
    """Exercise the one-sided singularity-ratio decrease constraint."""

    def _limit_and_configuration(
        self,
    ) -> tuple[
        ArmSetup,
        SingularityApproachLimit,
        mink.Configuration,
    ]:
        setup = _setup("right")
        configuration = mink.Configuration(
            setup.model,
            q=setup.data.qpos.copy(),
        )
        frame_task = mink.FrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
        )
        frame_task.set_target_from_configuration(configuration)
        limit = SingularityApproachLimit(
            setup.model,
            frame_task,
            _dof_indices_for_qpos(
                setup.model,
                _arm_qpos_indices(setup, "right"),
            ),
            characteristic_length=0.3,
            ratio_stop=0.01,
            ratio_slow=0.05,
            max_approach_rate=0.5,
            exponent=2.0,
        )
        return setup, limit, configuration

    def test_constraint_only_bounds_the_approaching_gradient_component(self) -> None:
        _, limit, configuration = self._limit_and_configuration()
        limit.prepare(configuration)
        constraint = limit.compute_qp_inequalities(configuration, dt=0.004)
        state = limit.last_state
        self.assertIsNotNone(state)
        assert state is not None
        assert constraint.G is not None
        assert constraint.h is not None

        approach = np.zeros(configuration.model.nv)
        approach[limit.dof_indices] = -state.gradient
        away = -approach
        self.assertGreater(float((constraint.G @ approach)[0]), 0.0)
        self.assertLess(float((constraint.G @ away)[0]), 0.0)
        self.assertAlmostEqual(
            float(constraint.h[0]),
            state.max_approach_rate * 0.004,
        )

    def test_measured_straight_arm_closes_the_approach_envelope(self) -> None:
        setup, limit, configuration = self._limit_and_configuration()
        measured = configuration.q
        setup.joint_resolver.set_qpos(
            measured,
            np.array([1.224145, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "right",
        )
        limit.update_measured_configuration(measured)
        limit.prepare(configuration)

        state = limit.last_state
        self.assertIsNotNone(state)
        assert state is not None
        self.assertIsNotNone(state.measured_ratio)
        self.assertLessEqual(state.effective_ratio, state.command_ratio)
        self.assertEqual(state.max_approach_rate, 0.0)

    def test_singularity_ratio_does_not_depend_on_frame_task_target(self) -> None:
        setup, targeted_limit, configuration = self._limit_and_configuration()
        untargeted_task = mink.FrameTask(
            frame_name="right_ee_control_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
        )
        untargeted_limit = SingularityApproachLimit(
            setup.model,
            untargeted_task,
            _dof_indices_for_qpos(
                setup.model,
                _arm_qpos_indices(setup, "right"),
            ),
            characteristic_length=0.3,
            ratio_stop=0.01,
            ratio_slow=0.05,
            max_approach_rate=0.5,
        )

        targeted_limit.prepare(configuration)
        untargeted_limit.prepare(configuration)

        assert targeted_limit.last_state is not None
        assert untargeted_limit.last_state is not None
        self.assertAlmostEqual(
            targeted_limit.last_state.command_ratio,
            untargeted_limit.last_state.command_ratio,
        )
        np.testing.assert_allclose(
            targeted_limit.last_state.gradient,
            untargeted_limit.last_state.gradient,
            atol=1e-12,
        )


class SolverTimingTest(unittest.TestCase):
    """Verify that ten substeps represent exactly one outer control period."""

    def test_cli_uses_tick_period_and_unscaled_velocity_limits(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        args = parser.parse_args(
            [
                "--tick-hz",
                "250",
                "--limit-velocity",
                "--frame-position-error-limit",
                "0.0003",
                "--frame-orientation-error-limit",
                "0.0048",
                "--frame-error-limit-linear-slow",
                "0.25",
                "--frame-error-limit-linear-fast",
                "0.55",
                "--frame-error-limit-activation-rise-rate",
                "5.0",
                "--frame-error-limit-activation-fall-rate",
                "1.5",
                "--joint-limit-recovery-velocity-scale",
                "1.2",
                "--joint-limit-braking",
                "--joint-limit-braking-slowdown-distance",
                "0.5",
                "--joint-limit-braking-exponent",
                "2",
                "--joint-limit-braking-reaction-time",
                "0.04",
                "--joint-limit-braking-distance-buffer",
                "0.01",
                "--singularity-approach-limit",
                "--singularity-ratio-stop",
                "0.02",
                "--singularity-ratio-slow",
                "0.08",
                "--singularity-max-approach-rate",
                "0.25",
                "--singularity-braking-exponent",
                "2",
                "--measured-state-timeout",
                "0.1",
                "--kinetic-energy-cost",
                "3e-5",
            ]
        )

        params = ik_params_from_args(args)

        self.assertAlmostEqual(params.dt, 0.004)
        self.assertEqual(params.frame_position_error_limit, 0.0003)
        self.assertEqual(params.frame_orientation_error_limit, 0.0048)
        self.assertEqual(params.frame_error_limit_linear_slow, 0.25)
        self.assertEqual(params.frame_error_limit_linear_fast, 0.55)
        self.assertEqual(params.frame_error_limit_activation_rise_rate, 5.0)
        self.assertEqual(params.frame_error_limit_activation_fall_rate, 1.5)
        self.assertEqual(params.joint_limit_recovery_velocity_scale, 1.2)
        self.assertTrue(params.joint_limit_braking)
        self.assertEqual(params.joint_limit_braking_slowdown_distance, 0.5)
        self.assertEqual(params.joint_limit_braking_exponent, 2.0)
        self.assertEqual(params.joint_limit_braking_reaction_time, 0.04)
        self.assertEqual(params.joint_limit_braking_distance_buffer, 0.01)
        self.assertTrue(params.singularity_approach_limit)
        self.assertEqual(params.singularity_ratio_stop, 0.02)
        self.assertEqual(params.singularity_ratio_slow, 0.08)
        self.assertEqual(params.singularity_max_approach_rate, 0.25)
        self.assertEqual(params.singularity_braking_exponent, 2.0)
        self.assertEqual(params.measured_state_timeout, 0.1)
        self.assertEqual(params.kinetic_energy_cost, 3e-5)
        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_kinetic_energy_task_uses_current_mujoco_inertia_api(self) -> None:
        setup = _setup("right")
        kinematics = Kinematics(
            setup,
            IKParams(
                posture_cost=0.0,
                nullspace_cost=0.0,
                kinetic_energy_cost=1e-7,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        task = solver._kinetic_energy_task
        self.assertIsNotNone(task)
        assert task is not None

        objective = task.compute_qp_objective(solver._config)

        self.assertEqual(objective.H.shape, (setup.model.nv, setup.model.nv))
        self.assertTrue(np.all(np.isfinite(objective.H)))
        np.testing.assert_allclose(objective.H, objective.H.T, atol=1e-12)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(objective.H))), -1e-9)
        np.testing.assert_array_equal(objective.c, np.zeros(setup.model.nv))

    def test_custom_arm_velocity_limits_override_defaults(self) -> None:
        custom_caps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "ik.yaml"
            config_path.write_text(
                "arm_velocity_limits:\n"
                + "".join(f"  - {cap}\n" for cap in custom_caps),
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            register_ik_args(parser)
            args = parser.parse_args(["--limit-velocity", "--config", str(config_path)])

            params = ik_params_from_args(args)

        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(custom_caps):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_driver_delta_limits_are_not_accepted_as_ik_velocity_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "driver.yaml"
            config_path.write_text(
                "joint_delta_position_limits: [1, 1, 1, 1, 1, 1, 1, 1]\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            register_ik_args(parser)
            args = parser.parse_args(["--limit-velocity", "--config", str(config_path)])

            with self.assertRaisesRegex(ValueError, "arm_velocity_limits"):
                ik_params_from_args(args)

    def test_substep_dt_and_velocity_limits_stay_in_physical_units(self) -> None:
        setup = _setup()
        per_joint_caps = np.linspace(0.2, 0.8, 7)
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(cap)
            for side in ("left", "right")
            for index, cap in enumerate(per_joint_caps)
        }
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                posture_cost=0.0,
                nullspace_cost=0.0,
                velocity_limits=velocity_limits,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        self.assertAlmostEqual(solver._substep_dt, 0.0004)

        velocity_limit = solver._limits[-1]
        self.assertIsInstance(velocity_limit, RecoverableConfigurationLimit)
        self.assertEqual(len(solver._limits), 1)
        np.testing.assert_allclose(
            np.sort(velocity_limit.limit),
            np.sort(np.tile(per_joint_caps, 2)),
        )

        velocity = np.zeros(setup.model.nv)
        for side in ("right", "left"):
            dof_indices = _dof_indices_for_qpos(
                setup.model, _arm_qpos_indices(setup, side)
            )
            velocity[dof_indices] = per_joint_caps
        qpos_before = solver._config.q

        kinematics.set_target("right", setup.read_ee_pose("right"))
        kinematics.set_target("left", setup.read_ee_pose("left"))
        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=velocity,
        ) as solve_ik:
            result = kinematics.solve()

        self.assertIsNotNone(result)
        self.assertEqual(solve_ik.call_count, 10)
        for call in solve_ik.call_args_list:
            self.assertAlmostEqual(call.args[2], 0.0004)

        expected_delta = velocity * 0.004
        np.testing.assert_allclose(
            solver._config.q - qpos_before,
            expected_delta,
            atol=1e-12,
        )


class RecoverableConfigurationLimitTest(unittest.TestCase):
    """Exercise the joint position/velocity intersection around overshoot."""

    def _solver_and_limit(
        self,
    ) -> tuple[Kinematics, RecoverableConfigurationLimit]:
        setup = _setup()
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(cap)
            for side in ("left", "right")
            for index, cap in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
        }
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=10,
                posture_cost=0.0,
                nullspace_cost=0.0,
                velocity_limits=velocity_limits,
                joint_limit_recovery_velocity_scale=1.1,
            ),
        )
        solver = kinematics._ik
        assert solver is not None
        self.assertEqual(len(solver._limits), 1)
        limit = solver._limits[0]
        self.assertIsInstance(limit, RecoverableConfigurationLimit)
        assert isinstance(limit, RecoverableConfigurationLimit)
        return kinematics, limit

    def test_inside_range_uses_normal_velocity_cap(self) -> None:
        kinematics, limit = self._solver_and_limit()
        solver = kinematics._ik
        assert solver is not None
        row = 0
        q = solver._config.q
        q[limit.qpos_indices[row]] = 0.5 * (limit.lower[row] + limit.upper[row])
        solver._config.update(q=q)

        inequalities = limit.compute_qp_inequalities(solver._config, solver._substep_dt)
        assert inequalities.h is not None
        count = limit.indices.size
        step_lower = -inequalities.h[count:]
        step_upper = inequalities.h[:count]
        normal_step = limit.limit[row] * solver._substep_dt

        self.assertAlmostEqual(step_lower[row], -normal_step)
        self.assertAlmostEqual(step_upper[row], normal_step)

    def test_outside_range_recovers_with_wider_feasible_cap(self) -> None:
        kinematics, limit = self._solver_and_limit()
        solver = kinematics._ik
        assert solver is not None
        row = 0
        qpos_index = limit.qpos_indices[row]
        recovery_step = (
            limit.limit[row] * solver._substep_dt * limit.recovery_velocity_scale
        )

        for q_value, expected_step in (
            (limit.lower[row] - 0.01, recovery_step),
            (limit.upper[row] + 0.01, -recovery_step),
        ):
            q = solver._config.q
            q[qpos_index] = q_value
            solver._config.update(q=q)
            inequalities = limit.compute_qp_inequalities(
                solver._config, solver._substep_dt
            )
            assert inequalities.h is not None
            count = limit.indices.size
            step_lower = -inequalities.h[count:]
            step_upper = inequalities.h[:count]

            self.assertLessEqual(step_lower[row], step_upper[row])
            self.assertAlmostEqual(step_lower[row], expected_step)
            self.assertAlmostEqual(step_upper[row], expected_step)

    def test_real_qp_recovers_from_joint_limit_overshoot(self) -> None:
        kinematics, limit = self._solver_and_limit()
        solver = kinematics._ik
        assert solver is not None
        row = 0
        qpos_index = limit.qpos_indices[row]
        q = solver._config.q
        q[qpos_index] = limit.lower[row] - 0.001
        solver._config.update(q=q)
        initial_q = float(solver._config.q[qpos_index])

        for side in kinematics.setup.sides:
            pose = read_ee_pose(
                solver._config.data,
                kinematics.setup.frame_ids[side],
                kinematics.setup.frame_types[side],
            )
            kinematics.set_target(side, pose)

        result = kinematics.solve()

        self.assertIsNotNone(result)
        self.assertGreater(float(solver._config.q[qpos_index]), initial_q)


class SolverRecoveryTest(unittest.TestCase):
    """Verify that failed solves do not corrupt bimanual solver state."""

    def test_failure_rolls_back_and_waits_for_fresh_target_pair(self) -> None:
        setup = _setup()
        kinematics = Kinematics(
            setup,
            IKParams(
                dt=0.004,
                max_iters=3,
                posture_cost=0.0,
                nullspace_cost=0.0,
            ),
        )
        solver = kinematics._ik
        assert solver is not None

        target_right = setup.read_ee_pose("right")
        target_left = setup.read_ee_pose("left")
        kinematics.set_target("right", target_right)
        kinematics.set_target("left", target_left)
        self.assertTrue(kinematics.ready())

        velocity = np.zeros(setup.model.nv)
        right_dofs = _dof_indices_for_qpos(
            setup.model, _arm_qpos_indices(setup, "right")
        )
        velocity[right_dofs[0]] = 0.5
        qpos_before = solver._config.q.copy()

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            side_effect=[velocity, mink.exceptions.NoSolutionFound("daqp")],
        ) as solve_ik:
            result = kinematics.solve()

        self.assertIsNone(result)
        self.assertEqual(solve_ik.call_count, 2)
        np.testing.assert_array_equal(solver._config.q, qpos_before)
        self.assertFalse(kinematics.ready())

        kinematics.set_target("right", target_right)
        self.assertFalse(kinematics.ready())
        kinematics.set_target("left", target_left)
        self.assertTrue(kinematics.ready())

        with mock.patch(
            "openarm_control.kinematics.mink.solve_ik",
            return_value=np.zeros(setup.model.nv),
        ):
            recovered = kinematics.solve()

        self.assertIsNotNone(recovered)
        self.assertFalse(kinematics.ready())


if __name__ == "__main__":
    unittest.main()
