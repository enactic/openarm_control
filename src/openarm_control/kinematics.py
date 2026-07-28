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

"""High-level FK + IK interface for OpenArm.

Poses are float32[7] = [px, py, pz, qw, qx, qy, qz], expressed in the
setup's origin frame (default: the scene's 'arm_origin' site; pass
--origin-frame world for world coordinates). FK returns poses in that
frame and IK targets are interpreted in it.

Usage:
    # FK only
    kin = Kinematics(setup)
    pose = kin.fk("right", joints)          # float32[7]
    pose_r, pose_l = kin.fk_bimanual(r, l)  # single mj_forward

    # FK + IK
    kin = Kinematics(setup, IKParams(dt=1.0 / 250.0, max_iters=10))
    kin.set_target("right", pose)
    kin.set_target("left", pose)
    result = kin.solve()                    # float32[16] or None
"""

from __future__ import annotations

import argparse
import pathlib
import time
from dataclasses import dataclass

import mink
import mink.exceptions
import mujoco
import numpy as np
import yaml

from openarm_control.config import (
    ARM_JOINT_VELOCITY_LIMITS_RAD_S,
    ArmSetup,
    frame_name,
)
from openarm_control.error_limited_frame_task import (
    ErrorLimitedFrameTask,
    ErrorLimitedRelativeFrameTask,
)
from openarm_control.joint_braking_limit import JointBrakingLimit
from openarm_control.kinetic_energy_task import (
    KineticEnergyRegularizationTask,
)
from openarm_control.nullspace_posture_task import NullspacePostureTask
from openarm_control.poses import pose_to_se3
from openarm_control.recoverable_configuration_limit import (
    RecoverableConfigurationLimit,
)
from openarm_control.singularity_approach_limit import SingularityApproachLimit


@dataclass
class IKParams:
    """Configuration for the mink QP-based IK solver."""

    position_cost: float = 1.0
    orientation_cost: float = 1.0
    frame_position_error_limit: float = 0.0
    frame_orientation_error_limit: float = 0.0
    frame_error_limit_linear_slow: float = 0.2
    frame_error_limit_linear_fast: float = 0.5
    frame_error_limit_activation_rise_rate: float = 4.0
    frame_error_limit_activation_fall_rate: float = 2.0
    lm_damping: float = 0.01
    damping: float = 0.25
    solver: str = "daqp"
    posture_cost: float = 0.01
    diag_reg: float = 0.0
    dt: float = 1.0 / 250.0
    max_iters: int = 10
    velocity_limits: dict[str, float] | None = None
    joint_limit_recovery_velocity_scale: float = 1.1
    nullspace_cost: float = 0.3
    nullspace_return_rate: float = 0.5
    nullspace_max_speed: float = 0.5
    nullspace_singularity_low: float = 0.02
    nullspace_singularity_high: float = 0.05
    nullspace_characteristic_length: float = 0.3
    joint_limit_braking: bool = False
    joint_limit_braking_slowdown_distance: float = 0.5
    joint_limit_braking_exponent: float = 2.0
    joint_limit_braking_guard_margin: float = 0.0
    joint_limit_braking_reaction_time: float = 0.0
    joint_limit_braking_distance_buffer: float = 0.0
    singularity_approach_limit: bool = False
    singularity_ratio_stop: float = 0.01
    singularity_ratio_slow: float = 0.05
    singularity_max_approach_rate: float = 0.5
    singularity_braking_exponent: float = 2.0
    singularity_gradient_epsilon: float = 1e-4
    measured_state_timeout: float = 0.1
    kinetic_energy_cost: float = 0.0


class Kinematics:
    """Unified FK + IK for OpenArm, backed by MuJoCo + mink.

    FK is always available. IK is enabled by passing ``IKParams``.
    Both share the same ``ArmSetup`` context (model, resolver, frame IDs).
    """

    def __init__(self, setup: ArmSetup, ik_params: IKParams | None = None) -> None:
        """Initialize."""
        self.setup = setup
        self._ik: _IKSolver | None = (
            _IKSolver(setup, ik_params) if ik_params is not None else None
        )

    # ── FK ───────────────────────────────────────────────────────────────────

    def fk(self, side: str, joints: np.ndarray) -> np.ndarray:
        """Set qpos for one arm, run mj_forward, return float32[7] EE pose."""
        self.setup.joint_resolver.set_qpos(self.setup.data.qpos, joints, side)
        mujoco.mj_forward(self.setup.model, self.setup.data)
        return self.setup.read_ee_pose(side)

    def fk_bimanual(
        self, right: np.ndarray, left: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set both arms and run a single mj_forward. Returns (pose_right, pose_left)."""
        self.setup.joint_resolver.set_qpos(self.setup.data.qpos, right, "right")
        self.setup.joint_resolver.set_qpos(self.setup.data.qpos, left, "left")
        mujoco.mj_forward(self.setup.model, self.setup.data)
        return self.setup.read_ee_pose("right"), self.setup.read_ee_pose("left")

    # ── IK ───────────────────────────────────────────────────────────────────

    def set_target(self, side: str, pose: np.ndarray) -> None:
        """Set EE target for one arm, in the origin frame.

        pose: float32[7] = [px, py, pz, qw, qx, qy, qz].
        """
        self._require_ik().set_target(side, pose)

    def sync(self, values16: np.ndarray) -> None:
        """Sync IK internal config from float32[16] driver state (right[8]+left[8])."""
        self._require_ik().sync(values16)

    def update_measured_state(
        self,
        qpos16: np.ndarray,
        qvel16: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Update measured q/dq used by safety limits without syncing IK state."""
        self._require_ik().update_measured_state(
            qpos16,
            qvel16,
            timestamp=timestamp,
        )

    def clear_measured_state(self) -> None:
        """Make state-aware limits fall back to command configuration only."""
        self._require_ik().clear_measured_state()

    def ready(self) -> bool:
        """Return True once all active arms have received at least one target this cycle."""
        return self._require_ik().ready()

    def solve(self) -> np.ndarray | None:
        """Run IK. Returns float32[16] (right[8]+left[8]) or None on failure."""
        return self._require_ik().solve()

    def set_gripper(self, side: str, value: float) -> None:
        """Pass through a gripper value; IK does not solve for it."""
        idx = 0 if side == "right" else 1
        self._require_ik()._gripper[idx] = value

    def _require_ik(self) -> _IKSolver:
        if self._ik is None:
            raise RuntimeError("Kinematics was not initialized with IKParams.")
        return self._ik


# ── internal IK implementation ────────────────────────────────────────────────


class _IKSolver:
    """mink QP-based differential IK. Managed by Kinematics; not public API."""

    def __init__(self, setup: ArmSetup, params: IKParams) -> None:
        if params.dt <= 0.0:
            raise ValueError("IK control timestep must be positive.")
        if params.max_iters <= 0:
            raise ValueError("IK max_iters must be positive.")

        self._sides = setup.sides
        self._solver_name = params.solver
        self._posture_cost = params.posture_cost
        self._model = setup.model
        self._joint_resolver = setup.joint_resolver
        self._arm_qpos_by_side = {
            side: _arm_qpos_indices(setup, side) for side in setup.sides
        }
        self._arm_dofs_by_side = {
            side: _dof_indices_for_qpos(
                setup.model,
                self._arm_qpos_by_side[side],
            )
            for side in setup.sides
        }
        self._control_dt = params.dt
        self._substep_dt = params.dt / params.max_iters
        self._max_iters = params.max_iters
        if not np.isfinite(params.measured_state_timeout) or (
            params.measured_state_timeout <= 0.0
        ):
            raise ValueError("Measured state timeout must be finite and positive.")
        self._measured_state_timeout = params.measured_state_timeout
        self._last_measured_state_time: float | None = None

        self._config = mink.Configuration(setup.model)
        self._config.update(q=setup.data.qpos.copy())
        home_qpos = self._config.data.qpos.copy()

        task_kwargs = {
            "position_cost": params.position_cost,
            "orientation_cost": params.orientation_cost,
            "lm_damping": params.lm_damping,
        }
        for name, value in (
            ("frame_position_error_limit", params.frame_position_error_limit),
            (
                "frame_orientation_error_limit",
                params.frame_orientation_error_limit,
            ),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        frame_error_limit_enabled = params.frame_position_error_limit > 0.0 or (
            params.frame_orientation_error_limit > 0.0
        )
        if frame_error_limit_enabled:
            if (
                not 0.0
                <= params.frame_error_limit_linear_slow
                < params.frame_error_limit_linear_fast
            ):
                raise ValueError(
                    "Expected 0 <= frame_error_limit_linear_slow < "
                    "frame_error_limit_linear_fast."
                )
            if params.frame_error_limit_activation_rise_rate <= 0.0 or (
                params.frame_error_limit_activation_fall_rate <= 0.0
            ):
                raise ValueError("Frame error-limit activation rates must be positive.")
        self._frame_error_limit_linear_slow = params.frame_error_limit_linear_slow
        self._frame_error_limit_linear_fast = params.frame_error_limit_linear_fast
        self._frame_error_limit_activation_rise_rate = (
            params.frame_error_limit_activation_rise_rate
        )
        self._frame_error_limit_activation_fall_rate = (
            params.frame_error_limit_activation_fall_rate
        )
        self._tasks: dict[str, mink.FrameTask | mink.RelativeFrameTask] = {}
        for side in setup.sides:
            frame_kwargs = {
                "frame_name": _frame_name(setup, side),
                "frame_type": setup.frame_types[side],
                **task_kwargs,
            }
            if setup.origin_id is not None:
                frame_kwargs.update(
                    root_name=frame_name(
                        setup.model,
                        setup.origin_id,
                        setup.origin_type,
                    ),
                    root_type=setup.origin_type,
                )
                task_type = (
                    ErrorLimitedRelativeFrameTask
                    if frame_error_limit_enabled
                    else mink.RelativeFrameTask
                )
            else:
                task_type = (
                    ErrorLimitedFrameTask
                    if frame_error_limit_enabled
                    else mink.FrameTask
                )
            if frame_error_limit_enabled:
                frame_kwargs.update(
                    position_error_limit=params.frame_position_error_limit,
                    orientation_error_limit=params.frame_orientation_error_limit,
                )
            self._tasks[side] = task_type(**frame_kwargs)

        self._frame_error_limit_activation = {side: 0.0 for side in setup.sides}
        self._frame_error_limit_previous_target_position: dict[
            str,
            np.ndarray,
        ] = {}
        for task in self._tasks.values():
            if isinstance(
                task,
                (ErrorLimitedFrameTask, ErrorLimitedRelativeFrameTask),
            ):
                task.set_limit_activation(0.0)

        active_qpos = {
            int(qpos_index)
            for side in self._sides
            for qpos_index in self._arm_qpos_by_side[side]
        }
        freeze_dofs = [
            int(setup.model.jnt_dofadr[j])
            for j in range(setup.model.njnt)
            if setup.model.jnt_qposadr[j] not in active_qpos
        ]
        self._freeze_task: mink.DofFreezingTask | None = (
            mink.DofFreezingTask(model=setup.model, dof_indices=freeze_dofs)
            if freeze_dofs
            else None
        )

        # Only constrain the DoFs that IK can move. Frozen DoFs such as the
        # gripper and lifter may be valid in driver space but outside the MuJoCo
        # model range; constraining and freezing them at the same time makes the
        # QP infeasible.
        if params.velocity_limits is not None:
            self._limits = [
                RecoverableConfigurationLimit(
                    model=setup.model,
                    qpos_indices=active_qpos,
                    velocities=params.velocity_limits,
                    recovery_velocity_scale=(
                        params.joint_limit_recovery_velocity_scale
                    ),
                )
            ]
        else:
            self._limits = [_configuration_limit_for_qpos(setup.model, active_qpos)]

        arm_velocity_limits = _arm_velocity_limit_mapping(
            setup,
            params.velocity_limits,
        )
        self._joint_braking_limit: JointBrakingLimit | None = None
        if params.joint_limit_braking:
            self._joint_braking_limit = JointBrakingLimit(
                model=setup.model,
                qpos_indices=active_qpos,
                velocities=arm_velocity_limits,
                slowdown_distance=params.joint_limit_braking_slowdown_distance,
                exponent=params.joint_limit_braking_exponent,
                guard_margin=params.joint_limit_braking_guard_margin,
                reaction_time=params.joint_limit_braking_reaction_time,
                distance_buffer=params.joint_limit_braking_distance_buffer,
            )
            self._limits.append(self._joint_braking_limit)

        self._singularity_limits: dict[str, SingularityApproachLimit] = {}
        if params.singularity_approach_limit:
            for side in setup.sides:
                singularity_limit = SingularityApproachLimit(
                    model=setup.model,
                    frame_task=self._tasks[side],
                    dof_indices=_dof_indices_for_qpos(
                        setup.model,
                        _arm_qpos_indices(setup, side),
                    ),
                    characteristic_length=params.nullspace_characteristic_length,
                    ratio_stop=params.singularity_ratio_stop,
                    ratio_slow=params.singularity_ratio_slow,
                    max_approach_rate=params.singularity_max_approach_rate,
                    exponent=params.singularity_braking_exponent,
                    gradient_epsilon=params.singularity_gradient_epsilon,
                )
                self._singularity_limits[side] = singularity_limit
                self._limits.append(singularity_limit)

        self._posture_task = mink.PostureTask(setup.model, cost=params.posture_cost)
        self._posture_task.set_target(home_qpos)

        self._kinetic_energy_task: KineticEnergyRegularizationTask | None = None
        if params.kinetic_energy_cost < 0.0:
            raise ValueError("Kinetic energy cost must be non-negative.")
        if params.kinetic_energy_cost > 0.0:
            self._kinetic_energy_task = KineticEnergyRegularizationTask(
                cost=params.kinetic_energy_cost
            )
            self._kinetic_energy_task.set_dt(self._substep_dt)

        self._nullspace_tasks: dict[str, NullspacePostureTask] = {}
        if params.nullspace_cost > 0.0:
            for side in setup.sides:
                arm_qpos = _arm_qpos_indices(setup, side)
                self._nullspace_tasks[side] = NullspacePostureTask(
                    model=setup.model,
                    frame_task=self._tasks[side],
                    dof_indices=_dof_indices_for_qpos(setup.model, arm_qpos),
                    home_qpos=home_qpos,
                    cost=params.nullspace_cost,
                    dt=self._substep_dt,
                    return_rate=params.nullspace_return_rate,
                    max_speed=params.nullspace_max_speed,
                    singularity_low=params.nullspace_singularity_low,
                    singularity_high=params.nullspace_singularity_high,
                    characteristic_length=params.nullspace_characteristic_length,
                )

        self._solver_params: dict = {"damping": params.damping}
        if params.diag_reg > 0.0:
            self._solver_params["diag_reg"] = params.diag_reg

        self._pending: set[str] = set(setup.sides)
        self._gripper = np.zeros(2, dtype=np.float32)

    def set_target(self, side: str, pose: np.ndarray) -> None:
        target_pose = np.asarray(pose, dtype=np.float64)
        self._tasks[side].set_target(pose_to_se3(target_pose))
        self._update_frame_error_limit_schedule(side, target_pose)
        self._pending.discard(side)

    def _update_frame_error_limit_schedule(
        self,
        side: str,
        target_pose: np.ndarray,
    ) -> None:
        """Activate bounded task error from desired translational speed only."""
        task = self._tasks[side]
        if not isinstance(
            task,
            (ErrorLimitedFrameTask, ErrorLimitedRelativeFrameTask),
        ):
            return

        target_position = target_pose[:3]
        previous = self._frame_error_limit_previous_target_position.get(side)
        linear_speed = (
            0.0
            if previous is None
            else float(np.linalg.norm(target_position - previous)) / self._control_dt
        )
        self._frame_error_limit_previous_target_position[side] = target_position.copy()

        unit_speed = np.clip(
            (linear_speed - self._frame_error_limit_linear_slow)
            / (
                self._frame_error_limit_linear_fast
                - self._frame_error_limit_linear_slow
            ),
            0.0,
            1.0,
        )
        target_activation = float(unit_speed * unit_speed * (3.0 - 2.0 * unit_speed))
        current_activation = self._frame_error_limit_activation[side]

        # Once translational motion has activated the limiter, do not release
        # accumulated lag as one large FrameTask request. The latch clears only
        # after the full positional error is close to the bounded request.
        if current_activation > 0.0 and task.position_error_limit > 0.0:
            full_error = task.compute_full_error(self._config)
            if float(np.linalg.norm(full_error[:3])) > (
                2.0 * task.position_error_limit
            ):
                target_activation = max(target_activation, current_activation)

        difference = target_activation - current_activation
        activation_rate = (
            self._frame_error_limit_activation_rise_rate
            if difference > 0.0
            else self._frame_error_limit_activation_fall_rate
        )
        current_activation += float(
            np.clip(
                difference,
                -activation_rate * self._control_dt,
                activation_rate * self._control_dt,
            )
        )
        self._frame_error_limit_activation[side] = current_activation
        task.set_limit_activation(current_activation)

    def sync(self, values16: np.ndarray) -> None:
        qpos = self._config.data.qpos.copy()
        self._joint_resolver.set_qpos(qpos, values16[:8], "right")
        self._joint_resolver.set_qpos(qpos, values16[8:16], "left")
        self._config.update(q=qpos)
        # Gripper is intentionally NOT synced here. set_gripper() is the sole
        # writer of self._gripper ("IK does not solve for it"); syncing it
        # from the raw driver state here would race with set_gripper() calls
        # (e.g. from a VR trigger) arriving on a similar cadence, causing the
        # commanded gripper to flicker between the real motor position and
        # the trigger-commanded value depending on event arrival order.

    def update_measured_state(
        self,
        qpos16: np.ndarray,
        qvel16: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Update state-aware limits without changing the IK configuration."""
        qpos16 = np.asarray(qpos16, dtype=np.float64)
        qvel16 = np.asarray(qvel16, dtype=np.float64)
        if qpos16.shape != (16,) or qvel16.shape != (16,):
            raise ValueError("Measured bimanual qpos and qvel must have shape (16,).")
        if not np.all(np.isfinite(qpos16)) or not np.all(np.isfinite(qvel16)):
            raise ValueError("Measured bimanual state must be finite.")
        measured_at = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(measured_at):
            raise ValueError("Measured state timestamp must be finite.")

        measured_qpos = self._config.q.copy()
        measured_qvel = np.zeros(self._model.nv, dtype=np.float64)
        offsets = {"right": 0, "left": 8}
        for side in self._sides:
            offset = offsets[side]
            self._joint_resolver.set_qpos(
                measured_qpos,
                qpos16[offset : offset + 8],
                side,
            )
            arm_dofs = self._arm_dofs_by_side[side]
            measured_qvel[arm_dofs] = qvel16[offset : offset + 7]

        if self._joint_braking_limit is not None:
            self._joint_braking_limit.update_measured_state(
                measured_qpos,
                measured_qvel,
            )
        for limit in self._singularity_limits.values():
            limit.update_measured_configuration(measured_qpos)
        self._last_measured_state_time = measured_at

    def clear_measured_state(self) -> None:
        """Clear measured state from every state-aware limit."""
        if self._joint_braking_limit is not None:
            self._joint_braking_limit.clear_measured_state()
        for limit in self._singularity_limits.values():
            limit.clear_measured_configuration()
        self._last_measured_state_time = None

    def ready(self) -> bool:
        return len(self._pending) == 0

    def solve(self) -> np.ndarray | None:
        self._expire_stale_measured_state()
        tasks = list(self._tasks.values())
        if self._posture_cost > 0.0:
            tasks.append(self._posture_task)
        if self._kinetic_energy_task is not None:
            tasks.append(self._kinetic_energy_task)
        tasks.extend(self._nullspace_tasks.values())
        constraints = [self._freeze_task] if self._freeze_task else []

        q_before = self._config.q.copy()
        for limit in self._singularity_limits.values():
            limit.prepare(self._config)
        # This solve attempt consumes the current target pair. Even on failure,
        # wait for a fresh target from every active side before trying again.
        self._pending = set(self._sides)

        for _ in range(self._max_iters):
            try:
                vel = mink.solve_ik(
                    self._config,
                    tasks,
                    self._substep_dt,
                    self._solver_name,
                    limits=self._limits,
                    constraints=constraints,
                    safety_break=False,
                    **self._solver_params,
                )
            except mink.exceptions.NoSolutionFound:
                # Earlier substeps may already have advanced the internal model,
                # while no command from this failed solve reaches the real arm.
                self._config.update(q=q_before)
                print("Warning: constrained IK solver failed. Skipping step.")
                return None
            self._config.integrate_inplace(vel, self._substep_dt)

        qpos = self._config.data.qpos
        right_joints, _ = self._joint_resolver.get_driver(qpos, "right")
        left_joints, _ = self._joint_resolver.get_driver(qpos, "left")
        return np.concatenate(
            [
                np.append(right_joints, self._gripper[0]),
                np.append(left_joints, self._gripper[1]),
            ]
        ).astype(np.float32)

    def _expire_stale_measured_state(self) -> None:
        if self._last_measured_state_time is None:
            return
        if time.monotonic() - self._last_measured_state_time > (
            self._measured_state_timeout
        ):
            self.clear_measured_state()


def _frame_name(setup: ArmSetup, side: str) -> str:
    return frame_name(setup.model, setup.frame_ids[side], setup.frame_types[side])


def _arm_qpos_indices(setup: ArmSetup, side: str) -> np.ndarray:
    resolved = (
        setup.joint_resolver._right if side == "right" else setup.joint_resolver._left
    )
    return np.asarray(resolved.arm_qpos, dtype=int)


def _dof_indices_for_qpos(
    model: mujoco.MjModel, qpos_indices: np.ndarray
) -> np.ndarray:
    """Map scalar arm-joint qpos addresses to tangent-space DoF addresses."""
    dof_indices: list[int] = []
    for qpos_index in qpos_indices:
        joint_ids = np.flatnonzero(model.jnt_qposadr == int(qpos_index))
        if joint_ids.size != 1:
            raise ValueError(
                f"Expected qpos index {qpos_index} to start exactly one joint."
            )
        joint_id = int(joint_ids[0])
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("IK arm joints must be scalar hinge or slide joints.")
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(dof_indices, dtype=int)


def _joint_velocity_cap(
    velocity_limits: dict[str, float] | None,
    joint_name: str,
    default: float,
) -> float:
    """Return a scalar configured joint velocity cap or its built-in default."""
    if velocity_limits is None:
        return default
    value = np.asarray(velocity_limits[joint_name], dtype=np.float64)
    if value.size != 1:
        raise ValueError(f"Velocity limit for {joint_name!r} must be scalar.")
    return float(value.reshape(-1)[0])


def _arm_velocity_limit_mapping(
    setup: ArmSetup,
    velocity_limits: dict[str, float] | None,
) -> dict[str, float]:
    """Return configured or built-in scalar velocity caps for active arms."""
    return {
        f"openarm_{side}_joint{index + 1}": _joint_velocity_cap(
            velocity_limits,
            f"openarm_{side}_joint{index + 1}",
            default,
        )
        for side in setup.sides
        for index, default in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    }


def _configuration_limit_for_qpos(
    model: mujoco.MjModel, qpos_indices: set[int]
) -> mink.ConfigurationLimit:
    limit = mink.ConfigurationLimit(model)
    active_qpos = {int(index) for index in qpos_indices}
    active_dofs = [
        int(model.jnt_dofadr[j])
        for j in range(model.njnt)
        if model.jnt_limited[j] and int(model.jnt_qposadr[j]) in active_qpos
    ]
    indices = np.asarray(active_dofs, dtype=int)
    indices.setflags(write=False)
    limit.indices = indices
    limit.projection_matrix = np.eye(model.nv)[indices] if indices.size else None
    return limit


def _load_velocity_caps(config_path: pathlib.Path | None) -> list[float]:
    """Return per-joint velocity caps in rad/s.

    With no config path, returns the built-in ARM_JOINT_VELOCITY_LIMITS_RAD_S.
    When a path is given, reads the legacy IK-specific
    'arm_velocity_limits' key from the YAML.
    """
    if config_path is None:
        return ARM_JOINT_VELOCITY_LIMITS_RAD_S

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config file {config_path} must contain a YAML mapping.")

    key = "arm_velocity_limits"
    if key not in data:
        raise ValueError(f"Config file {config_path} has no top-level '{key}' list.")

    expected = len(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    raw_caps = data[key]
    caps = [float(v) for v in raw_caps]
    if len(caps) != expected:
        raise ValueError(
            f"{key} in {config_path} has {len(caps)} arm entries; expected {expected}."
        )
    return caps


# ── CLI helpers ───────────────────────────────────────────────────────────────


def register_ik_args(parser: argparse.ArgumentParser) -> None:
    """Register IK-specific CLI flags. Call after register_common_args."""
    parser.add_argument(
        "--pos-cost", type=float, default=1.0, help="Position task cost (default: 1.0)"
    )
    parser.add_argument(
        "--ori-cost",
        type=float,
        default=1.0,
        help="Orientation task cost (default: 1.0)",
    )
    parser.add_argument(
        "--frame-position-error-limit",
        type=float,
        default=0.0,
        help=(
            "Maximum translational FrameTask error used by each IK substep in "
            "meters; 0 keeps the full error."
        ),
    )
    parser.add_argument(
        "--frame-orientation-error-limit",
        type=float,
        default=0.0,
        help=(
            "Maximum rotational FrameTask error used by each IK substep in "
            "radians; 0 keeps the full error."
        ),
    )
    parser.add_argument(
        "--frame-error-limit-linear-slow",
        type=float,
        default=0.2,
        help=(
            "Desired translational speed where FrameTask error limiting starts in m/s."
        ),
    )
    parser.add_argument(
        "--frame-error-limit-linear-fast",
        type=float,
        default=0.5,
        help=(
            "Desired translational speed where FrameTask error limiting reaches "
            "full strength in m/s."
        ),
    )
    parser.add_argument(
        "--frame-error-limit-activation-rise-rate",
        type=float,
        default=4.0,
        help="Maximum FrameTask error-limit activation increase per second.",
    )
    parser.add_argument(
        "--frame-error-limit-activation-fall-rate",
        type=float,
        default=2.0,
        help="Maximum FrameTask error-limit activation decrease per second.",
    )
    parser.add_argument(
        "--lm-damping",
        type=float,
        default=0.01,
        help="Per-task LM damping (default: 0.01)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.25,
        help="Global Tikhonov regularization (default: 0.25)",
    )
    parser.add_argument("--solver", default="daqp", help="QP backend (default: daqp)")
    parser.add_argument(
        "--max-iters", type=int, default=10, help="IK substeps per event (default: 10)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "Outer control period in seconds. Defaults to 1 / --tick-hz; "
            "each IK substep uses this value divided by --max-iters."
        ),
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=0.01,
        help="Posture task weight, 0=disabled (default: 0.01)",
    )
    parser.add_argument(
        "--diag-reg",
        type=float,
        default=0.0,
        help="QP diagonal regularization (default: 0.0)",
    )
    parser.add_argument(
        "--limit-velocity",
        action="store_true",
        help="Enable per-joint IK velocity limits.",
    )
    parser.add_argument(
        "--joint-limit-recovery-velocity-scale",
        type=float,
        default=1.1,
        help=(
            "Velocity-limit multiplier used only while a joint is outside its "
            "position range (default: 1.1)."
        ),
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional YAML file with 'arm_velocity_limits: [rad/s, ...]'. "
            "Without it, built-in limits are used. Used only with --limit-velocity."
        ),
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=500.0,
        help=(
            "Dora tick rate used when --dt is omitted; must match the dataflow "
            "timer (default: 500.0)."
        ),
    )
    parser.add_argument(
        "--nullspace-cost",
        type=float,
        default=0.3,
        help="One-dimensional nullspace posture cost (default: 0.3).",
    )
    parser.add_argument(
        "--nullspace-return-rate",
        type=float,
        default=0.5,
        help="Nullspace home return rate in 1/s (default: 0.5).",
    )
    parser.add_argument(
        "--nullspace-max-speed",
        type=float,
        default=0.5,
        help="Maximum nullspace-coordinate speed in rad/s (default: 0.5).",
    )
    parser.add_argument(
        "--nullspace-singularity-low",
        type=float,
        default=0.02,
        help="Singularity ratio where nullspace return is disabled (default: 0.02).",
    )
    parser.add_argument(
        "--nullspace-singularity-high",
        type=float,
        default=0.05,
        help="Singularity ratio where nullspace return is fully active (default: 0.05).",
    )
    parser.add_argument(
        "--nullspace-characteristic-length",
        type=float,
        default=0.3,
        help="Length in meters used to normalize translational Jacobian rows.",
    )
    parser.add_argument(
        "--joint-limit-braking",
        action="store_true",
        help=(
            "Enable state-aware distance velocity envelopes near both limits "
            "of every active arm joint."
        ),
    )
    parser.add_argument(
        "--joint-limit-braking-slowdown-distance",
        type=float,
        default=0.5,
        help="Distance in radians over which every joint-limit envelope activates.",
    )
    parser.add_argument(
        "--joint-limit-braking-exponent",
        type=float,
        default=2.0,
        help="Power applied to the joint-limit smoothstep envelope (default: 2).",
    )
    parser.add_argument(
        "--joint-limit-braking-guard-margin",
        type=float,
        default=0.0,
        help="Guard margin inside every physical joint range in radians.",
    )
    parser.add_argument(
        "--joint-limit-braking-reaction-time",
        type=float,
        default=0.0,
        help=(
            "Prediction time applied to measured velocity when estimating "
            "remaining joint-limit distance."
        ),
    )
    parser.add_argument(
        "--joint-limit-braking-distance-buffer",
        type=float,
        default=0.0,
        help="Additional measured-state joint-limit distance buffer in radians.",
    )
    parser.add_argument(
        "--singularity-approach-limit",
        action="store_true",
        help="Limit only the QP displacement component that decreases rho.",
    )
    parser.add_argument(
        "--singularity-ratio-stop",
        type=float,
        default=0.01,
        help="Singularity ratio where maximum approach rate reaches zero.",
    )
    parser.add_argument(
        "--singularity-ratio-slow",
        type=float,
        default=0.05,
        help="Singularity ratio where approach-rate braking starts.",
    )
    parser.add_argument(
        "--singularity-max-approach-rate",
        type=float,
        default=0.5,
        help="Maximum allowed decrease of rho per second outside the slow zone.",
    )
    parser.add_argument(
        "--singularity-braking-exponent",
        type=float,
        default=2.0,
        help="Power applied to the singularity approach smoothstep envelope.",
    )
    parser.add_argument(
        "--singularity-gradient-epsilon",
        type=float,
        default=1e-4,
        help="Central finite-difference step in radians for grad(rho).",
    )
    parser.add_argument(
        "--measured-state-timeout",
        type=float,
        default=0.1,
        help="Seconds before state-aware limits discard a measured q/dq sample.",
    )
    parser.add_argument(
        "--kinetic-energy-cost",
        type=float,
        default=0.0,
        help=(
            "Mink inertia-weighted velocity regularization cost; "
            "0 disables it (default: 0.0)."
        ),
    )


def ik_params_from_args(args: argparse.Namespace) -> IKParams:
    """Build IKParams from parsed args (requires register_ik_args to have been called)."""
    control_dt = args.dt
    if control_dt is None:
        if args.tick_hz <= 0.0:
            raise ValueError("--tick-hz must be positive when --dt is omitted.")
        control_dt = 1.0 / args.tick_hz

    velocity_limits: dict[str, float] | None = None
    if args.limit_velocity:
        caps = _load_velocity_caps(getattr(args, "config", None))
        velocity_limits = {
            f"openarm_{side}_joint{i + 1}": v
            for side in ("left", "right")
            for i, v in enumerate(caps)
        }

    return IKParams(
        position_cost=args.pos_cost,
        orientation_cost=args.ori_cost,
        frame_position_error_limit=args.frame_position_error_limit,
        frame_orientation_error_limit=args.frame_orientation_error_limit,
        frame_error_limit_linear_slow=args.frame_error_limit_linear_slow,
        frame_error_limit_linear_fast=args.frame_error_limit_linear_fast,
        frame_error_limit_activation_rise_rate=(
            args.frame_error_limit_activation_rise_rate
        ),
        frame_error_limit_activation_fall_rate=(
            args.frame_error_limit_activation_fall_rate
        ),
        lm_damping=args.lm_damping,
        damping=args.damping,
        solver=args.solver,
        posture_cost=args.posture_cost,
        diag_reg=args.diag_reg,
        dt=control_dt,
        max_iters=args.max_iters,
        velocity_limits=velocity_limits,
        joint_limit_recovery_velocity_scale=(args.joint_limit_recovery_velocity_scale),
        nullspace_cost=args.nullspace_cost,
        nullspace_return_rate=args.nullspace_return_rate,
        nullspace_max_speed=args.nullspace_max_speed,
        nullspace_singularity_low=args.nullspace_singularity_low,
        nullspace_singularity_high=args.nullspace_singularity_high,
        nullspace_characteristic_length=args.nullspace_characteristic_length,
        joint_limit_braking=args.joint_limit_braking,
        joint_limit_braking_slowdown_distance=(
            args.joint_limit_braking_slowdown_distance
        ),
        joint_limit_braking_exponent=args.joint_limit_braking_exponent,
        joint_limit_braking_guard_margin=args.joint_limit_braking_guard_margin,
        joint_limit_braking_reaction_time=(args.joint_limit_braking_reaction_time),
        joint_limit_braking_distance_buffer=(args.joint_limit_braking_distance_buffer),
        singularity_approach_limit=args.singularity_approach_limit,
        singularity_ratio_stop=args.singularity_ratio_stop,
        singularity_ratio_slow=args.singularity_ratio_slow,
        singularity_max_approach_rate=args.singularity_max_approach_rate,
        singularity_braking_exponent=args.singularity_braking_exponent,
        singularity_gradient_epsilon=args.singularity_gradient_epsilon,
        measured_state_timeout=args.measured_state_timeout,
        kinetic_energy_cost=args.kinetic_energy_cost,
    )
