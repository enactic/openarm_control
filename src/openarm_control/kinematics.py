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
    kin = Kinematics(setup, IKParams(dt=0.1, max_iters=5))
    kin.set_target("right", pose)
    kin.set_target("left", pose)
    result = kin.solve()                    # float32[16] or None
"""

from __future__ import annotations

import mink
import mink.exceptions
import mujoco
import numpy as np

from openarm_control.config import ArmSetup, frame_name
from openarm_control.geometry.poses import pose_to_se3
from openarm_control.ik_params import (
    IKParams,
    validate_ik_params,
)
from openarm_control.qp.arm_joint_limit import (
    ArmConfigurationLimit,
    ArmJointLimit,
)
from openarm_control.qp.bounded_frame_task import BoundedFrameTask
from openarm_control.qp.kinetic_energy_task import (
    KineticEnergyRegularizationTask,
)
from openarm_control.qp.nullspace_posture_task import NullspacePostureTask
from openarm_control.qp.singularity_approach_limit import (
    SingularityApproachLimit,
)


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
    ) -> None:
        """Update measured qpos used by state-aware limits."""
        self._require_ik().update_measured_state(qpos16)

    def clear_measured_state(self) -> None:
        """Discard measured qpos from state-aware limits."""
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
        validate_ik_params(params)
        self._setup = setup
        self._sides = setup.sides
        self._solver_name = params.solver
        self._posture_cost = params.posture_cost
        self._joint_resolver = setup.joint_resolver
        arm_qpos_by_side = {
            side: setup.joint_resolver.arm_qpos_indices(side) for side in setup.sides
        }
        self._arm_dofs_by_side = {
            side: setup.joint_resolver.arm_dof_indices(side) for side in setup.sides
        }
        self._substep_dt = params.dt / params.max_iters
        self._max_iters = params.max_iters

        self._config = mink.Configuration(setup.model)
        self._config.update(q=setup.data.qpos.copy())
        mid_qpos = self._config.data.qpos.copy()

        task_kwargs = dict(
            position_cost=params.position_cost,
            orientation_cost=params.orientation_cost,
            lm_damping=params.lm_damping,
        )
        self._tasks: dict[
            str,
            mink.FrameTask | mink.RelativeFrameTask | BoundedFrameTask,
        ]
        if setup.origin_id is not None:
            self._tasks = {
                side: mink.RelativeFrameTask(
                    frame_name=_frame_name(setup, side),
                    frame_type=setup.frame_types[side],
                    root_name=frame_name(
                        setup.model, setup.origin_id, setup.origin_type
                    ),
                    root_type=setup.origin_type,
                    **task_kwargs,
                )
                for side in setup.sides
            }
        else:
            self._tasks = {
                side: mink.FrameTask(
                    frame_name=_frame_name(setup, side),
                    frame_type=setup.frame_types[side],
                    **task_kwargs,
                )
                for side in setup.sides
            }

        if (
            params.frame_position_error_limit > 0.0
            or params.frame_orientation_error_limit > 0.0
        ):
            bounded_tasks = {
                side: BoundedFrameTask(
                    task,
                    position_error_limit=params.frame_position_error_limit,
                    orientation_error_limit=params.frame_orientation_error_limit,
                    control_dt=params.dt,
                    substeps=params.max_iters,
                    target_linear_speed_slow=params.target_linear_speed_slow,
                    target_linear_speed_fast=params.target_linear_speed_fast,
                    position_error_latch_threshold=(
                        params.position_error_latch_threshold
                    ),
                )
                for side, task in self._tasks.items()
            }
            for task in bounded_tasks.values():
                task.set_limit_activation(0.0)
            self._tasks = bounded_tasks

        active_qpos = {
            int(index) for side in setup.sides for index in arm_qpos_by_side[side]
        }
        active_dofs = {
            int(index) for side in setup.sides for index in self._arm_dofs_by_side[side]
        }
        freeze_dofs = [
            dof_index
            for dof_index in range(setup.model.nv)
            if dof_index not in active_dofs
        ]
        self._freeze_task: mink.DofFreezingTask | None = (
            mink.DofFreezingTask(model=setup.model, dof_indices=freeze_dofs)
            if freeze_dofs
            else None
        )

        self._joint_limit: ArmJointLimit | None = None
        if params.velocity_limits is not None:
            self._joint_limit = ArmJointLimit(
                model=setup.model,
                qpos_indices=active_qpos,
                velocities=params.velocity_limits,
                position_gain=params.joint_limit_gain,
                braking_distance=(
                    params.joint_braking_distance if params.joint_braking else None
                ),
                braking_exponent=params.joint_braking_exponent,
                braking_distance_buffer=params.joint_braking_distance_buffer,
            )
            self._limits: list[mink.Limit] = [self._joint_limit]
        else:
            self._limits = [
                ArmConfigurationLimit(
                    setup.model,
                    active_qpos,
                    gain=params.joint_limit_gain,
                )
            ]

        self._singularity_limits: dict[str, SingularityApproachLimit] = {}
        if params.singularity_max_approach_rate > 0.0:
            for side in setup.sides:
                limit = SingularityApproachLimit(
                    model=setup.model,
                    frame_task=self._tasks[side],
                    dof_indices=self._arm_dofs_by_side[side],
                    characteristic_length=params.jacobian_characteristic_length,
                    ratio_stop=params.singularity_ratio_stop,
                    ratio_slow=params.singularity_ratio_slow,
                    max_approach_rate=params.singularity_max_approach_rate,
                    exponent=params.singularity_braking_exponent,
                    gradient_epsilon=params.singularity_gradient_epsilon,
                )
                self._singularity_limits[side] = limit
                self._limits.append(limit)

        self._posture_task = mink.PostureTask(setup.model, cost=params.posture_cost)
        self._posture_task.set_target(mid_qpos)

        self._nullspace_tasks: dict[str, NullspacePostureTask] = {}
        if params.nullspace_cost > 0.0:
            for side in setup.sides:
                self._nullspace_tasks[side] = NullspacePostureTask(
                    model=setup.model,
                    frame_task=self._tasks[side],
                    dof_indices=self._arm_dofs_by_side[side],
                    home_qpos=mid_qpos,
                    cost=params.nullspace_cost,
                    dt=self._substep_dt,
                    return_rate=params.nullspace_return_rate,
                    max_speed=params.nullspace_max_speed,
                    singularity_low=params.nullspace_ratio_low,
                    singularity_high=params.nullspace_ratio_high,
                    characteristic_length=params.jacobian_characteristic_length,
                )

        self._kinetic_energy_task: KineticEnergyRegularizationTask | None = None
        if params.kinetic_energy_cost > 0.0:
            self._kinetic_energy_task = KineticEnergyRegularizationTask(
                cost=params.kinetic_energy_cost
            )
            self._kinetic_energy_task.set_dt(self._substep_dt)

        self._damping = params.damping
        self._pending: set[str] = set(setup.sides)
        self._gripper = np.zeros(2, dtype=np.float32)

    def set_target(self, side: str, pose: np.ndarray) -> None:
        target = np.asarray(pose, dtype=np.float64)
        task = self._tasks[side]
        transform = pose_to_se3(target)
        if isinstance(task, BoundedFrameTask):
            task.set_target_and_update_schedule(transform, self._config)
        else:
            task.set_target(transform)
        self._pending.discard(side)

    def sync(self, values16: np.ndarray) -> None:
        values = np.asarray(values16, dtype=np.float64)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ValueError("Bimanual driver qpos must be a finite shape-(16,) array.")
        qpos = self._config.data.qpos.copy()
        self._joint_resolver.set_qpos(qpos, values[:8], "right")
        self._joint_resolver.set_qpos(qpos, values[8:16], "left")
        self._config.update(q=qpos)
        # set_gripper() remains the sole gripper-command writer.

    def update_measured_state(
        self,
        qpos16: np.ndarray,
    ) -> None:
        """Update measured state without changing the IK command configuration."""
        measured_qpos = self._setup.driver_qpos_to_mujoco(
            qpos16,
            base_qpos=self._config.q,
        )

        if self._joint_limit is not None:
            self._joint_limit.update_measured_state(measured_qpos)
        for limit in self._singularity_limits.values():
            limit.update_measured_configuration(measured_qpos)

    def clear_measured_state(self) -> None:
        """Clear measured state from every state-aware limit."""
        if self._joint_limit is not None:
            self._joint_limit.clear_measured_state()
        for limit in self._singularity_limits.values():
            limit.clear_measured_configuration()

    def ready(self) -> bool:
        return len(self._pending) == 0

    def solve(self) -> np.ndarray | None:
        tasks = list(self._tasks.values())
        if self._posture_cost > 0.0:
            tasks.append(self._posture_task)
        tasks.extend(self._nullspace_tasks.values())
        if self._kinetic_energy_task is not None:
            tasks.append(self._kinetic_energy_task)
        constraints = [self._freeze_task] if self._freeze_task else []

        q_before = self._config.data.qpos.copy()
        for limit in self._singularity_limits.values():
            limit.prepare(self._config)
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
                    damping=self._damping,
                )
            except mink.exceptions.NoSolutionFound:
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


def _frame_name(setup: ArmSetup, side: str) -> str:
    return frame_name(setup.model, setup.frame_ids[side], setup.frame_types[side])
