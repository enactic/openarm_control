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

Usage:
    # FK only
    kin = Kinematics(setup)
    pose = kin.fk("right", joints)          # float32[7]
    pose_r, pose_l = kin.fk_bimanual(r, l)  # single mj_forward

    # FK + IK
    kin = Kinematics(setup, IKParams())
    kin.set_target("right", pose)
    kin.set_target("left", pose)
    result = kin.solve(dt=0.1, n_iters=5)   # float32[16] or None
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mink
import mink.exceptions
import mujoco
import numpy as np

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S, ArmSetup
from openarm_control.poses import pose_to_se3


@dataclass
class IKParams:
    """Configuration for the mink QP-based IK solver."""

    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 0.01
    damping: float = 0.25
    solver: str = "daqp"
    posture_cost: float = 0.01
    diag_reg: float = 0.0
    dt: float = 0.1
    max_iters: int = 5
    velocity_limits: dict[str, float] | None = None


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
        """Set EE target for one arm. pose: float32[7] = [px, py, pz, qw, qx, qy, qz]."""
        self._require_ik().set_target(side, pose)

    def sync(self, values16: np.ndarray) -> None:
        """Sync IK internal config from float32[16] driver state (right[8]+left[8])."""
        self._require_ik().sync(values16)

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
        self._sides = setup.sides
        self._solver_name = params.solver
        self._posture_cost = params.posture_cost
        self._joint_resolver = setup.joint_resolver
        self._dt = params.dt
        self._max_iters = params.max_iters

        self._config = mink.Configuration(setup.model)
        self._config.update(q=setup.data.qpos.copy())
        mid_qpos = self._config.data.qpos.copy()

        task_kwargs = dict(
            position_cost=params.position_cost,
            orientation_cost=params.orientation_cost,
            lm_damping=params.lm_damping,
        )
        self._tasks: dict[str, mink.FrameTask] = {
            side: mink.FrameTask(
                frame_name=_frame_name(setup, side),
                frame_type=setup.frame_types[side],
                **task_kwargs,
            )
            for side in setup.sides
        }

        active_qpos: set[int] = set(
            setup.joint_resolver._right.arm_qpos.tolist()
        ) | set(setup.joint_resolver._left.arm_qpos.tolist())
        freeze_dofs = [
            int(setup.model.jnt_dofadr[j])
            for j in range(setup.model.njnt)
            if setup.model.jnt_qposadr[j] not in active_qpos
        ]
        print(active_qpos)
        print(freeze_dofs)
        self._freeze_task: mink.DofFreezingTask | None = (
            mink.DofFreezingTask(model=setup.model, dof_indices=freeze_dofs)
            if freeze_dofs
            else None
        )

        self._limits = [mink.ConfigurationLimit(setup.model)]

        if params.velocity_limits is not None:
            self._limits.append(mink.VelocityLimit(setup.model, params.velocity_limits))

        self._posture_task = mink.PostureTask(setup.model, cost=params.posture_cost)
        self._posture_task.set_target(mid_qpos)

        self._solver_params: dict = {"damping": params.damping}
        if params.diag_reg > 0.0:
            self._solver_params["diag_reg"] = params.diag_reg

        self._pending: set[str] = set(setup.sides)
        self._gripper = np.zeros(2, dtype=np.float32)

    def set_target(self, side: str, pose: np.ndarray) -> None:
        self._tasks[side].set_target(pose_to_se3(pose))
        self._pending.discard(side)

    def sync(self, values16: np.ndarray) -> None:
        qpos = self._config.data.qpos.copy()
        self._joint_resolver.set_qpos(qpos, values16[:8], "right")
        self._joint_resolver.set_qpos(qpos, values16[8:16], "left")
        self._config.update(q=qpos)
        self._gripper[0] = values16[7]
        self._gripper[1] = values16[15]

    def ready(self) -> bool:
        return len(self._pending) == 0

    def solve(self) -> np.ndarray | None:
        tasks = list(self._tasks.values())
        if self._posture_cost > 0.0:
            tasks.append(self._posture_task)
        constraints = [self._freeze_task] if self._freeze_task else []

        for _ in range(self._max_iters):
            try:
                vel = mink.solve_ik(
                    self._config,
                    tasks,
                    self._dt,
                    self._solver_name,
                    limits=self._limits,
                    constraints=constraints,
                    safety_break=False,
                    **self._solver_params,
                )
            except mink.exceptions.NoSolutionFound:
                try:
                    vel = mink.solve_ik(
                        self._config,
                        tasks,
                        self._dt,
                        self._solver_name,
                        limits=[],
                        constraints=constraints,
                        safety_break=False,
                        **self._solver_params,
                    )
                except mink.exceptions.NoSolutionFound:
                    print(
                        "Warning: IK solver failed (constrained and unconstrained). Skipping step."
                    )
                    return None
            self._config.integrate_inplace(vel, self._dt)

        self._pending = set(self._sides)

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
    ftype = setup.frame_types[side]
    fid = setup.frame_ids[side]
    obj = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
    }[ftype]
    return mujoco.mj_id2name(setup.model, obj, fid)


def _convert_velocity(
    rad_per_sec: float,
    dt: float,
    max_iters: int,
    tick_hz: float,
) -> float:
    if max_iters <= 0 or dt <= 0.0 or tick_hz <= 0.0:
        raise ValueError("max_iters, dt, and tick_hz must all be positive.")
    return rad_per_sec / (max_iters * dt * tick_hz)


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
        "--max-iters", type=int, default=5, help="IK iterations per event (default: 5)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Integration timestep per iteration (default: 0.1)",
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
        "--vel-scale",
        type=float,
        default=None,
        help="Scale velocity limit safety. 1=90deg/s for shoulder. Unset = VelocityLimit disabled.",
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=500.0,
        help="Dora tick rate in Hz; used only for --vel-scale unit conversion",
    )


def ik_params_from_args(args: argparse.Namespace) -> IKParams:
    """Build IKParams from parsed args (requires register_ik_args to have been called)."""
    velocity_limits: dict[str, float] | None = None
    if args.vel_scale is not None:
        velocity_limits = {
            f"openarm_{side}_joint{i + 1}": _convert_velocity(
                rad_per_sec=v * args.vel_scale,
                dt=args.dt,
                max_iters=args.max_iters,
                tick_hz=args.tick_hz,
            )
            for side in ("left", "right")
            for i, v in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
        }

    return IKParams(
        position_cost=args.pos_cost,
        orientation_cost=args.ori_cost,
        lm_damping=args.lm_damping,
        damping=args.damping,
        solver=args.solver,
        posture_cost=args.posture_cost,
        diag_reg=args.diag_reg,
        dt=args.dt,
        max_iters=args.max_iters,
        velocity_limits=velocity_limits,
    )
