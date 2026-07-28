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

"""Joint position and velocity limits that remain feasible after overshoot."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import mink
import mujoco
import numpy as np
import numpy.typing as npt


class RecoverableConfigurationLimit(mink.Limit):
    """Combine scalar joint position and velocity bounds into one QP limit.

    Inside the configured range this is the intersection of Mink's
    ``ConfigurationLimit`` and ``VelocityLimit``. Outside the range, the
    position correction is clipped to a recoverable velocity bound so the two
    requirements cannot make the QP infeasible.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        qpos_indices: Collection[int],
        velocities: Mapping[str, npt.ArrayLike],
        *,
        gain: float = 0.95,
        recovery_velocity_scale: float = 1.1,
    ) -> None:
        """Initialize limits for selected scalar hinge and slide joints."""
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in the range (0, 1].")
        if recovery_velocity_scale < 1.0:
            raise ValueError("recovery_velocity_scale must be at least 1.0.")

        selected_qpos = {int(index) for index in qpos_indices}
        joint_ids: list[int] = []
        qpos_list: list[int] = []
        dof_list: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        velocity_limits: list[float] = []

        for joint_id in range(model.njnt):
            qpos_index = int(model.jnt_qposadr[joint_id])
            if qpos_index not in selected_qpos or not model.jnt_limited[joint_id]:
                continue
            joint_type = model.jnt_type[joint_id]
            if joint_type not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(
                    "RecoverableConfigurationLimit only supports scalar hinge "
                    "and slide joints."
                )

            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None or joint_name not in velocities:
                raise ValueError(f"Missing velocity limit for joint {joint_name!r}.")
            max_velocity = np.asarray(velocities[joint_name], dtype=np.float64)
            if max_velocity.size != 1:
                raise ValueError(
                    f"Velocity limit for scalar joint {joint_name!r} must have "
                    "exactly one value."
                )
            max_velocity_value = float(max_velocity.reshape(-1)[0])
            if not np.isfinite(max_velocity_value) or max_velocity_value <= 0.0:
                raise ValueError(
                    f"Velocity limit for joint {joint_name!r} must be positive."
                )

            joint_ids.append(joint_id)
            qpos_list.append(qpos_index)
            dof_list.append(int(model.jnt_dofadr[joint_id]))
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
            velocity_limits.append(max_velocity_value)

        self.model = model
        self.joint_ids = _readonly_int_array(joint_ids)
        self.qpos_indices = _readonly_int_array(qpos_list)
        self.indices = _readonly_int_array(dof_list)
        self.lower = _readonly_float_array(lower)
        self.upper = _readonly_float_array(upper)
        self.limit = _readonly_float_array(velocity_limits)
        self.gain = gain
        self.recovery_velocity_scale = recovery_velocity_scale
        self.projection_matrix = (
            np.eye(model.nv)[self.indices] if self.indices.size else None
        )

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        """Return recoverable position and velocity inequalities."""
        if self.projection_matrix is None:
            return mink.Constraint()
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        q = configuration.q[self.qpos_indices]
        position_lower = self.gain * (self.lower - q)
        position_upper = self.gain * (self.upper - q)

        outside = (q < self.lower) | (q > self.upper)
        velocity_scale = np.where(outside, self.recovery_velocity_scale, 1.0)
        max_displacement = dt * self.limit * velocity_scale

        step_lower = np.clip(position_lower, -max_displacement, max_displacement)
        step_upper = np.clip(position_upper, -max_displacement, max_displacement)

        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([step_upper, -step_lower])
        return mink.Constraint(G=G, h=h)


def _readonly_int_array(values: Collection[int]) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    array.setflags(write=False)
    return array


def _readonly_float_array(values: Collection[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array
