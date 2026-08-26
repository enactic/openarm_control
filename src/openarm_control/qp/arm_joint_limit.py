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

"""Joint position/velocity bounds with optional measured-position braking."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import mink
import mujoco
import numpy as np
import numpy.typing as npt


class ArmConfigurationLimit(mink.ConfigurationLimit):
    """Apply Mink's configuration limits only to selected scalar arm joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        qpos_indices: Collection[int],
        *,
        gain: float,
    ) -> None:
        """Build a configuration limit projected onto selected arm DoFs."""
        super().__init__(model, gain=gain)
        selected_qpos = {int(index) for index in qpos_indices}
        active_dofs: list[int] = []
        for joint_id in range(model.njnt):
            if (
                not model.jnt_limited[joint_id]
                or int(model.jnt_qposadr[joint_id]) not in selected_qpos
            ):
                continue
            if int(model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError("ArmConfigurationLimit only supports scalar joints.")
            active_dofs.append(int(model.jnt_dofadr[joint_id]))
        self.indices = _readonly(active_dofs, dtype=int)
        self.projection_matrix = (
            np.eye(model.nv, dtype=np.float64)[self.indices]
            if self.indices.size
            else None
        )


class ArmJointLimit(mink.Limit):
    """Apply one feasible position, velocity, and braking envelope per arm joint."""

    def __init__(
        self,
        model: mujoco.MjModel,
        qpos_indices: Collection[int],
        velocities: Mapping[str, npt.ArrayLike],
        *,
        position_gain: float,
        braking_distance: float | None = None,
        braking_exponent: float,
        braking_distance_buffer: float,
    ) -> None:
        """Scan selected scalar joints once and build their shared projection."""
        if not np.isfinite(position_gain) or not 0.0 < position_gain <= 1.0:
            raise ValueError("position_gain must be finite and in (0, 1].")
        if braking_distance is not None and (
            not np.isfinite(braking_distance) or braking_distance <= 0.0
        ):
            raise ValueError("braking_distance must be finite and positive.")
        if not np.isfinite(braking_exponent) or braking_exponent <= 0.0:
            raise ValueError("braking_exponent must be finite and positive.")
        if not np.isfinite(braking_distance_buffer) or braking_distance_buffer < 0.0:
            raise ValueError("braking_distance_buffer must be finite and non-negative.")

        selected_qpos = {int(index) for index in qpos_indices}
        qpos: list[int] = []
        dofs: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        max_velocity: list[float] = []

        for joint_id in range(model.njnt):
            qpos_index = int(model.jnt_qposadr[joint_id])
            if qpos_index not in selected_qpos:
                continue
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if not model.jnt_limited[joint_id]:
                print(
                    f"Warning: selected arm joint {name!r} has no position limit; "
                    "its velocity limit will be skipped."
                )
                continue
            if int(model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError("ArmJointLimit only supports scalar joints.")

            if name is None or name not in velocities:
                raise ValueError(f"Missing velocity limit for joint {name!r}.")
            velocity = np.asarray(velocities[name], dtype=np.float64)
            if velocity.size != 1:
                raise ValueError(f"Velocity limit for {name!r} must be scalar.")
            velocity_value = float(velocity.reshape(-1)[0])
            if not np.isfinite(velocity_value) or velocity_value <= 0.0:
                raise ValueError(f"Velocity limit for {name!r} must be positive.")

            qpos.append(qpos_index)
            dofs.append(int(model.jnt_dofadr[joint_id]))
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
            max_velocity.append(velocity_value)

        self.model = model
        self.qpos_indices = _readonly(qpos, dtype=int)
        self.dof_indices = _readonly(dofs, dtype=int)
        self.indices = self.dof_indices
        self.lower = _readonly(lower)
        self.upper = _readonly(upper)
        self.max_velocity = _readonly(max_velocity)
        self.position_gain = float(position_gain)
        self.braking_distance = braking_distance
        self.braking_exponent = float(braking_exponent)
        self.braking_distance_buffer = float(braking_distance_buffer)
        self.projection_matrix = (
            np.eye(model.nv, dtype=np.float64)[self.dof_indices]
            if self.dof_indices.size
            else None
        )
        self._measured_qpos: np.ndarray | None = None

    def update_measured_state(self, qpos: npt.ArrayLike) -> None:
        """Update measured position used by the preventive braking envelope."""
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != (self.model.nq,):
            raise ValueError(f"Expected measured qpos shape ({self.model.nq},).")
        if not np.all(np.isfinite(qpos_array)):
            raise ValueError("Measured joint positions must be finite.")
        self._measured_qpos = qpos_array.copy()

    def clear_measured_state(self) -> None:
        """Fall back to command-state braking until a fresh sample is supplied."""
        self._measured_qpos = None

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        """Return the intersection of recoverable and preventive step bounds."""
        if self.projection_matrix is None:
            return mink.Constraint()
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        command_q = configuration.q[self.qpos_indices]
        max_step = dt * self.max_velocity
        position_lower = self.position_gain * (self.lower - command_q)
        position_upper = self.position_gain * (self.upper - command_q)
        lower_step = np.clip(position_lower, -max_step, max_step)
        upper_step = np.clip(position_upper, -max_step, max_step)

        measured_q: np.ndarray | None = None
        lower_distance: np.ndarray | None = None
        upper_distance: np.ndarray | None = None
        if self.braking_distance is not None:
            lower_distance = command_q - self.lower
            upper_distance = self.upper - command_q
            if self._measured_qpos is not None:
                measured_q = self._measured_qpos[self.qpos_indices]
                measured_lower = measured_q - self.lower - self.braking_distance_buffer
                measured_upper = self.upper - measured_q - self.braking_distance_buffer
                lower_distance = np.minimum(lower_distance, measured_lower)
                upper_distance = np.minimum(upper_distance, measured_upper)

            lower_velocity = _distance_velocity_envelope(
                lower_distance,
                self.max_velocity,
                self.braking_distance,
                self.braking_exponent,
            )
            upper_velocity = _distance_velocity_envelope(
                upper_distance,
                self.max_velocity,
                self.braking_distance,
                self.braking_exponent,
            )
            lower_step = np.maximum(lower_step, -dt * lower_velocity)
            upper_step = np.minimum(upper_step, dt * upper_velocity)

        return mink.Constraint(
            G=np.vstack([self.projection_matrix, -self.projection_matrix]),
            h=np.hstack([upper_step, -lower_step]),
        )


def _distance_velocity_envelope(
    distance: np.ndarray,
    max_velocity: np.ndarray,
    slowdown_distance: float,
    exponent: float,
) -> np.ndarray:
    """Return the tested smooth distance-dependent approach-speed envelope."""
    u = np.clip(np.maximum(distance, 0.0) / slowdown_distance, 0.0, 1.0)
    smoothstep = u * u * (3.0 - 2.0 * u)
    return max_velocity * np.power(smoothstep, exponent)


def _readonly(
    values: Collection[float] | Collection[int],
    *,
    dtype: type = np.float64,
) -> np.ndarray:
    output = np.asarray(values, dtype=dtype)
    output.setflags(write=False)
    return output
