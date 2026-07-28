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

"""State-aware approach-speed envelopes near scalar joint limits."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from openarm_control.braking import distance_velocity_envelope


@dataclass(frozen=True)
class JointBrakingState:
    """Diagnostics from the most recently assembled joint braking constraint."""

    command_position: np.ndarray
    measured_position: np.ndarray | None
    measured_velocity: np.ndarray | None
    lower_distance: np.ndarray
    upper_distance: np.ndarray
    lower_approach_velocity: np.ndarray
    upper_approach_velocity: np.ndarray


class JointBrakingLimit(mink.Limit):
    """Limit only motion approaching lower and upper scalar-joint guards."""

    def __init__(
        self,
        model: mujoco.MjModel,
        qpos_indices: Collection[int],
        velocities: Mapping[str, npt.ArrayLike],
        *,
        slowdown_distance: float,
        exponent: float = 2.0,
        guard_margin: float = 0.0,
        reaction_time: float = 0.0,
        distance_buffer: float = 0.0,
    ) -> None:
        """Initialize preventive braking for selected limited scalar joints."""
        if not np.isfinite(slowdown_distance) or slowdown_distance <= 0.0:
            raise ValueError("Slowdown distance must be finite and positive.")
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("Braking exponent must be finite and positive.")
        if not np.isfinite(guard_margin) or guard_margin < 0.0:
            raise ValueError("Guard margin must be finite and non-negative.")
        if not np.isfinite(reaction_time) or reaction_time < 0.0:
            raise ValueError("Reaction time must be finite and non-negative.")
        if not np.isfinite(distance_buffer) or distance_buffer < 0.0:
            raise ValueError("Distance buffer must be finite and non-negative.")

        selected_qpos = {int(index) for index in qpos_indices}
        qpos_list: list[int] = []
        dof_list: list[int] = []
        lower_guards: list[float] = []
        upper_guards: list[float] = []
        velocity_limits: list[float] = []
        joint_names: list[str] = []

        for joint_id in range(model.njnt):
            qpos_index = int(model.jnt_qposadr[joint_id])
            if qpos_index not in selected_qpos or not model.jnt_limited[joint_id]:
                continue
            if model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(
                    "JointBrakingLimit only supports scalar hinge and slide joints."
                )
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None or joint_name not in velocities:
                raise ValueError(f"Missing velocity limit for joint {joint_name!r}.")
            max_velocity = np.asarray(velocities[joint_name], dtype=np.float64)
            if max_velocity.size != 1:
                raise ValueError(
                    f"Velocity limit for scalar joint {joint_name!r} must contain "
                    "one value."
                )
            max_velocity_value = float(max_velocity.reshape(-1)[0])
            if not np.isfinite(max_velocity_value) or max_velocity_value <= 0.0:
                raise ValueError(
                    f"Velocity limit for joint {joint_name!r} must be positive."
                )

            physical_lower, physical_upper = model.jnt_range[joint_id]
            lower = float(physical_lower + guard_margin)
            upper = float(physical_upper - guard_margin)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"Invalid braking guards [{lower}, {upper}] for {joint_name!r}."
                )
            if lower < physical_lower or upper > physical_upper:
                raise ValueError(
                    f"Braking guards for {joint_name!r} must remain inside its "
                    "physical joint range."
                )

            joint_names.append(joint_name)
            qpos_list.append(qpos_index)
            dof_list.append(int(model.jnt_dofadr[joint_id]))
            lower_guards.append(lower)
            upper_guards.append(upper)
            velocity_limits.append(max_velocity_value)

        self.model = model
        self.joint_names = tuple(joint_names)
        self.qpos_indices = _readonly_array(qpos_list, dtype=int)
        self.dof_indices = _readonly_array(dof_list, dtype=int)
        self.lower_guard = _readonly_array(lower_guards)
        self.upper_guard = _readonly_array(upper_guards)
        self.max_velocity = _readonly_array(velocity_limits)
        self.slowdown_distance = float(slowdown_distance)
        self.exponent = float(exponent)
        self.reaction_time = float(reaction_time)
        self.distance_buffer = float(distance_buffer)
        self._projection = (
            np.eye(model.nv, dtype=np.float64)[self.dof_indices]
            if self.dof_indices.size
            else None
        )
        self._measured_qpos: np.ndarray | None = None
        self._measured_qvel: np.ndarray | None = None
        self.last_state: JointBrakingState | None = None

    def update_measured_state(
        self,
        qpos: npt.ArrayLike,
        qvel: npt.ArrayLike,
    ) -> None:
        """Update the measured model state used for predictive guard distance."""
        qpos_array = np.asarray(qpos, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        if qpos_array.shape != (self.model.nq,):
            raise ValueError(f"Expected measured qpos shape ({self.model.nq},).")
        if qvel_array.shape != (self.model.nv,):
            raise ValueError(f"Expected measured qvel shape ({self.model.nv},).")
        if not np.all(np.isfinite(qpos_array)) or not np.all(np.isfinite(qvel_array)):
            raise ValueError("Measured joint state must be finite.")
        self._measured_qpos = qpos_array.copy()
        self._measured_qvel = qvel_array.copy()

    def clear_measured_state(self) -> None:
        """Fall back to command-configuration-only braking."""
        self._measured_qpos = None
        self._measured_qvel = None

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        """Return upper and lower approach-displacement inequalities."""
        if self._projection is None:
            return mink.Constraint()
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        command_q = configuration.q[self.qpos_indices]
        lower_distance = command_q - self.lower_guard
        upper_distance = self.upper_guard - command_q
        measured_q: np.ndarray | None = None
        measured_dq: np.ndarray | None = None

        if self._measured_qpos is not None and self._measured_qvel is not None:
            measured_q = self._measured_qpos[self.qpos_indices]
            measured_dq = self._measured_qvel[self.dof_indices]
            measured_lower = (
                measured_q
                - self.lower_guard
                - self.reaction_time * np.maximum(-measured_dq, 0.0)
                - self.distance_buffer
            )
            measured_upper = (
                self.upper_guard
                - measured_q
                - self.reaction_time * np.maximum(measured_dq, 0.0)
                - self.distance_buffer
            )
            lower_distance = np.minimum(lower_distance, measured_lower)
            upper_distance = np.minimum(upper_distance, measured_upper)

        slowdown = np.full_like(self.max_velocity, self.slowdown_distance)
        lower_velocity = distance_velocity_envelope(
            lower_distance,
            self.max_velocity,
            slowdown,
            exponent=self.exponent,
        )
        upper_velocity = distance_velocity_envelope(
            upper_distance,
            self.max_velocity,
            slowdown,
            exponent=self.exponent,
        )
        self.last_state = JointBrakingState(
            command_position=command_q.copy(),
            measured_position=None if measured_q is None else measured_q.copy(),
            measured_velocity=None if measured_dq is None else measured_dq.copy(),
            lower_distance=lower_distance.copy(),
            upper_distance=upper_distance.copy(),
            lower_approach_velocity=lower_velocity.copy(),
            upper_approach_velocity=upper_velocity.copy(),
        )

        G = np.vstack([self._projection, -self._projection])
        h = dt * np.hstack([upper_velocity, lower_velocity])
        return mink.Constraint(G=G, h=h)


def _readonly_array(
    values: Collection[float] | Collection[int],
    *,
    dtype: type = np.float64,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array
