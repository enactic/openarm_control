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

"""Limit only the configuration displacement approaching a singularity."""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from openarm_control.braking import powered_smoothstep
from openarm_control.singularity import (
    normalized_arm_jacobian,
    singularity_ratio,
)


@dataclass(frozen=True)
class SingularityApproachState:
    """Diagnostics for the current linearized singularity-rate constraint."""

    command_ratio: float
    measured_ratio: float | None
    effective_ratio: float
    activation: float
    max_approach_rate: float
    gradient: np.ndarray
    gradient_norm: float
    singular_values: np.ndarray


class SingularityApproachLimit(mink.Limit):
    """Bound the maximum first-order decrease of sigma_min / sigma_max."""

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_task: mink.FrameTask | mink.RelativeFrameTask,
        dof_indices: npt.ArrayLike,
        *,
        characteristic_length: float,
        ratio_stop: float,
        ratio_slow: float,
        max_approach_rate: float,
        exponent: float = 2.0,
        gradient_epsilon: float = 1e-4,
    ) -> None:
        """Initialize a one-sided singularity approach-rate constraint."""
        indices = np.asarray(dof_indices, dtype=int)
        if indices.shape != (7,):
            raise ValueError(f"Expected seven arm DoF indices, got {indices.shape}.")
        if np.unique(indices).size != indices.size:
            raise ValueError("Arm DoF indices must be unique.")
        if np.any(indices < 0) or np.any(indices >= model.nv):
            raise ValueError("Arm DoF indices are outside the model tangent space.")
        if characteristic_length <= 0.0:
            raise ValueError("Characteristic length must be positive.")
        if not 0.0 <= ratio_stop < ratio_slow:
            raise ValueError("Expected 0 <= ratio_stop < ratio_slow.")
        if not np.isfinite(max_approach_rate) or max_approach_rate <= 0.0:
            raise ValueError("Maximum singularity approach rate must be positive.")
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("Singularity braking exponent must be positive.")
        if not np.isfinite(gradient_epsilon) or gradient_epsilon <= 0.0:
            raise ValueError("Gradient epsilon must be finite and positive.")

        self.model = model
        self.frame_task = frame_task
        self.dof_indices = indices.copy()
        self.characteristic_length = float(characteristic_length)
        self.ratio_stop = float(ratio_stop)
        self.ratio_slow = float(ratio_slow)
        self.max_rate = float(max_approach_rate)
        self.exponent = float(exponent)
        self.gradient_epsilon = float(gradient_epsilon)
        self._scratch = mink.Configuration(model)
        self._measured_qpos: np.ndarray | None = None
        self._G: np.ndarray | None = None
        self._allowed_rate = self.max_rate
        self.last_state: SingularityApproachState | None = None

    def update_measured_configuration(self, qpos: npt.ArrayLike) -> None:
        """Update the measured configuration used to activate the envelope."""
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != (self.model.nq,):
            raise ValueError(f"Expected measured qpos shape ({self.model.nq},).")
        if not np.all(np.isfinite(qpos_array)):
            raise ValueError("Measured configuration must be finite.")
        self._measured_qpos = qpos_array.copy()

    def clear_measured_configuration(self) -> None:
        """Fall back to the command configuration singularity ratio."""
        self._measured_qpos = None

    def prepare(self, configuration: mink.Configuration) -> None:
        """Linearize rho(q) once for the next outer IK control step."""
        command_ratio, singular_values = self._ratio(configuration)
        measured_ratio: float | None = None
        if self._measured_qpos is not None:
            self._scratch.update(q=self._measured_qpos)
            measured_ratio, _ = self._ratio(self._scratch)
        effective_ratio = (
            command_ratio
            if measured_ratio is None
            else min(command_ratio, measured_ratio)
        )
        u = (effective_ratio - self.ratio_stop) / (self.ratio_slow - self.ratio_stop)
        activation = float(powered_smoothstep(u, exponent=self.exponent))
        self._allowed_rate = self.max_rate * activation
        gradient = self._finite_difference_gradient(configuration)

        G = np.zeros((1, self.model.nv), dtype=np.float64)
        G[0, self.dof_indices] = -gradient
        self._G = G
        self.last_state = SingularityApproachState(
            command_ratio=command_ratio,
            measured_ratio=measured_ratio,
            effective_ratio=effective_ratio,
            activation=activation,
            max_approach_rate=self._allowed_rate,
            gradient=gradient.copy(),
            gradient_norm=float(np.linalg.norm(gradient)),
            singular_values=singular_values.copy(),
        )

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        """Return -grad(rho)^T delta_q <= max_rate(rho) * dt."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self._G is None:
            self.prepare(configuration)
        assert self._G is not None
        return mink.Constraint(
            G=self._G,
            h=np.array([self._allowed_rate * dt], dtype=np.float64),
        )

    def _ratio(self, configuration: mink.Configuration) -> tuple[float, np.ndarray]:
        jacobian = normalized_arm_jacobian(
            self.frame_task,
            configuration,
            self.dof_indices,
            self.characteristic_length,
        )
        return singularity_ratio(jacobian)

    def _finite_difference_gradient(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        q0 = configuration.q.copy()
        tangent = np.zeros(self.model.nv, dtype=np.float64)
        gradient = np.empty(self.dof_indices.size, dtype=np.float64)
        eps = self.gradient_epsilon

        for index, dof in enumerate(self.dof_indices):
            tangent[dof] = 1.0
            q_plus = q0.copy()
            mujoco.mj_integratePos(self.model, q_plus, tangent, eps)
            self._scratch.update(q=q_plus)
            ratio_plus, _ = self._ratio(self._scratch)

            q_minus = q0.copy()
            mujoco.mj_integratePos(self.model, q_minus, tangent, -eps)
            self._scratch.update(q=q_minus)
            ratio_minus, _ = self._ratio(self._scratch)
            gradient[index] = (ratio_plus - ratio_minus) / (2.0 * eps)
            tangent[dof] = 0.0

        return gradient
