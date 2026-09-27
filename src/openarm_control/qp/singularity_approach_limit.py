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

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from openarm_control.geometry.jacobian import (
    frame_is_independent_of_dofs,
    normalized_arm_jacobian,
    relative_root_is_independent_of_dofs,
    singularity_ratio,
)


class SingularityApproachLimit(mink.Limit):
    """Bound the maximum first-order decrease of sigma_min / sigma_max."""

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_task: mink.Task,
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
        self._jacobian_columns = (
            slice(int(indices[0]), int(indices[-1]) + 1)
            if np.all(np.diff(indices) == 1)
            else self.dof_indices
        )
        self.characteristic_length = float(characteristic_length)
        self.ratio_stop = float(ratio_stop)
        self.ratio_slow = float(ratio_slow)
        self.max_rate = float(max_approach_rate)
        self.exponent = float(exponent)
        self.gradient_epsilon = float(gradient_epsilon)
        self._root_is_independent_of_dofs = relative_root_is_independent_of_dofs(
            frame_task,
            model,
            indices,
        )
        self._scratch = mink.Configuration(model)
        native_task = getattr(frame_task, "frame_task", frame_task)
        self._frame_id = getattr(model, native_task.frame_type)(
            native_task.frame_name
        ).id
        self._jacobian_func = {
            "body": mujoco.mj_jacBody,
            "site": mujoco.mj_jacSite,
            "geom": mujoco.mj_jacGeom,
        }[native_task.frame_type]
        self._world_jacobian = np.empty((6, model.nv), dtype=np.float64)
        self._tangent = np.zeros(model.nv, dtype=np.float64)
        self._perturbed_q = np.empty(model.nq, dtype=np.float64)
        self._perturbed_jacobians = np.empty((indices.size, 2, 6, indices.size))
        self._gradient = np.empty(indices.size, dtype=np.float64)
        self._measured_qpos: np.ndarray | None = None
        self._G: np.ndarray | None = None
        self._allowed_rate = self.max_rate

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

    def prepare(
        self,
        configuration: mink.Configuration,
        *,
        command_ratio: float | None = None,
        gradient: np.ndarray | None = None,
    ) -> None:
        """Linearize rho(q) once, optionally reusing a paired gradient."""
        if command_ratio is None:
            command_ratio, _ = self._ratio(configuration)
        measured_ratio: float | None = None
        if self._measured_qpos is not None:
            self._update_scratch(self._measured_qpos)
            measured_ratio, _ = self._ratio(self._scratch)
        effective_ratio = (
            command_ratio
            if measured_ratio is None
            else min(command_ratio, measured_ratio)
        )
        u = (effective_ratio - self.ratio_stop) / (self.ratio_slow - self.ratio_stop)
        unit_margin = float(np.clip(u, 0.0, 1.0))
        smoothstep = unit_margin * unit_margin * (3.0 - 2.0 * unit_margin)
        activation = float(smoothstep**self.exponent)
        self._allowed_rate = self.max_rate * activation
        if gradient is None:
            gradient = self._finite_difference_gradient(configuration)

        G = np.zeros((1, self.model.nv), dtype=np.float64)
        G[0, self.dof_indices] = -gradient
        self._G = G

    def can_share_perturbations(self, other: SingularityApproachLimit) -> bool:
        """Check both frames and roots for cross-arm kinematic dependencies."""
        if (
            self.model is not other.model
            or self.gradient_epsilon != other.gradient_epsilon
            or np.intersect1d(self.dof_indices, other.dof_indices).size
        ):
            return False
        for limit, other_dofs in (
            (self, other.dof_indices),
            (other, self.dof_indices),
        ):
            task = getattr(limit.frame_task, "frame_task", limit.frame_task)
            if not (
                frame_is_independent_of_dofs(
                    self.model, task.frame_name, task.frame_type, other_dofs
                )
                and relative_root_is_independent_of_dofs(
                    limit.frame_task, self.model, other_dofs
                )
            ):
                return False
        return True

    def _paired_gradients(
        self, configuration: mink.Configuration, other: SingularityApproachLimit
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a pair already validated by can_share_perturbations."""
        gradient = self._finite_difference_gradient(configuration, paired_limit=other)
        return gradient, other._gradient

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
        return singularity_ratio(self._jacobian(configuration))

    def _jacobian(
        self, configuration: mink.Configuration, *, out: np.ndarray | None = None
    ) -> np.ndarray:
        if self._root_is_independent_of_dofs:
            jacobian = self._world_jacobian
            self._jacobian_func(
                self.model,
                configuration.data,
                jacobian[:3],
                jacobian[3:],
                self._frame_id,
            )
            # Pure frame rotation preserves singular values, also after length scaling.
            if out is None:
                out = np.empty((6, self.dof_indices.size), dtype=np.float64)
            out[:] = jacobian[:, self._jacobian_columns]
            out[:3] /= self.characteristic_length
            return out
        jacobian = normalized_arm_jacobian(
            self.frame_task,
            configuration,
            self.dof_indices,
            self.characteristic_length,
            root_is_independent_of_dofs=self._root_is_independent_of_dofs,
        )
        if out is not None:
            out[:] = jacobian
            return out
        return jacobian

    def _finite_difference_gradient(
        self,
        configuration: mink.Configuration,
        *,
        paired_limit: SingularityApproachLimit | None = None,
    ) -> np.ndarray:
        q0 = configuration.q
        tangent = self._tangent
        tangent.fill(0.0)
        q = self._perturbed_q
        jacobians = self._perturbed_jacobians
        eps = self.gradient_epsilon

        for index, dof in enumerate(self.dof_indices):
            tangent[dof] = 1.0
            if paired_limit is not None:
                tangent[paired_limit.dof_indices[index]] = 1.0
            q[:] = q0
            mujoco.mj_integratePos(self.model, q, tangent, eps)
            self._update_scratch(q)
            self._jacobian(self._scratch, out=jacobians[index, 0])
            if paired_limit is not None:
                paired_limit._jacobian(
                    self._scratch, out=paired_limit._perturbed_jacobians[index, 0]
                )

            q[:] = q0
            mujoco.mj_integratePos(self.model, q, tangent, -eps)
            self._update_scratch(q)
            self._jacobian(self._scratch, out=jacobians[index, 1])
            tangent[dof] = 0.0
            if paired_limit is not None:
                paired_limit._jacobian(
                    self._scratch, out=paired_limit._perturbed_jacobians[index, 1]
                )
                tangent[paired_limit.dof_indices[index]] = 0.0

        gradient = self._gradient_from_samples()
        if paired_limit is not None:
            paired_limit._gradient_from_samples()
        return gradient

    def _gradient_from_samples(self) -> np.ndarray:
        # Batch the same positive/negative perturbations without changing the stencil.
        singular_values = np.linalg.svd(self._perturbed_jacobians, compute_uv=False)
        largest = singular_values[..., 0]
        ratios = np.divide(
            singular_values[..., -1],
            largest,
            out=np.zeros_like(largest),
            where=largest > 0.0,
        )
        np.subtract(ratios[:, 0], ratios[:, 1], out=self._gradient)
        self._gradient /= 2.0 * self.gradient_epsilon
        return self._gradient

    def _update_scratch(self, q: np.ndarray) -> None:
        # Ratio evaluation only needs frame transforms and geometric Jacobians.
        self._scratch.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self._scratch.data)
        mujoco.mj_comPos(self.model, self._scratch.data)
