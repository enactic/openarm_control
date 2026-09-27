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

"""One-dimensional nullspace posture regularization for a 7-DoF arm."""

from __future__ import annotations

import math

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from openarm_control.geometry.jacobian import (
    CachedConfiguration,
    normalized_arm_jacobian,
    relative_root_is_independent_of_dofs,
)


def smoothstep_activation(value: float, low: float, high: float) -> float:
    """Map ``value`` to [0, 1] with zero slope at both thresholds."""
    if not 0.0 <= low < high:
        raise ValueError("Expected 0 <= low < high.")
    u = float((value - low) / (high - low))
    u = 0.0 if u < 0.0 else 1.0 if u > 1.0 else u
    return u * u * (3.0 - 2.0 * u)


def structural_nullspace_direction(
    jacobian: npt.NDArray[np.floating],
    previous: npt.NDArray[np.floating] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the structural 1D nullspace direction of a full-rank 6x7 Jacobian."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape != (6, 7):
        raise ValueError(f"Expected a 6x7 Jacobian, got {jacobian.shape}.")

    _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=True)
    return _align_direction(vh[-1], previous), singular_values


def _align_direction(direction: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    direction = direction.copy()
    if previous is not None:
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != (7,):
            raise ValueError(
                f"Expected previous direction shape (7,), got {previous.shape}."
            )
        if float(direction @ previous) < 0.0:
            direction = -direction
    return direction


class NullspacePostureTask(mink.Task):
    """Return a 7-DoF arm toward home only along its structural nullspace."""

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_task: mink.Task,
        dof_indices: npt.ArrayLike,
        home_qpos: npt.ArrayLike,
        *,
        cost: float,
        dt: float,
        return_rate: float,
        max_speed: float,
        singularity_low: float,
        singularity_high: float,
        characteristic_length: float,
    ) -> None:
        """Initialize the task with a fixed home and physical-rate parameters."""
        if cost < 0.0:
            raise ValueError("cost must be non-negative.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if return_rate < 0.0:
            raise ValueError("return_rate must be non-negative.")
        if max_speed < 0.0:
            raise ValueError("max_speed must be non-negative.")
        if characteristic_length <= 0.0:
            raise ValueError("characteristic_length must be positive.")
        if not 0.0 <= singularity_low < singularity_high:
            raise ValueError("Expected 0 <= singularity_low < singularity_high.")

        indices = np.asarray(dof_indices, dtype=int)
        if indices.shape != (7,):
            raise ValueError(f"Expected seven arm DoF indices, got {indices.shape}.")
        if np.unique(indices).size != indices.size:
            raise ValueError("Arm DoF indices must be unique.")
        if np.any(indices < 0) or np.any(indices >= model.nv):
            raise ValueError("Arm DoF indices are outside the model tangent space.")

        home_qpos = np.asarray(home_qpos, dtype=np.float64)
        if home_qpos.shape != (model.nq,):
            raise ValueError(
                f"Expected home_qpos shape ({model.nq},), got {home_qpos.shape}."
            )

        super().__init__(cost=np.array([cost], dtype=np.float64))
        self._base_cost = cost
        self._model = model
        self._frame_task = frame_task
        self._dof_indices = indices.copy()
        self._home_qpos = home_qpos.copy()
        self._dt = dt
        self._return_rate = return_rate
        self._max_speed = max_speed
        self._singularity_low = singularity_low
        self._singularity_high = singularity_high
        self._characteristic_length = characteristic_length
        self._root_is_independent_of_dofs = relative_root_is_independent_of_dofs(
            frame_task,
            model,
            indices,
        )
        self._identity = np.eye(model.nv, dtype=np.float64)
        self._configuration_error = np.empty(model.nv, dtype=np.float64)
        self._previous_direction: np.ndarray | None = None
        self._svd_partner: NullspacePostureTask | None = None
        self._svd_batch: np.ndarray | None = None

    def _batch_svd_with(self, other: NullspacePostureTask) -> None:
        """Let this task populate both spectra for a fixed pair in one batch."""
        self._svd_partner = other
        self._svd_batch = np.empty((2, 6, 7), dtype=np.float64)

    def _svd(self, configuration: mink.Configuration) -> tuple[np.ndarray, np.ndarray]:
        cache = (
            configuration._arm_svds
            if isinstance(configuration, CachedConfiguration)
            else None
        )
        cached = cache.get(self) if cache is not None else None
        if cached is not None:
            return cached
        if (
            cache is not None
            and self._svd_partner is not None
            and self._svd_partner not in cache
        ):
            assert self._svd_batch is not None
            tasks = (self, self._svd_partner)
            for index, task in enumerate(tasks):
                self._svd_batch[index] = task._jacobian(configuration)
            _, singular_values, vh = np.linalg.svd(self._svd_batch, full_matrices=True)
            singular_values.setflags(write=False)
            vh.setflags(write=False)
            for index, task in enumerate(tasks):
                cache[task] = (singular_values[index], vh[index, -1])
            return cache[self]
        _, singular_values, vh = np.linalg.svd(
            self._jacobian(configuration), full_matrices=True
        )
        direction = vh[-1]
        singular_values.setflags(write=False)
        direction.setflags(write=False)
        if cache is not None:
            cache[self] = (singular_values, direction)
        return singular_values, direction

    def _jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        return normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._dof_indices,
            self._characteristic_length,
            root_is_independent_of_dofs=self._root_is_independent_of_dofs,
        )

    def compute_singularity_ratio(self, configuration: mink.Configuration) -> float:
        """Expose the shared spectrum without advancing direction-continuity state."""
        singular_values, _ = self._svd(configuration)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        return float(singular_values[-1] / largest) if largest > 0.0 else 0.0

    def _compute_terms(
        self, configuration: mink.Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        singular_values, direction = self._svd(configuration)
        direction = _align_direction(direction, self._previous_direction)
        self._previous_direction = direction

        largest = float(singular_values[0]) if singular_values.size else 0.0
        ratio = float(singular_values[-1] / largest) if largest > 0.0 else 0.0
        activation = smoothstep_activation(
            ratio, self._singularity_low, self._singularity_high
        )

        configuration_error = self._configuration_error
        mujoco.mj_differentiatePos(
            m=self._model,
            qvel=configuration_error,
            dt=1.0,
            qpos1=self._home_qpos,
            qpos2=configuration.data.qpos,
        )
        posture_error = float(direction @ configuration_error[self._dof_indices])
        return_speed = -self._return_rate * posture_error
        if return_speed < -self._max_speed:
            return_speed = -self._max_speed
        elif return_speed > self._max_speed:
            return_speed = self._max_speed
        displacement = return_speed * self._dt

        effective_cost = math.sqrt(activation) * self._base_cost
        self.cost[0] = effective_cost
        jacobian = np.zeros((1, self._model.nv), dtype=np.float64)
        jacobian[0, self._dof_indices] = direction
        error = np.array([-displacement], dtype=np.float64)

        return error, jacobian

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the one-step nullspace displacement error."""
        error, _ = self._compute_terms(configuration)
        return error

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the nullspace-coordinate Jacobian."""
        _, jacobian = self._compute_terms(configuration)
        return jacobian

    def compute_qp_objective(self, configuration: mink.Configuration) -> mink.Objective:
        """Assemble both terms from one SVD so they use the same direction."""
        error, jacobian = self._compute_terms(configuration)
        return self._assemble_qp(error, jacobian, self._identity)

    def compute_qp_residual(
        self,
        configuration: mink.Configuration,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute the fused residual from a single nullspace SVD."""
        error, jacobian = self._compute_terms(configuration)
        return self._weighted_residual(error, jacobian)
