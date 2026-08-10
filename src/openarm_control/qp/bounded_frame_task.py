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

"""Mink frame-task wrapper that bounds its error request."""

from __future__ import annotations

import mink
import numpy as np


class BoundedFrameTask(mink.Task):
    """Bound a native frame task's first-order request without changing its target."""

    def __init__(
        self,
        frame_task: mink.FrameTask | mink.RelativeFrameTask,
        *,
        position_error_limit: float,
        orientation_error_limit: float = 0.0,
        control_dt: float,
        substeps: int,
        target_linear_speed_slow: float,
        target_linear_speed_fast: float,
        position_error_latch_threshold: float,
    ) -> None:
        """Wrap either a world-frame or relative-frame Mink task."""
        if not isinstance(frame_task, (mink.FrameTask, mink.RelativeFrameTask)):
            raise TypeError("frame_task must be a Mink frame task.")
        for name, value in (
            ("position_error_limit", position_error_limit),
            ("orientation_error_limit", orientation_error_limit),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not np.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("control_dt must be finite and positive.")
        if (
            not isinstance(substeps, (int, np.integer))
            or isinstance(substeps, (bool, np.bool_))
            or substeps <= 0
        ):
            raise ValueError("substeps must be a positive integer.")
        if not np.isfinite(target_linear_speed_slow) or target_linear_speed_slow < 0.0:
            raise ValueError(
                "target_linear_speed_slow must be finite and non-negative."
            )
        if (
            not np.isfinite(target_linear_speed_fast)
            or target_linear_speed_fast <= target_linear_speed_slow
        ):
            raise ValueError(
                "target_linear_speed_fast must be finite and greater than "
                "target_linear_speed_slow."
            )
        if (
            not np.isfinite(position_error_latch_threshold)
            or position_error_latch_threshold <= 0.0
        ):
            raise ValueError(
                "position_error_latch_threshold must be finite and positive."
            )

        super().__init__(
            cost=frame_task.cost.copy(),
            gain=frame_task.gain,
            lm_damping=frame_task.lm_damping,
        )
        self.frame_task = frame_task
        self.position_error_limit = float(position_error_limit)
        self.orientation_error_limit = float(orientation_error_limit)
        self._control_dt = float(control_dt)
        self._substeps = int(substeps)
        self._target_linear_speed_slow = float(target_linear_speed_slow)
        self._target_linear_speed_fast = float(target_linear_speed_fast)
        self._position_error_latch_threshold = float(position_error_latch_threshold)
        self._previous_target_position: np.ndarray | None = None
        self._identity: np.ndarray | None = None
        self.limit_activation = 1.0

    def set_target(self, transform: mink.SE3) -> None:
        """Set the target on the wrapped native task."""
        self.frame_task.set_target(transform)

    def set_target_and_update_schedule(
        self,
        transform: mink.SE3,
        configuration: mink.Configuration,
    ) -> None:
        """Set a target and schedule error limiting from speed and accumulated lag."""
        self.set_target(transform)
        position = transform.translation()
        previous = self._previous_target_position
        speed = (
            0.0
            if previous is None
            else float(np.linalg.norm(position - previous)) / self._control_dt
        )
        self._previous_target_position = position.copy()

        u = np.clip(
            (speed - self._target_linear_speed_slow)
            / (self._target_linear_speed_fast - self._target_linear_speed_slow),
            0.0,
            1.0,
        )
        activation = float(u * u * (3.0 - 2.0 * u))
        if self.position_error_limit > 0.0 and self.limit_activation > 0.0:
            full_error = self.compute_full_error(configuration)
            if (
                float(np.linalg.norm(full_error[:3]))
                > self._position_error_latch_threshold
            ):
                activation = max(activation, self.limit_activation)
        self.set_limit_activation(activation)

    def set_target_from_configuration(self, configuration: mink.Configuration) -> None:
        """Set the wrapped target from a configuration."""
        self.frame_task.set_target_from_configuration(configuration)

    def set_limit_activation(self, activation: float) -> None:
        """Blend continuously between the full and bounded error requests."""
        if not np.isfinite(activation) or not 0.0 <= activation <= 1.0:
            raise ValueError("Frame error-limit activation must be in [0, 1].")
        self.limit_activation = float(activation)

    def compute_full_error(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the native task error."""
        return self.frame_task.compute_error(configuration)

    def compute_limited_error(self, configuration: mink.Configuration) -> np.ndarray:
        """Return one substep of the total position and orientation budgets."""
        return self._clip_error(self.compute_full_error(configuration))

    def _clip_error(self, full_error: np.ndarray) -> np.ndarray:
        """Limit a previously computed frame error."""
        error = full_error.copy()
        for part, total_limit in (
            (slice(0, 3), self.position_error_limit),
            (slice(3, 6), self.orientation_error_limit),
        ):
            limit = total_limit / self._substeps
            norm = float(np.linalg.norm(error[part]))
            if limit > 0.0 and norm > limit:
                error[part] *= limit / norm
        return error

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the full error, matching the wrapped Mink task API."""
        return self.compute_full_error(configuration)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the wrapped task Jacobian."""
        return self.frame_task.compute_jacobian(configuration)

    def compute_qp_residual(
        self,
        configuration: mink.Configuration,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Opt out of Mink's fused residual path so bounding is not bypassed.

        Mink 1.2's solver prefers a task's residual and only falls back to
        :meth:`compute_qp_objective` when it is ``None``. The inherited
        ``mink.Task`` residual is built from :meth:`compute_error`, which
        deliberately reports the *full* error, so accepting it would silently
        drop the bounding this class exists to apply.
        """
        return None

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        """Modulate position by its schedule and always bound orientation."""
        bound_position = self.position_error_limit > 0.0 and self.limit_activation > 0.0
        bound_orientation = self.orientation_error_limit > 0.0
        if not bound_position and not bound_orientation:
            return self.frame_task.compute_qp_objective(configuration)

        full_error = self.compute_full_error(configuration)
        limited_error = self._clip_error(full_error)
        error = full_error.copy()
        if bound_position:
            error[:3] += self.limit_activation * (limited_error[:3] - full_error[:3])
        if bound_orientation:
            error[3:] = limited_error[3:]
        nv = configuration.model.nv
        if self._identity is None or self._identity.shape != (nv, nv):
            self._identity = np.eye(nv, dtype=np.float64)
        return self._assemble_qp(
            error,
            self.compute_jacobian(configuration),
            self._identity,
        )
