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

"""Frame tasks with a bounded first-order error request."""

from __future__ import annotations

import mink
import numpy as np


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(output))
    if limit > 0.0 and norm > limit:
        output *= limit / norm
    return output


class _ErrorLimitMixin:
    """Shared independent position and orientation error limiting."""

    position_error_limit: float
    orientation_error_limit: float
    limit_activation: float

    def _set_error_limits(
        self,
        position_error_limit: float,
        orientation_error_limit: float,
    ) -> None:
        if not np.isfinite(position_error_limit) or position_error_limit < 0.0:
            raise ValueError(
                "Frame position_error_limit must be finite and non-negative."
            )
        if not np.isfinite(orientation_error_limit) or orientation_error_limit < 0.0:
            raise ValueError(
                "Frame orientation_error_limit must be finite and non-negative."
            )
        self.position_error_limit = float(position_error_limit)
        self.orientation_error_limit = float(orientation_error_limit)
        self.limit_activation = 1.0

    def set_limit_activation(self, activation: float) -> None:
        """Blend the bounded request with the native full task error."""
        if not np.isfinite(activation) or not 0.0 <= activation <= 1.0:
            raise ValueError("Frame error-limit activation must be in [0, 1].")
        self.limit_activation = float(activation)

    def _limit_error(self, error: np.ndarray) -> np.ndarray:
        limited_position = _limit_norm(error[:3], self.position_error_limit)
        limited_orientation = _limit_norm(
            error[3:],
            self.orientation_error_limit,
        )
        error[:3] += self.limit_activation * (limited_position - error[:3])
        error[3:] += self.limit_activation * (limited_orientation - error[3:])
        return error

    def _limit_disabled(self) -> bool:
        return self.limit_activation == 0.0 or (
            self.position_error_limit == 0.0 and self.orientation_error_limit == 0.0
        )


class ErrorLimitedFrameTask(_ErrorLimitMixin, mink.FrameTask):
    """Limit the pose error presented to one world-frame Mink task.

    The full target remains stored by :class:`mink.FrameTask`. Only the
    first-order error in ``J Delta q = -gain * error`` is norm-limited, so
    later control cycles continue closing any tracking lag. A zero limit keeps
    that part of the error unchanged.
    """

    def __init__(
        self,
        *,
        frame_name: str,
        frame_type: str,
        position_cost: np.ndarray | float,
        orientation_cost: np.ndarray | float,
        position_error_limit: float,
        orientation_error_limit: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ) -> None:
        """Configure independent translational and rotational error bounds."""
        mink.FrameTask.__init__(
            self,
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self._set_error_limits(position_error_limit, orientation_error_limit)

    def compute_full_error(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        """Return the native unbounded world-frame task error."""
        return mink.FrameTask.compute_error(self, configuration)

    def compute_limited_error(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        """Return the frame error after independent SE(3) norm limits."""
        return self._limit_error(self.compute_full_error(configuration))

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        """Assemble the normal Mink objective with the bounded error request."""
        if self._limit_disabled():
            return mink.FrameTask.compute_qp_objective(self, configuration)
        error = self.compute_limited_error(configuration)
        jacobian = mink.FrameTask.compute_jacobian(self, configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)


class ErrorLimitedRelativeFrameTask(_ErrorLimitMixin, mink.RelativeFrameTask):
    """Limit the pose error presented to one relative-frame Mink task."""

    def __init__(
        self,
        *,
        frame_name: str,
        frame_type: str,
        root_name: str,
        root_type: str,
        position_cost: np.ndarray | float,
        orientation_cost: np.ndarray | float,
        position_error_limit: float,
        orientation_error_limit: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ) -> None:
        """Configure independent translational and rotational error bounds."""
        mink.RelativeFrameTask.__init__(
            self,
            frame_name=frame_name,
            frame_type=frame_type,
            root_name=root_name,
            root_type=root_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self._set_error_limits(position_error_limit, orientation_error_limit)

    def compute_full_error(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        """Return the native unbounded relative-frame task error."""
        return mink.RelativeFrameTask.compute_error(self, configuration)

    def compute_limited_error(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        """Return the relative-frame error after independent SE(3) norm limits."""
        return self._limit_error(self.compute_full_error(configuration))

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        """Assemble the relative Mink objective with the bounded error request."""
        if self._limit_disabled():
            return mink.RelativeFrameTask.compute_qp_objective(self, configuration)
        error = self.compute_limited_error(configuration)
        jacobian = mink.RelativeFrameTask.compute_jacobian(self, configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)
