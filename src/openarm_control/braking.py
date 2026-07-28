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

"""Shared approach-speed envelopes for preventive IK limits."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def powered_smoothstep(
    value: npt.ArrayLike,
    *,
    exponent: float = 1.0,
) -> np.ndarray:
    """Map a normalized margin to [0, 1] with configurable early braking."""
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("Smoothstep exponent must be finite and positive.")
    u = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    smooth = u * u * (3.0 - 2.0 * u)
    return np.power(smooth, exponent)


def distance_velocity_envelope(
    distance: npt.ArrayLike,
    max_velocity: npt.ArrayLike,
    slowdown_distance: npt.ArrayLike,
    *,
    exponent: float = 1.0,
) -> np.ndarray:
    """Return maximum approach velocity as a function of remaining distance."""
    distance_array = np.asarray(distance, dtype=np.float64)
    max_velocity_array = np.asarray(max_velocity, dtype=np.float64)
    slowdown_array = np.asarray(slowdown_distance, dtype=np.float64)
    if np.any(~np.isfinite(max_velocity_array)) or np.any(max_velocity_array <= 0.0):
        raise ValueError("Maximum velocities must be finite and positive.")
    if np.any(~np.isfinite(slowdown_array)) or np.any(slowdown_array <= 0.0):
        raise ValueError("Slowdown distances must be finite and positive.")
    activation = powered_smoothstep(
        np.maximum(distance_array, 0.0) / slowdown_array,
        exponent=exponent,
    )
    return max_velocity_array * activation
