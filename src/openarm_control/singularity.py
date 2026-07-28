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

"""Jacobian normalization and singularity metrics shared by IK components."""

from __future__ import annotations

import mink
import numpy as np
import numpy.typing as npt


def normalized_arm_jacobian(
    frame_task: mink.FrameTask | mink.RelativeFrameTask,
    configuration: mink.Configuration,
    dof_indices: npt.ArrayLike,
    characteristic_length: float,
) -> np.ndarray:
    """Return a dimensionless geometric 6-by-arm-DoF frame Jacobian."""
    if characteristic_length <= 0.0:
        raise ValueError("Characteristic length must be positive.")
    indices = np.asarray(dof_indices, dtype=int)
    jacobian = configuration.get_frame_jacobian(
        frame_task.frame_name,
        frame_task.frame_type,
    )
    if isinstance(frame_task, mink.RelativeFrameTask):
        root_jacobian = configuration.get_frame_jacobian(
            frame_task.root_name,
            frame_task.root_type,
        )
        transform_frame_to_root = configuration.get_transform(
            frame_task.frame_name,
            frame_task.frame_type,
            frame_task.root_name,
            frame_task.root_type,
        )
        jacobian = (
            jacobian - transform_frame_to_root.inverse().adjoint() @ root_jacobian
        )
    jacobian = jacobian[:, indices].copy()
    jacobian[:3] /= characteristic_length
    return jacobian


def singularity_ratio(jacobian: npt.ArrayLike) -> tuple[float, np.ndarray]:
    """Return sigma_min / sigma_max and the Jacobian singular values."""
    singular_values = np.linalg.svd(
        np.asarray(jacobian, dtype=np.float64),
        compute_uv=False,
    )
    largest = float(singular_values[0]) if singular_values.size else 0.0
    ratio = float(singular_values[-1] / largest) if largest > 0.0 else 0.0
    return ratio, singular_values
