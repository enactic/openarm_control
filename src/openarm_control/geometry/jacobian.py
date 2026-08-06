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

"""Arm Jacobian geometry and metrics shared by IK components."""

from __future__ import annotations

import mink
import mujoco
import numpy as np
import numpy.typing as npt


def relative_root_is_independent_of_dofs(
    frame_task: mink.Task,
    model: mujoco.MjModel,
    dof_indices: npt.ArrayLike,
) -> bool:
    """Return whether a relative task's root is unaffected by selected DoFs."""
    native_task = getattr(frame_task, "frame_task", frame_task)
    if isinstance(native_task, mink.FrameTask):
        return True
    if not isinstance(native_task, mink.RelativeFrameTask):
        raise TypeError("Expected a Mink frame task or a frame-task wrapper.")

    root_id = mujoco.mj_name2id(
        model,
        {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }[native_task.root_type],
        native_task.root_name,
    )
    if root_id < 0:
        raise ValueError(
            f"Unknown {native_task.root_type} frame {native_task.root_name!r}."
        )
    if native_task.root_type == "body":
        body_id = root_id
    elif native_task.root_type == "site":
        body_id = int(model.site_bodyid[root_id])
    else:
        body_id = int(model.geom_bodyid[root_id])

    selected_dofs = set(np.asarray(dof_indices, dtype=int).tolist())
    while body_id > 0:
        dof_start = int(model.body_dofadr[body_id])
        dof_count = int(model.body_dofnum[body_id])
        if any(dof in selected_dofs for dof in range(dof_start, dof_start + dof_count)):
            return False
        body_id = int(model.body_parentid[body_id])
    return True


def normalized_arm_jacobian(
    frame_task: mink.Task,
    configuration: mink.Configuration,
    dof_indices: npt.ArrayLike,
    characteristic_length: float,
    *,
    root_is_independent_of_dofs: bool = False,
) -> np.ndarray:
    """Return a dimensionless geometric 6-by-arm-DoF frame Jacobian."""
    if characteristic_length <= 0.0:
        raise ValueError("Characteristic length must be positive.")
    native_task = getattr(frame_task, "frame_task", frame_task)
    if not isinstance(native_task, (mink.FrameTask, mink.RelativeFrameTask)):
        raise TypeError("Expected a Mink frame task or a frame-task wrapper.")
    indices = np.asarray(dof_indices, dtype=int)
    jacobian = configuration.get_frame_jacobian(
        native_task.frame_name,
        native_task.frame_type,
    )[:, indices].copy()
    if (
        isinstance(native_task, mink.RelativeFrameTask)
        and not root_is_independent_of_dofs
    ):
        root_jacobian = configuration.get_frame_jacobian(
            native_task.root_name,
            native_task.root_type,
        )[:, indices]
        transform_frame_to_root = configuration.get_transform(
            native_task.frame_name,
            native_task.frame_type,
            native_task.root_name,
            native_task.root_type,
        )
        jacobian = (
            jacobian - transform_frame_to_root.inverse().adjoint() @ root_jacobian
        )
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
