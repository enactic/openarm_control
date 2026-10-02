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


class CachedConfiguration(mink.Configuration):
    """Cache read-only geometric quantities until the next configuration update."""

    def __init__(self, model: mujoco.MjModel, q: np.ndarray | None = None) -> None:
        """Initialize caches before Mink's constructor calls update."""
        self._frame_jacobians: dict[tuple[str | int, str], np.ndarray] = {}
        self._arm_svds: dict[mink.Task, tuple[np.ndarray, np.ndarray]] = {}
        super().__init__(model, q=q)

    def update(self, q: np.ndarray | None = None) -> None:
        """Invalidate geometry whenever forward kinematics is updated."""
        self._frame_jacobians.clear()
        self._arm_svds.clear()
        super().update(q=q)

    def get_frame_jacobian(self, frame_name: str | int, frame_type: str) -> np.ndarray:
        """Return a read-only body Jacobian shared by tasks at this configuration."""
        key = (frame_name, frame_type)
        jacobian = self._frame_jacobians.get(key)
        if jacobian is None:
            jacobian = super().get_frame_jacobian(frame_name, frame_type)
            jacobian.setflags(write=False)
            self._frame_jacobians[key] = jacobian
        return jacobian


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

    return frame_is_independent_of_dofs(
        model, native_task.root_name, native_task.root_type, dof_indices
    )


def frame_is_independent_of_dofs(
    model: mujoco.MjModel,
    frame_name: str | int,
    frame_type: str,
    dof_indices: npt.ArrayLike,
) -> bool:
    """Return whether none of the selected DoFs is an ancestor of the frame."""
    frame_id = (
        mujoco.mj_name2id(
            model,
            {
                "body": mujoco.mjtObj.mjOBJ_BODY,
                "site": mujoco.mjtObj.mjOBJ_SITE,
                "geom": mujoco.mjtObj.mjOBJ_GEOM,
            }[frame_type],
            frame_name,
        )
        if isinstance(frame_name, str)
        else int(frame_name)
    )
    if frame_id < 0:
        raise ValueError(f"Unknown {frame_type} frame {frame_name!r}.")
    if frame_type == "body":
        body_id = frame_id
    elif frame_type == "site":
        body_id = int(model.site_bodyid[frame_id])
    else:
        body_id = int(model.geom_bodyid[frame_id])

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
