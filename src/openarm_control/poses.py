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

"""Pose conversion utilities shared across FK, IK, and controller nodes.

Convention: float32[7] = [px, py, pz, qw, qx, qy, qz]
"""

from __future__ import annotations

import mujoco
import numpy as np


def read_ee_pose(data: mujoco.MjData, fid: int, ftype: str) -> np.ndarray:
    """Read EE pose from MjData after mj_forward.

    Returns float32[7] = [px, py, pz, qw, qx, qy, qz].
    """
    if ftype == "body":
        pos = data.xpos[fid]
        quat = data.xquat[fid]
    elif ftype == "site":
        pos = data.site_xpos[fid]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[fid])
    else:  # geom
        pos = data.geom_xpos[fid]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.geom_xmat[fid])
    return np.concatenate([pos, quat]).astype(np.float32)


def pose_to_se3(pose: np.ndarray):  # -> mink.SE3
    """float32[7] [px, py, pz, qw, qx, qy, qz] → mink.SE3.

    mink internal layout: wxyz_xyz = [qw, qx, qy, qz, x, y, z].
    """
    import mink  # lazy: FK nodes don't depend on mink

    wxyz_xyz = np.empty(7, dtype=np.float64)
    wxyz_xyz[:4] = pose[3:7]  # quat: wxyz
    wxyz_xyz[4:] = pose[:3]  # translation: xyz
    return mink.SE3(wxyz_xyz=wxyz_xyz)


def se3_to_pose(se3) -> np.ndarray:  # se3: mink.SE3
    """mink.SE3 → float32[7] [px, py, pz, qw, qx, qy, qz]."""
    wxyz_xyz = se3.wxyz_xyz
    return np.concatenate([wxyz_xyz[4:], wxyz_xyz[:4]]).astype(np.float32)
