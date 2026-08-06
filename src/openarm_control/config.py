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

"""Shared MuJoCo context for OpenArm FK, IK, and controller nodes."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco
from openarm_mujoco.v2 import JointResolver

_DEFAULT_XML = openarm_mujoco.openarm_cell_xml()

_DEFAULT_FRAME_RIGHT = "right_ee_control_point"
_DEFAULT_FRAME_TYPE_RIGHT = "site"
_DEFAULT_FRAME_LEFT = "left_ee_control_point"
_DEFAULT_FRAME_TYPE_LEFT = "site"

_DEFAULT_ORIGIN_FRAME = "arm_origin"
_DEFAULT_ORIGIN_FRAME_TYPE = "site"

# Sentinel --origin-frame value: no origin frame, poses stay world-relative.
WORLD_FRAME = "world"

_FRAME_OBJ = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "site": mujoco.mjtObj.mjOBJ_SITE,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
}

# Per-arm-joint velocity caps in rad/s. Enabled via --limit-velocity.
ARM_JOINT_VELOCITY_LIMITS_RAD_S: list[float] = [
    2.0,  # joint1 DM 8009
    2.0,  # joint2 DM 8009
    3.14,  # joint3 DM 4340
    3.14,  # joint4 DM 4340
    6.3,  # joint5 DM 4310
    6.3,  # joint6 DM 4310
    6.3,  # joint7 DM 4310
]


class ArmSetup:
    """MuJoCo context shared across FK, IK, and controller nodes.

    Bundles the model, data, joint resolver, active arm sides, and per-arm
    EE frame IDs/types. Instantiate once per process; pass into any solver
    or controller that needs model access.

    Pose convention throughout: float32[7] = [px, py, pz, qw, qx, qy, qz],
    expressed in the origin frame when one is set (default: the scene's
    'arm_origin' site), else in world coordinates.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_resolver: JointResolver,
        sides: list[str],
        frame_ids: dict[str, int],
        frame_types: dict[str, str],
        origin_id: int | None = None,
        origin_type: str = "site",
    ) -> None:
        """Initialize."""
        self.model = model
        self.data = data
        self.joint_resolver = joint_resolver
        self.sides = sides
        self.frame_ids = frame_ids  # side → MuJoCo object ID
        self.frame_types = frame_types  # side → "body" | "site" | "geom"
        self.origin_type = origin_type
        self.origin_id = origin_id  # None → poses stay world-relative

    @classmethod
    def from_args(
        cls,
        xml: str,
        mode: str,
        frame_right: str,
        frame_type_right: str,
        frame_left: str,
        frame_type_left: str,
        keyframe: str | None = "home",
        origin_frame: str = _DEFAULT_ORIGIN_FRAME,
        origin_frame_type: str = _DEFAULT_ORIGIN_FRAME_TYPE,
    ) -> ArmSetup:
        """Build from XML."""
        model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)

        if keyframe:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(model, data, key_id)
            else:
                print(f"Warning: keyframe '{keyframe}' not found, using defaults.")

        mujoco.mj_forward(model, data)

        sides: list[str] = []
        if mode in ("right", "bimanual"):
            sides.append("right")
        if mode in ("left", "bimanual"):
            sides.append("left")

        frame_ids: dict[str, int] = {}
        frame_types: dict[str, str] = {}
        for side in sides:
            name = frame_right if side == "right" else frame_left
            ftype = frame_type_right if side == "right" else frame_type_left
            frame_ids[side] = _resolve_frame_id(model, name, ftype)
            frame_types[side] = ftype

        origin_id = (
            None
            if origin_frame == WORLD_FRAME
            else _resolve_frame_id(model, origin_frame, origin_frame_type)
        )
        return cls(
            model=model,
            data=data,
            joint_resolver=JointResolver(model),
            sides=sides,
            frame_ids=frame_ids,
            frame_types=frame_types,
            origin_id=origin_id,
            origin_type=origin_frame_type,
        )

    def read_ee_pose(self, side: str) -> np.ndarray:
        """Return float32[7] = [px, py, pz, qw, qx, qy, qz] for the given arm.

        Expressed in the origin frame when one is set, else in world
        coordinates.
        """
        from openarm_control.geometry.poses import read_ee_pose, relative_pose

        pose = read_ee_pose(self.data, self.frame_ids[side], self.frame_types[side])
        if self.origin_id is None:
            return pose
        origin = read_ee_pose(self.data, self.origin_id, self.origin_type)
        return relative_pose(origin, pose)

    def driver_qpos_to_mujoco(
        self,
        qpos16: np.ndarray,
        *,
        base_qpos: np.ndarray,
    ) -> np.ndarray:
        """Map bimanual driver qpos into a full MuJoCo configuration."""
        driver_qpos = np.asarray(qpos16, dtype=np.float64)
        model_qpos = np.asarray(base_qpos, dtype=np.float64)
        if driver_qpos.shape != (16,):
            raise ValueError("Measured bimanual qpos must have shape (16,).")
        if model_qpos.shape != (self.model.nq,):
            raise ValueError(f"Base qpos must have shape ({self.model.nq},).")
        if not np.all(np.isfinite(driver_qpos)) or not np.all(np.isfinite(model_qpos)):
            raise ValueError("Measured driver qpos and base qpos must be finite.")

        model_qpos = model_qpos.copy()
        for side, offset in (("right", 0), ("left", 8)):
            if side not in self.sides:
                continue
            self.joint_resolver.set_qpos(
                model_qpos,
                driver_qpos[offset : offset + 8],
                side,
            )
        return model_qpos


def _resolve_frame_id(model: mujoco.MjModel, name: str, ftype: str) -> int:
    obj = _frame_obj(ftype)
    fid = mujoco.mj_name2id(model, obj, name)
    if fid < 0:
        raise ValueError(f"{ftype.capitalize()} '{name}' not found in model.")
    return fid


def frame_name(model: mujoco.MjModel, fid: int, ftype: str) -> str:
    """Return the model name of a frame, given its MuJoCo object ID and type.

    Frame IDs are what MjData is indexed by, so they are what ArmSetup
    stores; name-based APIs (mink tasks) go through here.
    """
    return mujoco.mj_id2name(model, _frame_obj(ftype), fid)


def _frame_obj(ftype: str) -> mujoco.mjtObj:
    obj = _FRAME_OBJ.get(ftype)
    if obj is None:
        raise ValueError(f"Unknown frame_type '{ftype}'. Expected body/site/geom.")
    return obj


def register_common_args(parser: argparse.ArgumentParser) -> None:
    """Register shared CLI flags used by all arm nodes: --xml, --keyframe, --mode, --frame-*."""
    parser.add_argument(
        "--xml",
        default=_DEFAULT_XML,
        help=f"MJCF scene file (default: {_DEFAULT_XML})",
    )
    parser.add_argument(
        "--keyframe",
        "-k",
        default="home",
        help="Initial keyframe name (default: home)",
    )
    parser.add_argument(
        "--mode",
        choices=["right", "left", "bimanual"],
        default="bimanual",
        help="Which arm(s) to compute (default: bimanual)",
    )
    parser.add_argument(
        "--frame-right",
        default=_DEFAULT_FRAME_RIGHT,
        help=f"EE frame name for right arm (default: {_DEFAULT_FRAME_RIGHT})",
    )
    parser.add_argument(
        "--frame-type-right",
        choices=["body", "site", "geom"],
        default=_DEFAULT_FRAME_TYPE_RIGHT,
        help="EE frame type for right arm (default: site)",
    )
    parser.add_argument(
        "--frame-left",
        default=_DEFAULT_FRAME_LEFT,
        help=f"EE frame name for left arm (default: {_DEFAULT_FRAME_LEFT})",
    )
    parser.add_argument(
        "--frame-type-left",
        choices=["body", "site", "geom"],
        default=_DEFAULT_FRAME_TYPE_LEFT,
        help="EE frame type for left arm (default: site)",
    )
    parser.add_argument(
        "--origin-frame",
        default=_DEFAULT_ORIGIN_FRAME,
        help=(
            "Frame whose pose is the origin of all FK/IK poses "
            f"(default: {_DEFAULT_ORIGIN_FRAME}). "
            f"Pass '{WORLD_FRAME}' for raw world-frame poses."
        ),
    )
    parser.add_argument(
        "--origin-frame-type",
        choices=["body", "site", "geom"],
        default=_DEFAULT_ORIGIN_FRAME_TYPE,
        help="Origin frame type (default: site)",
    )


def setup_from_args(args: argparse.Namespace) -> ArmSetup:
    """Build ArmSetup from a namespace that contains the common CLI flags."""
    return ArmSetup.from_args(
        xml=args.xml,
        mode=args.mode,
        frame_right=args.frame_right,
        frame_type_right=args.frame_type_right,
        frame_left=args.frame_left,
        frame_type_left=args.frame_type_left,
        keyframe=args.keyframe,
        origin_frame=args.origin_frame,
        origin_frame_type=args.origin_frame_type,
    )
