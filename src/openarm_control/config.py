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
import openarm_mujoco_v2 as openarm_mujoco
from openarm_mujoco_v2 import JointResolver

_DEFAULT_XML = openarm_mujoco.openarm_cell_xml()

_DEFAULT_FRAME_RIGHT = "right_ee_control_point"
_DEFAULT_FRAME_TYPE_RIGHT = "site"
_DEFAULT_FRAME_LEFT = "left_ee_control_point"
_DEFAULT_FRAME_TYPE_LEFT = "site"

_FRAME_OBJ = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "site": mujoco.mjtObj.mjOBJ_SITE,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
}

# Per-arm-joint velocity caps in rad/s at --vel-scale=1.0
ARM_JOINT_VELOCITY_LIMITS_RAD_S: list[float] = [
    1.57,  # joint1 DM 8009
    1.57,  # joint2 DM 8009
    3.14,  # joint3 DM 4340
    3.14,  # joint4 DM 4340
    12.6,  # joint5 DM 4310
    12.6,  # joint6 DM 4310
    12.6,  # joint7 DM 4310
]


class ArmSetup:
    """MuJoCo context shared across FK, IK, and controller nodes.

    Bundles the model, data, joint resolver, active arm sides, and per-arm
    EE frame IDs/types. Instantiate once per process; pass into any solver
    or controller that needs model access.

    Pose convention throughout: float32[7] = [px, py, pz, qw, qx, qy, qz]
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_resolver: JointResolver,
        sides: list[str],
        frame_ids: dict[str, int],
        frame_types: dict[str, str],
    ) -> None:
        self.model = model
        self.data = data
        self.joint_resolver = joint_resolver
        self.sides = sides
        self.frame_ids = frame_ids      # side → MuJoCo object ID
        self.frame_types = frame_types  # side → "body" | "site" | "geom"

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
    ) -> ArmSetup:
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

        return cls(
            model=model,
            data=data,
            joint_resolver=JointResolver(model),
            sides=sides,
            frame_ids=frame_ids,
            frame_types=frame_types,
        )

    def read_ee_pose(self, side: str) -> "np.ndarray":
        """Return float32[7] = [px, py, pz, qw, qx, qy, qz] for the given arm."""
        from openarm_control.poses import read_ee_pose
        return read_ee_pose(self.data, self.frame_ids[side], self.frame_types[side])


def _resolve_frame_id(model: mujoco.MjModel, name: str, ftype: str) -> int:
    obj = _FRAME_OBJ.get(ftype)
    if obj is None:
        raise ValueError(f"Unknown frame_type '{ftype}'. Expected body/site/geom.")
    fid = mujoco.mj_name2id(model, obj, name)
    if fid < 0:
        raise ValueError(f"{ftype.capitalize()} '{name}' not found in model.")
    return fid


def register_common_args(parser: argparse.ArgumentParser) -> None:
    """Register shared CLI flags used by all arm nodes: --xml, --keyframe, --mode, --frame-*."""
    parser.add_argument(
        "--xml",
        default=_DEFAULT_XML,
        help=f"MJCF scene file (default: {_DEFAULT_XML})",
    )
    parser.add_argument(
        "--keyframe", "-k",
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
    )
