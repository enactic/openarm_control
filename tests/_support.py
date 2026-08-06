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

"""Shared MuJoCo fixtures for OpenArm control tests."""

from __future__ import annotations

import numpy as np
import openarm_mujoco.v2 as openarm_mujoco

from openarm_control import ArmSetup
from openarm_control.config import (
    ARM_JOINT_VELOCITY_LIMITS_RAD_S,
    WORLD_FRAME,
)


def make_setup(
    mode: str = "bimanual",
    *,
    origin_frame: str = WORLD_FRAME,
    origin_frame_type: str = "site",
) -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
        origin_frame=origin_frame,
        origin_frame_type=origin_frame_type,
    )


def driver_state(setup: ArmSetup) -> np.ndarray:
    values = []
    for side in ("right", "left"):
        joints, gripper = setup.joint_resolver.get_driver(setup.data.qpos, side)
        values.append(np.append(joints, gripper))
    return np.concatenate(values).astype(np.float32)


def velocity_mapping(*sides: str) -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": value
        for side in sides
        for index, value in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    }
