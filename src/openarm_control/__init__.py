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

"""Kinematics and control utilities for OpenArm."""

from .config import ArmSetup, register_common_args, setup_from_args
from .kinematics import (
    IKParams,
    Kinematics,
    register_ik_args,
    ik_params_from_args,
)
from .poses import read_ee_pose, pose_to_se3, se3_to_pose

__all__ = [
    # context
    "ArmSetup",
    # high-level interface
    "Kinematics",
    "IKParams",
    # CLI helpers
    "register_common_args",
    "register_ik_args",
    "setup_from_args",
    "ik_params_from_args",
    # pose utilities
    "read_ee_pose",
    "pose_to_se3",
    "se3_to_pose",
]
