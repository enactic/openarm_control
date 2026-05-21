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

"""Helpers for building collision-avoidance geom pairs from an OpenArm MJCF.

Categorizes collision geoms by parent body name into four groups:

  left_arm   – bodies whose name starts with ``openarm_left_``
  right_arm  – bodies whose name starts with ``openarm_right_``
  body       – the ``openarm_lifter_link`` body (currently has no collision geoms
               in cell.xml, included for forward compatibility)
  env        – everything else with non-zero contype/conaffinity

Visual-only geoms (``contype == 0`` and ``conaffinity == 0``) are skipped.

Returns the 5-group product (L,R), (L,B), (R,B), (L,E), (R,E) with empty
groups filtered out, suitable for ``mink.CollisionAvoidanceLimit(geom_pairs=...)``.
"""

from __future__ import annotations

import mujoco

_LEFT_PREFIX = "openarm_left_"
_RIGHT_PREFIX = "openarm_right_"
_BODY_NAME = "openarm_lifter_link"


def categorize_geoms(model: mujoco.MjModel) -> dict[str, list[str]]:
    """Return ``{"left_arm": [...], "right_arm": [...], "body": [...], "env": [...]}``.

    Visual-only geoms and unnamed geoms are skipped.
    """
    cats: dict[str, list[str]] = {"left_arm": [], "right_arm": [], "body": [], "env": []}
    for i in range(model.ngeom):
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue
        gname = model.geom(i).name
        if not gname:
            continue
        bname = model.body(model.geom_bodyid[i]).name or ""
        if bname.startswith(_LEFT_PREFIX):
            cats["left_arm"].append(gname)
        elif bname.startswith(_RIGHT_PREFIX):
            cats["right_arm"].append(gname)
        elif bname == _BODY_NAME:
            cats["body"].append(gname)
        else:
            cats["env"].append(gname)
    return cats


def geom_pairs_for_arms(model: mujoco.MjModel) -> list[tuple[list[str], list[str]]]:
    """Build the geom-pair list passed to ``mink.CollisionAvoidanceLimit``.

    Pair groups, in order: arm-arm, arm-body (x2), arm-env (x2). Empty groups
    are dropped so mink does not see ``([], [])`` pairs.
    """
    c = categorize_geoms(model)
    L, R, B, E = c["left_arm"], c["right_arm"], c["body"], c["env"]
    candidates = [(L, R), (L, B), (R, B), (L, E), (R, E)]
    return [(a, b) for (a, b) in candidates if a and b]
