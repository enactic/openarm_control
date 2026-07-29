# OpenArm Control

Reusable kinematics and control utilities for OpenArm, backed by MuJoCo and [mink](https://github.com/kevinzakka/mink).

## Install

```bash
uv sync
```

## Usage

### `Kinematics`

Poses are `float32[7] = [px, py, pz, qw, qx, qy, qz]`, expressed in the
scene's `arm_origin` site frame by default (`--origin-frame world` restores
world-frame poses).

```python
from openarm_control import Kinematics, IKParams, ArmSetup

# FK only
kin = Kinematics(setup)
pose = kin.fk("right", joints)               # float32[7]
pose_r, pose_l = kin.fk_bimanual(r, l)       # single mj_forward

# IK
kin = Kinematics(setup, IKParams(damping=0.25, posture_cost=0.01))
kin.set_target("right", pose_r)
kin.set_target("left", pose_l)
result = kin.solve()                         # float32[16] right[8]+left[8]
```

### `IKParams`

Solver configuration passed to `Kinematics`. All fields have defaults.

| Field | Default | Description |
|---|---|---|
| `position_cost` | `12.0` | Position task weight |
| `orientation_cost` | `1.5` | Orientation task weight |
| `lm_damping` | `0.01` | Per-task Levenberg-Marquardt damping |
| `damping` | `0.1` | Global Tikhonov regularization |
| `solver` | `"daqp"` | QP backend |
| `posture_cost` | `0.0` | Full home posture task weight (0 = disabled) |
| `dt` | `0.004` | Outer control period, divided across IK iterations |
| `max_iters` | `5` | IK sub-iterations per solve |
| `velocity_limits` | `None` | Per-joint velocity limits (applied in rad/s from `config.py`); `None` = disabled |
| `frame_position_error_limit` | `0.015` | Total position-error request per outer IK solve |
| `frame_orientation_error_limit` | `0.25` | Total orientation-error request per outer IK solve |
| `nullspace_cost` | `7.0` | Fixed-home nullspace posture cost |
| `nullspace_return_rate` | `1.6` | Nullspace return rate in s⁻¹ |
| `joint_braking_distance` | `0.2` | Joint-limit braking distance in radians |
| `singularity_max_approach_rate` | `0.25` | Maximum singularity-ratio approach rate |
| `kinetic_energy_cost` | `2e-5` | Kinetic-energy regularization cost |

Build from CLI args with `register_ik_args` + `ik_params_from_args`:

```bash
python your_ik_node.py --tick-hz 250 --limit-velocity
```

The remaining curve-shape and threshold parameters are regular `IKParams`
fields but are intentionally not exposed as CLI flags. Call
`update_measured_state()` with fresh driver qpos for state-aware limits; the
caller owns freshness and should call `clear_measured_state()` when that state
expires.

## Related links

- 💬 Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
- 📬 Contact us through <openarm@enactic.ai>

## License

Licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
