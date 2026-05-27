# OpenArm Control

Reusable kinematics and control utilities for OpenArm, backed by MuJoCo and [mink](https://github.com/kevinzakka/mink).

## Install

```bash
uv sync
```

## Usage

### `Kinematics`

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
result = kin.solve(dt=0.1, n_iters=5)        # float32[16] right[8]+left[8]
```

### `IKParams`

Solver configuration passed to `Kinematics`. All fields have defaults.

| Field | Default | Description |
|---|---|---|
| `position_cost` | `1.0` | Position task weight |
| `orientation_cost` | `1.0` | Orientation task weight |
| `lm_damping` | `0.01` | Per-task Levenberg-Marquardt damping |
| `damping` | `0.25` | Global Tikhonov regularization |
| `solver` | `"daqp"` | QP backend |
| `posture_cost` | `0.01` | Neutral posture task weight (0 = disabled) |
| `diag_reg` | `0.0` | QP diagonal regularization |
| `dt` | `0.1` | Integration timestep per iteration |
| `max_iters` | `5` | IK iterations per solve |
| `velocity_limits` | `None` | Per-joint velocity limits (applied in rad/s from `config.py`); `None` = disabled |

Build from CLI args with `register_ik_args` + `ik_params_from_args`:

## Related links

- 💬 Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
- 📬 Contact us through <openarm@enactic.ai>

## License

Licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
