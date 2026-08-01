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
pose = kin.fk("right", joints)  # float32[7]
pose_r, pose_l = kin.fk_bimanual(r, l)  # single mj_forward

# IK
kin = Kinematics(setup, IKParams())
kin.set_target("right", pose_r)
kin.set_target("left", pose_l)
result = kin.solve()  # float32[16] right[8]+left[8]
```

### `IKParams`

Solver configuration passed to `Kinematics`. Key fields are listed below; all
fields have defaults.

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
| `velocity_limits` | `None` | Joint-name to velocity-cap mapping in rad/s; `None` keeps position-only limits |
| `frame_position_error_limit` | `0.02` | Total position-error request per outer IK solve |
| `frame_orientation_error_limit` | `0.25` | Total orientation-error request per outer IK solve |
| `nullspace_cost` | `8.5` | Fixed-home nullspace posture cost |
| `nullspace_return_rate` | `1.6` | Nullspace return rate in s⁻¹ |
| `joint_braking` | `True` | Enable preventive braking when velocity limits are active |
| `joint_braking_distance` | `0.2` | Joint-limit braking distance in radians |
| `singularity_max_approach_rate` | `0.25` | Maximum singularity-ratio approach rate |
| `kinetic_energy_cost` | `2e-5` | Kinetic-energy regularization cost |

Build from CLI args with `register_ik_args` + `ik_params_from_args`:

```bash
python your_ik_node.py --tick-hz 250 --limit-velocity
```

Without `--config`, `--limit-velocity` uses the built-in per-arm caps
`[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s`. An optional YAML override uses a
seven-entry `arm_velocity_limits` list:

```yaml
arm_velocity_limits: [2.0, 2.0, 3.14, 3.14, 6.3, 6.3, 6.3]
```

```bash
python your_ik_node.py --limit-velocity --config ./ik_limits.yaml
```

### Constrained IK behavior

A bare `IKParams()` instance enables four complementary mechanisms:

- **Frame-error bounding** limits the Cartesian correction requested by one
  outer solve while retaining the full target for subsequent cycles. Position
  bounding activates smoothly with target linear speed and remains latched
  while accumulated position lag is significant; orientation bounding is
  always active when configured.
- **Nullspace posture regulation** guides the redundant arm motion toward the
  initial home posture without applying a full joint-space posture objective.
- **Singularity approach limiting** slows only motion that moves the arm toward
  a poorly conditioned configuration.
- **Kinetic-energy regularization** provides a weak mass-matrix-based
  tie-breaker between kinematically similar solutions. It is not inverse
  dynamics or gravity compensation.

Frame-error bounding limits accumulated feedback correction, not target
velocity, and does not replace joint velocity limits. The full-joint home
`PostureTask` is a separate optional objective controlled by `posture_cost` and
is disabled by default.

The bare library default leaves hardware-specific per-joint velocity limits
opt-in: provide `velocity_limits` directly or use `--limit-velocity`. The CLI
flag selects the built-in caps unless `--config` overrides them. Enabling these
limits replaces the position-only arm limit with one recoverable
position/velocity envelope, so a command already slightly outside a position
bound can return over multiple feasible steps. Preventive braking then reduces
only the velocity approaching a nearby position bound; it is enabled by default
with velocity limits and can be disabled with `--no-joint-braking`.

These limits constrain the QP solution. A downstream driver may apply its own
independently configured execution-layer velocity envelope; the two sets of
caps are not required to match.

Set both frame-error limits to zero to remove frame-error bounding. Set
`nullspace_cost`, `singularity_max_approach_rate`, or `kinetic_energy_cost` to
zero to disable the corresponding mechanism. These are kinematic command-layer
mechanisms, not a certified whole-robot safety controller.

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
