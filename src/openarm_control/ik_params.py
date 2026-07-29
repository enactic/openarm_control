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

"""Parameters, validation, and CLI construction for OpenArm inverse kinematics."""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import yaml

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S


@dataclass
class IKParams:
    """Configuration for the mink QP-based IK solver."""

    position_cost: float = 12.0
    orientation_cost: float = 1.5
    lm_damping: float = 0.01
    damping: float = 0.1
    solver: str = "daqp"
    posture_cost: float = 0.0
    dt: float = 0.004
    max_iters: int = 5
    velocity_limits: dict[str, float] | None = None

    frame_position_error_limit: float = 0.015
    frame_orientation_error_limit: float = 0.25
    target_linear_speed_slow: float = 0.6
    target_linear_speed_fast: float = 0.9
    position_error_latch_threshold: float = 0.006

    joint_limit_gain: float = 0.95
    joint_braking: bool = True
    joint_braking_distance: float = 0.2
    joint_braking_exponent: float = 2.0
    joint_braking_distance_buffer: float = 0.01

    jacobian_characteristic_length: float = 0.3
    nullspace_cost: float = 7.0
    nullspace_return_rate: float = 1.6
    nullspace_max_speed: float = 1.0
    nullspace_ratio_low: float = 0.02
    nullspace_ratio_high: float = 0.05

    singularity_ratio_stop: float = 0.02
    singularity_ratio_slow: float = 0.08
    singularity_max_approach_rate: float = 0.25
    singularity_braking_exponent: float = 2.0
    singularity_gradient_epsilon: float = 1e-4

    kinetic_energy_cost: float = 2e-5


def validate_ik_params(params: IKParams) -> None:
    """Validate tracking, task, and limit parameters."""
    if not np.isfinite(params.dt) or params.dt <= 0.0:
        raise ValueError("IK control timestep must be finite and positive.")
    if params.max_iters <= 0:
        raise ValueError("IK max_iters must be positive.")
    non_negative = (
        ("position_cost", params.position_cost),
        ("orientation_cost", params.orientation_cost),
        ("lm_damping", params.lm_damping),
        ("damping", params.damping),
        ("posture_cost", params.posture_cost),
        ("frame_position_error_limit", params.frame_position_error_limit),
        ("frame_orientation_error_limit", params.frame_orientation_error_limit),
        ("target_linear_speed_slow", params.target_linear_speed_slow),
        ("joint_braking_distance_buffer", params.joint_braking_distance_buffer),
        ("nullspace_cost", params.nullspace_cost),
        ("nullspace_return_rate", params.nullspace_return_rate),
        ("nullspace_max_speed", params.nullspace_max_speed),
        (
            "singularity_max_approach_rate",
            params.singularity_max_approach_rate,
        ),
        ("kinetic_energy_cost", params.kinetic_energy_cost),
    )
    for name, value in non_negative:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    positive = (
        ("target_linear_speed_fast", params.target_linear_speed_fast),
        (
            "position_error_latch_threshold",
            params.position_error_latch_threshold,
        ),
        ("joint_braking_distance", params.joint_braking_distance),
        ("joint_braking_exponent", params.joint_braking_exponent),
        (
            "jacobian_characteristic_length",
            params.jacobian_characteristic_length,
        ),
        ("singularity_braking_exponent", params.singularity_braking_exponent),
        ("singularity_gradient_epsilon", params.singularity_gradient_epsilon),
    )
    for name, value in positive:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    if not 0.0 < params.joint_limit_gain <= 1.0:
        raise ValueError("joint_limit_gain must be in (0, 1].")
    if params.target_linear_speed_fast <= params.target_linear_speed_slow:
        raise ValueError(
            "target_linear_speed_fast must be greater than target_linear_speed_slow."
        )
    if not 0.0 <= params.nullspace_ratio_low < params.nullspace_ratio_high:
        raise ValueError("Expected 0 <= nullspace_ratio_low < nullspace_ratio_high.")
    if not 0.0 <= params.singularity_ratio_stop < params.singularity_ratio_slow:
        raise ValueError(
            "Expected 0 <= singularity_ratio_stop < singularity_ratio_slow."
        )


def _load_velocity_caps(config_path: pathlib.Path | None) -> list[float]:
    """Return per-joint velocity caps in rad/s."""
    if config_path is None:
        return ARM_JOINT_VELOCITY_LIMITS_RAD_S

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "arm_velocity_limits" not in data:
        raise ValueError(
            f"Config file {config_path} has no top-level 'arm_velocity_limits' list."
        )

    caps = [float(v) for v in data["arm_velocity_limits"]]
    expected = len(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    if len(caps) != expected:
        raise ValueError(
            f"arm_velocity_limits in {config_path} has {len(caps)} entries; "
            f"expected {expected}."
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in caps):
        raise ValueError("arm_velocity_limits entries must be finite and positive.")
    return caps


def register_ik_args(parser: argparse.ArgumentParser) -> None:
    """Register IK-specific CLI flags. Call after register_common_args."""
    defaults = IKParams()
    default_tick_hz = 1.0 / defaults.dt
    parser.add_argument(
        "--pos-cost",
        type=float,
        default=defaults.position_cost,
        help=f"Position task cost (default: {defaults.position_cost})",
    )
    parser.add_argument(
        "--ori-cost",
        type=float,
        default=defaults.orientation_cost,
        help=f"Orientation task cost (default: {defaults.orientation_cost})",
    )
    parser.add_argument(
        "--lm-damping",
        type=float,
        default=defaults.lm_damping,
        help=f"Per-task LM damping (default: {defaults.lm_damping})",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=defaults.damping,
        help=f"Global Tikhonov regularization (default: {defaults.damping})",
    )
    parser.add_argument(
        "--solver",
        default=defaults.solver,
        help=f"QP backend (default: {defaults.solver})",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=defaults.max_iters,
        help=f"IK iterations per event (default: {defaults.max_iters})",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Outer control period; defaults to 1 / --tick-hz.",
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=defaults.posture_cost,
        help=(
            f"Full posture task weight, 0=disabled (default: {defaults.posture_cost})"
        ),
    )
    parser.add_argument(
        "--limit-velocity",
        action="store_true",
        help="Enable recoverable per-joint position and velocity limits.",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=(
            "YAML file with a top-level 'arm_velocity_limits: [rad/s, ...]' list that "
            "overrides the built-in per-joint caps. Used only with --limit-velocity."
        ),
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=default_tick_hz,
        help=(
            "Nominal target/control rate used when --dt is omitted "
            f"(default: {default_tick_hz})."
        ),
    )

    parser.add_argument(
        "--frame-position-error-limit",
        type=float,
        default=defaults.frame_position_error_limit,
        help="Total position-error request per outer IK solve in meters.",
    )
    parser.add_argument(
        "--frame-orientation-error-limit",
        type=float,
        default=defaults.frame_orientation_error_limit,
        help="Total orientation-error request per outer IK solve in radians.",
    )
    parser.add_argument(
        "--nullspace-cost",
        type=float,
        default=defaults.nullspace_cost,
    )
    parser.add_argument(
        "--nullspace-return-rate",
        type=float,
        default=defaults.nullspace_return_rate,
    )
    parser.add_argument(
        "--joint-braking",
        action=argparse.BooleanOptionalAction,
        default=defaults.joint_braking,
        help="Enable preventive joint braking with velocity limits (default: enabled).",
    )
    parser.add_argument(
        "--joint-braking-distance",
        type=float,
        default=defaults.joint_braking_distance,
    )
    parser.add_argument(
        "--singularity-max-approach-rate",
        type=float,
        default=defaults.singularity_max_approach_rate,
    )
    parser.add_argument(
        "--kinetic-energy-cost",
        type=float,
        default=defaults.kinetic_energy_cost,
    )


def ik_params_from_args(args: argparse.Namespace) -> IKParams:
    """Build IKParams from parsed args."""
    dt = args.dt
    if dt is None:
        if not np.isfinite(args.tick_hz) or args.tick_hz <= 0.0:
            raise ValueError("--tick-hz must be finite and positive.")
        dt = 1.0 / args.tick_hz

    velocity_limits: dict[str, float] | None = None
    if args.limit_velocity:
        caps = _load_velocity_caps(getattr(args, "config", None))
        velocity_limits = {
            f"openarm_{side}_joint{i + 1}": v
            for side in ("left", "right")
            for i, v in enumerate(caps)
        }

    return IKParams(
        position_cost=args.pos_cost,
        orientation_cost=args.ori_cost,
        lm_damping=args.lm_damping,
        damping=args.damping,
        solver=args.solver,
        posture_cost=args.posture_cost,
        dt=dt,
        max_iters=args.max_iters,
        velocity_limits=velocity_limits,
        frame_position_error_limit=args.frame_position_error_limit,
        frame_orientation_error_limit=args.frame_orientation_error_limit,
        nullspace_cost=args.nullspace_cost,
        nullspace_return_rate=args.nullspace_return_rate,
        joint_braking=args.joint_braking,
        joint_braking_distance=args.joint_braking_distance,
        singularity_max_approach_rate=args.singularity_max_approach_rate,
        kinetic_energy_cost=args.kinetic_energy_cost,
    )
