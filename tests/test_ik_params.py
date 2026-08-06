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

"""Regression tests for the public IK parameter contract."""

from __future__ import annotations

import argparse
import pathlib

from openarm_control import (
    IKParams,
    Kinematics,
    ik_params_from_args,
    register_ik_args,
)
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from _support import make_setup


def test_cli_resolves_tested_defaults() -> None:
    parser = argparse.ArgumentParser()
    register_ik_args(parser)
    params = ik_params_from_args(parser.parse_args(["--limit-velocity"]))

    assert params.position_cost == 12.0
    assert params.orientation_cost == 1.5
    assert params.lm_damping == 0.01
    assert params.damping == 0.1
    assert params.posture_cost == 0.0
    assert params.max_iters == 5
    assert params.dt == 0.004
    assert params.frame_position_error_limit == 0.02
    assert params.frame_orientation_error_limit == 0.25
    assert params.position_error_latch_threshold == 0.006
    assert params.nullspace_cost == 8.5
    assert params.nullspace_return_rate == 1.6
    assert params.joint_braking
    assert params.joint_braking_distance == 0.2
    assert not hasattr(params, "joint_braking_reaction_time")
    assert params.singularity_max_approach_rate == 0.25
    assert params.kinetic_energy_cost == 2e-5
    assert not hasattr(params, "diag_reg")
    assert params.velocity_limits is not None
    for side in ("left", "right"):
        for index, expected in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S):
            assert (
                params.velocity_limits[f"openarm_{side}_joint{index + 1}"] == expected
            )


def test_unexposed_parameters_remain_regular_ik_fields() -> None:
    parser = argparse.ArgumentParser()
    register_ik_args(parser)
    options = {option for action in parser._actions for option in action.option_strings}
    params = IKParams(
        target_linear_speed_slow=0.4,
        joint_braking_exponent=3.0,
        singularity_ratio_stop=0.01,
    )

    assert "--ik-profile" not in options
    assert "--target-linear-speed-slow" not in options
    assert "--joint-braking-exponent" not in options
    assert "--singularity-ratio-stop" not in options
    assert params.target_linear_speed_slow == 0.4
    assert params.joint_braking_exponent == 3.0
    assert params.singularity_ratio_stop == 0.01


def test_control_overrides_and_velocity_yaml(tmp_path: pathlib.Path) -> None:
    custom_caps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    path = tmp_path / "velocity.yaml"
    path.write_text(
        "arm_velocity_limits:\n" + "".join(f"  - {value}\n" for value in custom_caps),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    register_ik_args(parser)
    params = ik_params_from_args(
        parser.parse_args(
            [
                "--limit-velocity",
                "--config",
                str(path),
                "--frame-position-error-limit",
                "0.004",
                "--frame-orientation-error-limit",
                "0.05",
                "--nullspace-cost",
                "8",
                "--nullspace-return-rate",
                "1.2",
                "--joint-braking-distance",
                "0.4",
                "--singularity-max-approach-rate",
                "0.2",
                "--kinetic-energy-cost",
                "0.00004",
            ]
        )
    )

    assert params.frame_position_error_limit == 0.004
    assert params.frame_orientation_error_limit == 0.05
    assert params.nullspace_cost == 8.0
    assert params.nullspace_return_rate == 1.2
    assert params.joint_braking_distance == 0.4
    assert params.singularity_max_approach_rate == 0.2
    assert params.kinetic_energy_cost == 4e-5
    assert params.velocity_limits is not None
    for side in ("left", "right"):
        for index, expected in enumerate(custom_caps):
            assert (
                params.velocity_limits[f"openarm_{side}_joint{index + 1}"] == expected
            )


def test_diag_reg_is_not_a_registered_solver_option() -> None:
    parser = argparse.ArgumentParser()
    register_ik_args(parser)
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--diag-reg" not in options


def test_braking_can_be_disabled_without_disabling_velocity_limits() -> None:
    parser = argparse.ArgumentParser()
    register_ik_args(parser)
    params = ik_params_from_args(
        parser.parse_args(["--limit-velocity", "--no-joint-braking"])
    )

    assert not params.joint_braking
    assert params.velocity_limits is not None
    solver = Kinematics(make_setup("right"), params)._ik
    assert solver is not None
    assert solver._joint_limit is not None
    assert solver._joint_limit.braking_distance is None
