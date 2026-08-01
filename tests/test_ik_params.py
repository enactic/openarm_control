"""Regression tests for the public IK parameter contract."""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import unittest

from openarm_control import (
    IKParams,
    Kinematics,
    ik_params_from_args,
    register_ik_args,
)
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from tests._support import make_setup


class ParameterTest(unittest.TestCase):
    """Verify the flat public configuration contract."""

    def test_cli_resolves_tested_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        params = ik_params_from_args(parser.parse_args(["--limit-velocity"]))

        self.assertEqual(params.position_cost, 12.0)
        self.assertEqual(params.orientation_cost, 1.5)
        self.assertEqual(params.lm_damping, 0.01)
        self.assertEqual(params.damping, 0.1)
        self.assertEqual(params.posture_cost, 0.0)
        self.assertEqual(params.max_iters, 5)
        self.assertEqual(params.dt, 0.004)
        self.assertEqual(params.frame_position_error_limit, 0.02)
        self.assertEqual(params.frame_orientation_error_limit, 0.25)
        self.assertEqual(params.position_error_latch_threshold, 0.006)
        self.assertEqual(params.nullspace_cost, 8.5)
        self.assertEqual(params.nullspace_return_rate, 1.6)
        self.assertTrue(params.joint_braking)
        self.assertEqual(params.joint_braking_distance, 0.2)
        self.assertFalse(hasattr(params, "joint_braking_reaction_time"))
        self.assertEqual(params.singularity_max_approach_rate, 0.25)
        self.assertEqual(params.kinetic_energy_cost, 2e-5)
        self.assertFalse(hasattr(params, "diag_reg"))
        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_unexposed_parameters_remain_regular_ik_fields(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        options = {
            option for action in parser._actions for option in action.option_strings
        }
        params = IKParams(
            target_linear_speed_slow=0.4,
            joint_braking_exponent=3.0,
            singularity_ratio_stop=0.01,
        )

        self.assertNotIn("--ik-profile", options)
        self.assertNotIn("--target-linear-speed-slow", options)
        self.assertNotIn("--joint-braking-exponent", options)
        self.assertNotIn("--singularity-ratio-stop", options)
        self.assertEqual(params.target_linear_speed_slow, 0.4)
        self.assertEqual(params.joint_braking_exponent, 3.0)
        self.assertEqual(params.singularity_ratio_stop, 0.01)

    def test_control_overrides_and_velocity_yaml(self) -> None:
        custom_caps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "velocity.yaml"
            path.write_text(
                "arm_velocity_limits:\n"
                + "".join(f"  - {value}\n" for value in custom_caps),
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

        self.assertEqual(params.frame_position_error_limit, 0.004)
        self.assertEqual(params.frame_orientation_error_limit, 0.05)
        self.assertEqual(params.nullspace_cost, 8.0)
        self.assertEqual(params.nullspace_return_rate, 1.2)
        self.assertEqual(params.joint_braking_distance, 0.4)
        self.assertEqual(params.singularity_max_approach_rate, 0.2)
        self.assertEqual(params.kinetic_energy_cost, 4e-5)
        assert params.velocity_limits is not None
        for side in ("left", "right"):
            for index, expected in enumerate(custom_caps):
                self.assertEqual(
                    params.velocity_limits[f"openarm_{side}_joint{index + 1}"],
                    expected,
                )

    def test_diag_reg_is_not_a_registered_solver_option(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        options = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertNotIn("--diag-reg", options)

    def test_braking_can_be_disabled_without_disabling_velocity_limits(self) -> None:
        parser = argparse.ArgumentParser()
        register_ik_args(parser)
        params = ik_params_from_args(
            parser.parse_args(["--limit-velocity", "--no-joint-braking"])
        )

        self.assertFalse(params.joint_braking)
        self.assertIsNotNone(params.velocity_limits)
        solver = Kinematics(make_setup("right"), params)._ik
        assert solver is not None
        assert solver._joint_limit is not None
        self.assertIsNone(solver._joint_limit.braking_distance)


if __name__ == "__main__":
    unittest.main()
