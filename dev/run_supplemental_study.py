#!/usr/bin/env python3
"""Run supplemental IK studies omitted from the original broad sweep."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import ik_safety_study as study
import numpy as np
import pyarrow.parquet as pq

DEFAULT_RECORD_ROOT = Path(
    "/hdd_data/rollout/"
    "pillow_0702_history_with_visual_from_flatten_hist_base_30k_0711_window_20_80k"
    "/dataset/episodes"
)


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _current_compact_profile(name: str = "compact_candidate") -> study.Profile:
    return study.current_profile(
        name,
        velocity_caps=(2.5, *study.CURRENT_CAPS[1:]),
        kinetic_energy_cost=5e-5,
        frame_error_limit_linear_slow=0.6,
        frame_error_limit_linear_fast=0.9,
        frame_error_limit_activation_rise_rate=1e6,
        frame_error_limit_activation_fall_rate=1e6,
        nullspace_singularity_high=0.08,
    )


def base_qp_profiles() -> list[study.Profile]:
    """Sweep core QP weights and numerical parameters one factor at a time."""
    profiles = [study.current_profile("current")]
    for value in (2.5, 5.0, 10.0, 20.0, 40.0):
        profiles.append(
            study.current_profile(
                f"position_cost_{_slug(value)}",
                position_cost=value,
            )
        )
    for value in (0.2, 0.5, 1.0, 2.0, 5.0):
        profiles.append(
            study.current_profile(
                f"orientation_cost_{_slug(value)}",
                orientation_cost=value,
            )
        )
    for value in (0.01, 0.03, 0.1, 0.3, 1.0):
        profiles.append(
            study.current_profile(
                f"global_damping_{_slug(value)}",
                damping=value,
            )
        )
    for value in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
        profiles.append(
            study.current_profile(
                f"lm_damping_{_slug(value)}",
                lm_damping=value,
            )
        )
    for value in (0.2, 0.3, 0.4, 0.5):
        profiles.append(
            study.current_profile(
                f"characteristic_length_{_slug(value)}",
                nullspace_characteristic_length=value,
            )
        )
    for value in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        profiles.append(
            study.current_profile(
                f"singularity_gradient_epsilon_{_slug(value)}",
                singularity_gradient_epsilon=value,
            )
        )
    profiles.extend(
        [
            study.current_profile(
                "task_cost_scale_0p5",
                position_cost=5.0,
                orientation_cost=0.5,
            ),
            study.current_profile(
                "task_cost_scale_2",
                position_cost=20.0,
                orientation_cost=2.0,
            ),
            study.current_profile(
                "task_cost_scale_4",
                position_cost=40.0,
                orientation_cost=4.0,
            ),
        ]
    )
    return list({profile.name: profile for profile in profiles}.values())


def interaction_profiles() -> list[study.Profile]:
    """Return a full factorial over the five objective/safety shapers."""
    profiles: list[study.Profile] = []
    labels = ("null", "sing", "brake", "error", "energy")
    for enabled in itertools.product((False, True), repeat=len(labels)):
        flags = dict(zip(labels, enabled, strict=True))
        name = "factor_" + "_".join(
            label if flags[label] else f"no{label}" for label in labels
        )
        profiles.append(
            study.current_profile(
                name,
                nullspace_cost=12.0 if flags["null"] else 0.0,
                singularity_approach_limit=flags["sing"],
                joint_limit_braking=flags["brake"],
                frame_position_error_limit=0.003 if flags["error"] else 0.0,
                kinetic_energy_cost=3e-5 if flags["energy"] else 0.0,
            )
        )
    return profiles


def core_scenarios() -> list[study.Scenario]:
    """Use adversarial and normal paths for parameter and interaction scans."""
    factory = study.PoseFactory()
    return [
        study.make_reach_scenario(factory, speed=0.4),
        study.make_reach_scenario(factory, speed=0.8, lateral=-0.10),
        study.make_reach_scenario(factory, speed=0.8, lateral=0.10),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.4,
            lateral=-0.10,
        ),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=0.10,
        ),
        study.make_extended_circle_scenario(factory, speed=0.4),
        study.make_extended_circle_scenario(factory, speed=0.8),
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=10.0,
            extended=True,
        ),
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=10.0,
            extended=False,
        ),
        study.make_normal_lissajous_scenario(
            factory,
            speed=0.6,
            mode="bimanual",
        ),
    ]


def interaction_scenarios() -> list[study.Scenario]:
    """Keep the factorial compact while retaining each failure mechanism."""
    scenarios = core_scenarios()
    selected = (
        "reach_right_p0p00_v0p40",
        "reach_right_m0p10_v0p80",
        "retract_diag_m0p10_v0p40",
        "retract_diag_p0p10_v0p80",
        "extended_circle_right_v0p80",
        "extended_wrist_right_w10p0",
        "normal_wrist_right_w10p0",
        "normal_bimanual_v0p60",
    )
    by_name = {scenario.name: scenario for scenario in scenarios}
    return [by_name[name] for name in selected]


def _poses_from_joint_path(
    factory: study.PoseFactory,
    right_q: np.ndarray,
    left_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right_poses: list[np.ndarray] = []
    left_poses: list[np.ndarray] = []
    for right, left in zip(right_q, left_q, strict=True):
        right_pose, left_pose = factory.bimanual(right, left)
        right_poses.append(right_pose)
        left_poses.append(left_pose)
    return (
        study._continuous_quaternions(np.asarray(right_poses)),
        study._continuous_quaternions(np.asarray(left_poses)),
    )


def _joint_range(
    factory: study.PoseFactory,
    side: str,
    joint_index: int,
) -> tuple[float, float]:
    joint = factory.setup.model.joint(
        f"openarm_{side}_joint{joint_index + 1}"
    )
    return float(joint.range[0]), float(joint.range[1])


def make_joint_limit_scenario(
    factory: study.PoseFactory,
    *,
    joint_index: int,
    boundary: str,
    joint_speed: float = 1.5,
) -> study.Scenario:
    """Generate an exactly feasible FK path toward one physical joint bound."""
    initial_right = np.array([0.0, 0.45, 0.0, 1.35, 0.0, 0.0, 0.0])
    initial_left = study.START_Q_LEFT.copy()
    lower, upper = _joint_range(factory, "right", joint_index)
    start = initial_right[joint_index]
    target = lower + 0.01 if boundary == "lower" else upper - 0.01
    duration = max(0.3, 1.5 * abs(target - start) / joint_speed)
    move_count = max(2, math.ceil(duration / study.CONTROL_DT))
    hold_count = round(0.35 / study.CONTROL_DT)
    settle_count = round(0.20 / study.CONTROL_DT)

    amounts = np.linspace(0.0, 1.0, move_count + 1)[1:]
    amounts = amounts * amounts * (3.0 - 2.0 * amounts)
    forward = np.repeat(initial_right[None, :], move_count, axis=0)
    forward[:, joint_index] = start + amounts * (target - start)
    reverse = forward[::-1].copy()
    right_q = np.vstack(
        [
            np.repeat(initial_right[None, :], settle_count, axis=0),
            forward,
            np.repeat(forward[-1][None, :], hold_count, axis=0),
            reverse,
            np.repeat(initial_right[None, :], hold_count, axis=0),
        ]
    )
    left_q = np.repeat(initial_left[None, :], right_q.shape[0], axis=0)
    right_pose, _ = _poses_from_joint_path(factory, right_q, left_q)
    phase = np.concatenate(
        [
            np.zeros(settle_count, dtype=np.int16),
            np.ones(move_count, dtype=np.int16),
            np.full(hold_count, 2, dtype=np.int16),
            np.full(move_count, 3, dtype=np.int16),
            np.full(hold_count, 4, dtype=np.int16),
        ]
    )
    return study.Scenario(
        name=f"joint{joint_index + 1}_{boundary}_v{_slug(joint_speed)}",
        family="joint_limit",
        mode="right",
        speed=joint_speed,
        times=np.arange(right_q.shape[0]) * study.CONTROL_DT,
        phase=phase,
        target_right=right_pose,
        target_left=np.full_like(right_pose, np.nan),
        initial_right=initial_right,
        initial_left=initial_left,
        description=(
            f"FK-generated path moving right J{joint_index + 1} to "
            f"0.01 rad inside its {boundary} physical limit."
        ),
    )


def joint_limit_scenarios() -> list[study.Scenario]:
    factory = study.PoseFactory()
    return [
        make_joint_limit_scenario(
            factory,
            joint_index=joint_index,
            boundary=boundary,
        )
        for joint_index in range(7)
        for boundary in ("lower", "upper")
    ]


def joint_limit_profiles() -> list[study.Profile]:
    isolated = {
        "nullspace_cost": 0.0,
        "singularity_approach_limit": False,
        "frame_position_error_limit": 0.0,
        "kinetic_energy_cost": 0.0,
    }
    profiles = [
        study.current_profile("current"),
        study.current_profile("current_no_braking", joint_limit_braking=False),
        study.current_profile(
            "isolated_no_braking",
            joint_limit_braking=False,
            **isolated,
        ),
    ]
    for distance in (0.10, 0.20, 0.35, 0.50, 0.70):
        profiles.append(
            study.current_profile(
                f"isolated_braking_{_slug(distance)}",
                joint_limit_braking=True,
                joint_limit_braking_slowdown_distance=distance,
                **isolated,
            )
        )
    return profiles


def braking_dynamics_profiles() -> list[study.Profile]:
    """Scan measured-state prediction margins around the active braking setup."""
    profiles = [
        study.current_profile("current"),
        study.current_profile("no_braking", joint_limit_braking=False),
        study.current_profile(
            "gravity_compensation",
            gravity_compensation=True,
        ),
        study.current_profile(
            "guard_0p05_gravity_compensation",
            joint_limit_braking_guard_margin=0.05,
            gravity_compensation=True,
        ),
    ]
    for reaction_time in (0.08, 0.12, 0.16, 0.20):
        profiles.append(
            study.current_profile(
                f"reaction_{_slug(reaction_time)}",
                joint_limit_braking_reaction_time=reaction_time,
            )
        )
    for buffer in (0.03, 0.05, 0.08, 0.12):
        profiles.append(
            study.current_profile(
                f"buffer_{_slug(buffer)}",
                joint_limit_braking_distance_buffer=buffer,
            )
        )
    for guard_margin in (0.03, 0.05, 0.08, 0.12):
        profiles.append(
            study.current_profile(
                f"guard_{_slug(guard_margin)}",
                joint_limit_braking_guard_margin=guard_margin,
            )
        )
    for reaction_time, buffer in ((0.08, 0.03), (0.12, 0.05), (0.16, 0.08)):
        profiles.append(
            study.current_profile(
                f"reaction_{_slug(reaction_time)}_buffer_{_slug(buffer)}",
                joint_limit_braking_reaction_time=reaction_time,
                joint_limit_braking_distance_buffer=buffer,
            )
        )
    for guard_margin, reaction_time in ((0.05, 0.08), (0.08, 0.12)):
        profiles.append(
            study.current_profile(
                f"guard_{_slug(guard_margin)}_reaction_{_slug(reaction_time)}",
                joint_limit_braking_guard_margin=guard_margin,
                joint_limit_braking_reaction_time=reaction_time,
            )
        )
    return profiles


def braking_dynamics_scenarios() -> list[study.Scenario]:
    """Stress gravity-sensitive shoulder/elbow bounds at several speeds."""
    factory = study.PoseFactory()
    scenarios: list[study.Scenario] = []
    speeds_by_joint = {
        0: (0.5, 1.0, 1.5, 2.0),
        1: (1.0, 1.5, 2.0),
        3: (1.0, 1.5, 2.0),
    }
    for joint_index, speeds in speeds_by_joint.items():
        for boundary in ("lower", "upper"):
            for speed in speeds:
                scenarios.append(
                    make_joint_limit_scenario(
                        factory,
                        joint_index=joint_index,
                        boundary=boundary,
                        joint_speed=speed,
                    )
                )
    return scenarios


def make_random_cartesian_scenario(
    factory: study.PoseFactory,
    *,
    seed: int,
    speed: float,
    mode: str,
    extended: bool,
) -> study.Scenario:
    """Generate a deterministic smooth VR-like Cartesian trajectory."""
    rng = np.random.default_rng(seed)
    initial_right = (
        study.EXTENDED_Q_RIGHT.copy()
        if extended
        else study.START_Q_RIGHT.copy()
    )
    initial_left = (
        study.EXTENDED_Q_LEFT.copy()
        if extended
        else study.START_Q_LEFT.copy()
    )
    right_pose, left_pose = factory.bimanual(initial_right, initial_left)
    active_sides = study.SIDES if mode == "bimanual" else (mode,)
    base = {"right": right_pose, "left": left_pose}
    position_scale = 0.018 if extended else 0.040
    orientation_scale = 0.12 if extended else 0.22
    multipliers = np.array([0.7, 1.1, 1.6])
    amplitudes: dict[str, np.ndarray] = {}
    phases: dict[str, np.ndarray] = {}
    rot_amplitudes: dict[str, np.ndarray] = {}
    rot_phases: dict[str, np.ndarray] = {}
    for side in active_sides:
        amplitudes[side] = position_scale * rng.uniform(0.65, 1.0, size=3)
        phases[side] = rng.uniform(-math.pi, math.pi, size=3)
        rot_amplitudes[side] = orientation_scale * rng.uniform(0.6, 1.0, size=3)
        rot_phases[side] = rng.uniform(-math.pi, math.pi, size=3)

    velocity_bound = max(
        float(np.linalg.norm(amplitudes[side] * multipliers))
        for side in active_sides
    )
    omega = speed / max(velocity_bound, 1e-6)
    duration = 4.0
    builder = study.PathBuilder({side: base[side] for side in active_sides})
    builder.hold(0.25)

    def target_at(elapsed: float) -> dict[str, np.ndarray]:
        targets: dict[str, np.ndarray] = {}
        for side in active_sides:
            phase = phases[side]
            position_delta = amplitudes[side] * (
                np.sin(multipliers * omega * elapsed + phase) - np.sin(phase)
            )
            rot_phase = rot_phases[side]
            rotvec = rot_amplitudes[side] * (
                np.sin(0.65 * multipliers * omega * elapsed + rot_phase)
                - np.sin(rot_phase)
            )
            pose = base[side].copy()
            pose[:3] += position_delta
            targets[side] = study.rotate_pose_local(pose, rotvec)
        return targets

    builder.sample(duration, target_at, phase=1)
    builder.hold(0.4, phase=2)
    return study._scenario_from_builder(
        name=(
            f"random_{'extended' if extended else 'normal'}_{mode}_"
            f"seed{seed}_v{_slug(speed)}"
        ),
        family="randomized",
        mode=mode,
        speed=speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description="Seeded smooth Cartesian motion with simultaneous orientation.",
    )


def randomized_scenarios() -> list[study.Scenario]:
    factory = study.PoseFactory()
    scenarios: list[study.Scenario] = []
    for seed in range(4):
        scenarios.append(
            make_random_cartesian_scenario(
                factory,
                seed=seed,
                speed=0.35,
                mode="right",
                extended=False,
            )
        )
        scenarios.append(
            make_random_cartesian_scenario(
                factory,
                seed=100 + seed,
                speed=0.70,
                mode="bimanual" if seed % 2 else "right",
                extended=False,
            )
        )
        scenarios.append(
            make_random_cartesian_scenario(
                factory,
                seed=200 + seed,
                speed=0.30,
                mode="right",
                extended=True,
            )
        )
    return scenarios


def randomized_profiles() -> list[study.Profile]:
    return [
        study.current_profile("current"),
        _current_compact_profile(),
        study.current_profile("no_nullspace", nullspace_cost=0.0),
        study.current_profile(
            "no_singularity_limit",
            singularity_approach_limit=False,
        ),
        study.current_profile("no_joint_braking", joint_limit_braking=False),
        study.current_profile("no_error_limit", frame_position_error_limit=0.0),
        study.current_profile("no_energy", kinetic_energy_cost=0.0),
    ]


def robustness_profiles() -> list[study.Profile]:
    profiles = [study.current_profile("current")]
    for delay in (0.02, 0.04, 0.08):
        profiles.append(
            study.current_profile(
                f"state_delay_{round(delay * 1e3)}ms",
                state_delay_s=delay,
            )
        )
    for delay in (0.008, 0.02, 0.04):
        profiles.append(
            study.current_profile(
                f"command_delay_{round(delay * 1e3)}ms",
                command_delay_s=delay,
            )
        )
    for delay in (0.02, 0.04):
        profiles.append(
            study.current_profile(
                f"state_command_delay_{round(delay * 1e3)}ms",
                state_delay_s=delay,
                command_delay_s=delay,
            )
        )
    for rate in (125.0, 50.0, 20.0):
        profiles.append(
            study.current_profile(
                f"state_rate_{int(rate)}hz",
                state_rate_hz=rate,
            )
        )
    for duration in (0.15, 0.30):
        profiles.append(
            study.current_profile(
                f"state_dropout_{round(duration * 1e3)}ms",
                state_dropout_start_s=0.45,
                state_dropout_duration_s=duration,
            )
        )
    profiles.extend(
        [
            study.current_profile(
                "actuator_kp_kv_0p7",
                actuator_kp_scale=0.7,
                actuator_kv_scale=0.7,
            ),
            study.current_profile(
                "actuator_kp_kv_0p5",
                actuator_kp_scale=0.5,
                actuator_kv_scale=0.5,
            ),
            study.current_profile(
                "actuator_kp_0p7",
                actuator_kp_scale=0.7,
            ),
            study.current_profile(
                "actuator_kv_0p7",
                actuator_kv_scale=0.7,
            ),
            study.current_profile(
                "actuator_kp_kv_1p3",
                actuator_kp_scale=1.3,
                actuator_kv_scale=1.3,
            ),
            study.current_profile(
                "no_measured_state",
                use_measured_state=False,
            ),
        ]
    )
    return profiles


def robustness_scenarios() -> list[study.Scenario]:
    factory = study.PoseFactory()
    return [
        study.make_reach_scenario(factory, speed=0.8),
        study.make_reach_scenario(factory, speed=0.8, lateral=-0.10),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=-0.10,
        ),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=0.10,
        ),
        study.make_extended_circle_scenario(factory, speed=0.8),
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=10.0,
            extended=True,
        ),
        study.make_normal_lissajous_scenario(
            factory,
            speed=0.6,
            mode="bimanual",
        ),
    ]


def _resampled_action(
    record_root: Path,
    episode: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        record_root
        / str(episode)
        / "action"
        / "arms"
        / side
        / "state.parquet",
        columns=["timestamp", "qpos"],
    )
    timestamp = (
        table["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        * 1e-9
    )
    qpos = np.asarray(table["qpos"].to_pylist(), dtype=np.float64)[:, :7]
    keep = np.concatenate([[True], np.diff(timestamp) > 1e-6])
    timestamp = timestamp[keep]
    qpos = qpos[keep]
    uniform_time = np.arange(timestamp[0], timestamp[-1], study.CONTROL_DT)
    uniform_q = np.column_stack(
        [
            np.interp(uniform_time, timestamp, qpos[:, index])
            for index in range(7)
        ]
    )
    return uniform_time, uniform_q


def _window_center(signal: np.ndarray, count: int, excluded: list[int]) -> int:
    squared = np.square(signal)
    kernel = np.ones(count, dtype=np.float64) / count
    score = np.convolve(squared, kernel, mode="same")
    guard = count // 2 + 1
    score[:guard] = -np.inf
    score[-guard:] = -np.inf
    for center in excluded:
        lo = max(0, center - count)
        hi = min(score.size, center + count)
        score[lo:hi] = -np.inf
    return int(np.argmax(score))


def replay_scenarios(record_root: Path) -> list[study.Scenario]:
    """Reconstruct realistic EEF targets from valid intervention command paths."""
    factory = study.PoseFactory()
    scenarios: list[study.Scenario] = []
    window_count = round(3.0 / study.CONTROL_DT)
    hold_count = round(0.35 / study.CONTROL_DT)
    for episode in (187, 189):
        for side in study.SIDES:
            timestamp, qpos = _resampled_action(record_root, episode, side)
            dq = np.zeros_like(qpos)
            dq[1:] = np.diff(qpos, axis=0) / study.CONTROL_DT
            signals = {
                "shoulder_elbow": np.linalg.norm(dq[:, [0, 1, 3]], axis=1),
                "wrist": np.linalg.norm(dq[:, 4:7], axis=1),
            }
            centers: list[int] = []
            for tag, signal in signals.items():
                center = _window_center(signal, window_count, centers)
                centers.append(center)
                start = center - window_count // 2
                stop = start + window_count
                active_q = qpos[start:stop]
                initial_right = study.START_Q_RIGHT.copy()
                initial_left = study.START_Q_LEFT.copy()
                if side == "right":
                    initial_right = active_q[0].copy()
                    right_q = active_q
                    left_q = np.repeat(
                        initial_left[None, :],
                        active_q.shape[0],
                        axis=0,
                    )
                else:
                    initial_left = active_q[0].copy()
                    left_q = active_q
                    right_q = np.repeat(
                        initial_right[None, :],
                        active_q.shape[0],
                        axis=0,
                    )
                right_pose, left_pose = _poses_from_joint_path(
                    factory,
                    right_q,
                    left_q,
                )
                target = right_pose if side == "right" else left_pose
                target = np.vstack(
                    [target, np.repeat(target[-1][None, :], hold_count, axis=0)]
                )
                phase = np.concatenate(
                    [
                        np.ones(active_q.shape[0], dtype=np.int16),
                        np.full(hold_count, 2, dtype=np.int16),
                    ]
                )
                scenario_count = target.shape[0]
                max_joint_speed = float(np.max(np.abs(dq[start:stop])))
                scenarios.append(
                    study.Scenario(
                        name=f"intervention_{episode}_{side}_{tag}",
                        family="intervention_replay",
                        mode=side,
                        speed=max_joint_speed,
                        times=np.arange(scenario_count) * study.CONTROL_DT,
                        phase=phase,
                        target_right=(
                            target
                            if side == "right"
                            else np.full_like(target, np.nan)
                        ),
                        target_left=(
                            target
                            if side == "left"
                            else np.full_like(target, np.nan)
                        ),
                        initial_right=initial_right,
                        initial_left=initial_left,
                        description=(
                            f"FK reconstruction of episode {episode} {side} "
                            f"{tag} command window at "
                            f"{timestamp[start] - timestamp[0]:.2f}s."
                        ),
                    )
                )
    return scenarios


def replay_profiles() -> list[study.Profile]:
    return [
        study.current_profile("current"),
        _current_compact_profile(),
        study.current_profile("no_error_limit", frame_position_error_limit=0.0),
        study.current_profile(
            "minimal_safety",
            nullspace_cost=0.0,
            singularity_approach_limit=False,
            joint_limit_braking=False,
            frame_position_error_limit=0.0,
            kinetic_energy_cost=0.0,
        ),
    ]


def finalist_profiles() -> list[study.Profile]:
    """Cross promising controller settings with realistic plant perturbations."""
    compact = {
        "velocity_caps": (2.5, *study.CURRENT_CAPS[1:]),
        "kinetic_energy_cost": 5e-5,
        "frame_error_limit_linear_slow": 0.6,
        "frame_error_limit_linear_fast": 0.9,
        "frame_error_limit_activation_rise_rate": 1e6,
        "frame_error_limit_activation_fall_rate": 1e6,
        "nullspace_singularity_high": 0.08,
    }
    algorithms = (
        ("current", {}),
        ("compact", compact),
        ("current_ori2", {"orientation_cost": 2.0}),
        ("compact_ori2", {**compact, "orientation_cost": 2.0}),
        (
            "compact_ori2_damping1",
            {**compact, "orientation_cost": 2.0, "damping": 1.0},
        ),
        (
            "lower_task_ori2",
            {
                "position_cost": 5.0,
                "orientation_cost": 2.0,
                "kinetic_energy_cost": 5e-5,
            },
        ),
    )
    perturbations = (
        ("nominal", {}),
        ("cmd24ms", {"command_delay_s": 0.024}),
        (
            "actuator0p7",
            {"actuator_kp_scale": 0.7, "actuator_kv_scale": 0.7},
        ),
        (
            "cmd24ms_actuator0p7",
            {
                "command_delay_s": 0.024,
                "actuator_kp_scale": 0.7,
                "actuator_kv_scale": 0.7,
            },
        ),
    )
    return [
        study.current_profile(
            f"{algorithm}__{perturbation}",
            **algorithm_values,
            **perturbation_values,
        )
        for algorithm, algorithm_values in algorithms
        for perturbation, perturbation_values in perturbations
    ]


def finalist_scenarios(record_root: Path) -> list[study.Scenario]:
    """Combine adversarial, random, and recorded paths for finalist selection."""
    random_by_name = {
        scenario.name: scenario for scenario in randomized_scenarios()
    }
    random_names = (
        "random_normal_right_seed0_v0p35",
        "random_normal_bimanual_seed101_v0p7",
        "random_extended_right_seed200_v0p3",
        "random_extended_right_seed203_v0p3",
    )
    replay = replay_scenarios(record_root)
    replay_selected = [
        scenario
        for scenario in replay
        if scenario.name.startswith("intervention_187_")
    ]
    return [
        *interaction_scenarios(),
        *(random_by_name[name] for name in random_names),
        *replay_selected,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "base_qp",
            "interactions",
            "joint_limits",
            "braking_dynamics",
            "randomized",
            "robustness",
            "replay",
            "finalists",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--save-all-traces", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suites = {
        "base_qp": (base_qp_profiles, core_scenarios),
        "interactions": (interaction_profiles, interaction_scenarios),
        "joint_limits": (joint_limit_profiles, joint_limit_scenarios),
        "braking_dynamics": (
            braking_dynamics_profiles,
            braking_dynamics_scenarios,
        ),
        "randomized": (randomized_profiles, randomized_scenarios),
        "robustness": (robustness_profiles, robustness_scenarios),
        "replay": (
            replay_profiles,
            lambda: replay_scenarios(args.record_root.resolve()),
        ),
        "finalists": (
            finalist_profiles,
            lambda: finalist_scenarios(args.record_root.resolve()),
        ),
    }
    selected = suites if args.suite == "all" else {args.suite: suites[args.suite]}
    for name, (profiles_fn, scenarios_fn) in selected.items():
        profiles = profiles_fn()
        scenarios = scenarios_fn()
        print(
            f"{name}: {len(profiles)} profiles x {len(scenarios)} scenarios",
            flush=True,
        )
        study.run_matrix(
            profiles,
            scenarios,
            args.output_root.resolve() / name,
            workers=args.workers,
            save_all_traces=args.save_all_traces,
        )


if __name__ == "__main__":
    main()
