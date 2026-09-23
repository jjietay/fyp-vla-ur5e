"""Unit tests for the radius limit in `WorkspaceEnvelope`.

The box itself is exercised through the skills and planner tests. These cover
only the reach limit added on 23 Sep 2026, and they read the REAL config, so a
change to `workspace.max_radius` moves the tests with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope
from fyp.shared.helpers.config import get_config

CFG = get_config()
W = CFG["workspace"]
R = float(W["max_radius"])
Z = (W["z"][0] + W["z"][1]) / 2


@pytest.fixture
def env():
    return WorkspaceEnvelope(CFG)


def test_the_config_makes_the_radius_bind_inside_the_box():
    """
    If the radius were larger than every corner of the box, the limit would never
    fire and these tests would pass for the wrong reason.
    """
    corner = np.hypot(max(abs(v) for v in W["x"]), max(abs(v) for v in W["y"]))
    assert R < corner


def test_a_corner_of_the_box_is_rejected_by_the_radius(env):
    """The case from the bench: inside every axis bound and still out of reach."""
    x, y = W["x"][1], W["y"][1]
    assert np.hypot(x, y) > R
    with pytest.raises(SafetyViolation, match="radius"):
        env.check_point([x, y, Z])
    assert not env.contains([x, y, Z])


@pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 8, endpoint=False))
def test_the_limit_is_round_in_every_direction(env, angle):
    """A point just inside the radius passes and just outside fails, at any bearing."""
    inside = [0.99 * R * np.cos(angle), 0.99 * R * np.sin(angle), Z]
    outside = [1.01 * R * np.cos(angle), 1.01 * R * np.sin(angle), Z]
    if env.contains(inside):  # the box may cut some bearings shorter than R
        env.check_point(inside)
    with pytest.raises(SafetyViolation):
        env.check_point(outside)


def test_the_radius_ignores_height(env):
    """Horizontal on purpose, so the same x and y pass at the bottom and the top of z."""
    xy = [0.9 * R, 0.0]
    env.check_point([*xy, W["z"][0]])
    env.check_point([*xy, W["z"][1]])


def test_a_box_failure_is_reported_by_axis_first(env):
    """A point outside both the box and the radius names the axis, which is more useful."""
    with pytest.raises(SafetyViolation, match=r"x=.*above max"):
        env.check_point([W["x"][1] + 0.1, 0.0, Z])


def test_check_pose_applies_the_radius_too(env):
    """Skills and Architecture B both call check_pose, not check_point."""
    with pytest.raises(SafetyViolation, match="radius"):
        env.check_pose([W["x"][1], W["y"][1], Z, 0.0, 3.14, 0.0])


def test_describe_states_the_radius(env):
    """The planner prompt is built from this, so the LLM is told about the reach limit."""
    assert f"{R:.3f}" in env.describe()


# ------------------------------------------------------------- joint moves

J = float(W["max_joint_step_deg"])
Q0 = [0.1, -1.5, -1.4, -1.7, 1.6, 4.4]


def test_a_small_joint_move_passes(env):
    target = [q + np.radians(J / 2) for q in Q0]
    assert env.check_joint_move(Q0, target) == pytest.approx(target)


def test_a_large_joint_move_is_refused_by_joint(env):
    target = list(Q0)
    target[2] += np.radians(J + 5)
    with pytest.raises(SafetyViolation, match="joint 2"):
        env.check_joint_move(Q0, target)


def test_the_same_wrist_angle_one_turn_away_is_refused(env):
    """The full wrist spin. Same physical pose, joint value 360 degrees away."""
    target = list(Q0)
    target[5] -= 2 * np.pi
    with pytest.raises(SafetyViolation, match="joint 5"):
        env.check_joint_move(Q0, target)


@pytest.mark.parametrize("bad", [[0.0] * 5, [0.0] * 5 + [float("nan")]])
def test_a_malformed_joint_target_is_refused(env, bad):
    with pytest.raises(SafetyViolation):
        env.check_joint_move(Q0, bad)
