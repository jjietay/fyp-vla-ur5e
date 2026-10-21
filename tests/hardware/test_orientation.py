"""Rotation tests at the down pose, against the real UR5e. Requires the arm on RTDE.

THIS MOVES THE ARM, but only by turning the tool a few degrees in place. Park it
with `scripts/go_home.py` first: every test here skips unless the TCP points down.
Nothing may be between the fingers or near the wrist.

## What these settle

Koji reported that with the TCP pointing down, a command to turn it +10 degrees
sometimes goes the long way round. Two things can cause that:

  arithmetic on rotation numbers. At the down pose the rotation vector has a
  length near pi, and adding degrees to its components, or to Euler roll at its
  +pi/-pi wrap point, can describe a very different orientation

  a joint-space move. UR joints turn +/-360 degrees, so one physical wrist angle
  has two joint values, and a target with the other value turns the wrist a
  full circle

Our code avoids the first by building every rotation with rotation matrices, and
guards the second with `WorkspaceEnvelope.check_joint_move`. What was never
checked is what the CONTROLLER does with our targets at this pose:

  a small tool rotation built with matrices must turn the short way, with no
  joint moving more than `testing.rotate_max_joint_deg`

  the current orientation written as the opposite-sign rotation vector, which
  is the same orientation, must not move the arm at all. `R_to_rotvec` can
  produce either sign at the down pose, so Architecture B's servo targets can
  arrive written the other way from the pose the controller reports

If all of these pass, a long turn seen elsewhere comes from how that code builds
its target, not from the controller.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.hardware.ur5e_controller import URController
from fyp.shared.helpers.config import get_config
from fyp.shared.helpers.rotations import R_to_rotvec, rotvec_to_R

pytestmark = pytest.mark.integration

_CFG = get_config()
_T = _CFG["testing"]
ROTATE_DEG = float(_T["rotate_deg"])
MAX_JOINT_DEG = float(_T["rotate_max_joint_deg"])
JOINT_TOL_DEG = float(_T["rotate_joint_tolerance_deg"])
DOWN_TOL_DEG = float(_T["down_tolerance_deg"])
TOLERANCE_RAD = float(_T["tolerance_rad"])
SETTLE_S = float(_T["settle_s"])
DOWN_R = rotvec_to_R(np.asarray(_CFG["architecture_a"]["skills"]["down_rotvec"], dtype=float))


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """It takes two rotation matrices and gives you the angle between them in degrees."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def other_sign(rotvec) -> np.ndarray:
    """
    It takes a rotation vector and gives you the same rotation written the other
    way: the remaining angle, 2*pi minus the angle, about the opposite axis.
    """
    r = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(r))
    return r * (theta - 2.0 * np.pi) / theta


@pytest.fixture
def controller():
    """
    Connected controller that puts the arm back where it found it, by a checked
    joint move, so a failed test cannot leave the wrist turned.
    """
    c = URController()
    start_q = list(c.get_state()["joint_pos"])
    yield c
    try:
        c.servo_stop()
        now = c.get_state()["joint_pos"]
        c.move_joints(WorkspaceEnvelope().check_joint_move(now, start_q, "test restore"))
    finally:
        c.close()


@pytest.fixture
def at_down(controller):
    """It takes the controller and gives it back, or skips unless the TCP points down."""
    R = rotvec_to_R(np.asarray(controller.get_state()["tcp_pose"][3:], dtype=float))
    off = angle_deg(R, DOWN_R)
    if off > DOWN_TOL_DEG:
        pytest.skip(f"TCP is {off:.1f} deg from down_rotvec. Run scripts/go_home.py first")
    return controller


def joints_deg(c) -> np.ndarray:
    return np.degrees(np.asarray(c.get_state()["joint_pos"], dtype=float))


@pytest.mark.parametrize("sign", [+1, -1])
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["tool_x", "tool_y", "tool_z"])
def test_a_small_tool_rotation_turns_the_short_way(at_down, axis, sign):
    """
    Koji's case. Turn the tool by `rotate_deg` about one of its own axes, with the
    target built by multiplying rotation matrices, and watch every joint.

    A long way round turns some joint by about 350 degrees, so the joint bound
    separates the two cases with a wide margin. About tool z, only wrist 3 should
    turn, and by exactly the requested angle.
    """
    c = at_down
    pose0 = np.asarray(c.get_state()["tcp_pose"], dtype=float)
    q0 = joints_deg(c)

    step = np.zeros(3)
    step[axis] = sign * np.radians(ROTATE_DEG)
    # Right-multiplying turns about the TOOL's own axis, not the base's.
    R_target = rotvec_to_R(pose0[3:]) @ rotvec_to_R(step)
    target = [*pose0[:3], *R_to_rotvec(R_target)]
    WorkspaceEnvelope().check_pose(target, "rotation test target")

    assert c.move_to_pose(target) is not False
    time.sleep(SETTLE_S)

    dq = joints_deg(c) - q0
    assert np.max(np.abs(dq)) < MAX_JOINT_DEG, f"a joint took the long way: {np.round(dq, 1)}"
    if axis == 2:
        assert abs(abs(dq[5]) - ROTATE_DEG) < JOINT_TOL_DEG, f"wrist 3 turned {dq[5]:.1f} deg"
        assert np.max(np.abs(dq[:5])) < JOINT_TOL_DEG, f"other joints moved: {np.round(dq, 1)}"

    R_final = rotvec_to_R(np.asarray(c.get_state()["tcp_pose"][3:], dtype=float))
    assert np.radians(angle_deg(R_final, R_target)) < TOLERANCE_RAD


def test_the_same_orientation_written_the_other_way_does_not_move_with_movel(at_down):
    """
    The same orientation, written as the opposite-sign rotation vector, must be
    read as "stay here". If the controller compared the numbers instead of the
    rotations, the wrist would turn most of a circle.
    """
    c = at_down
    pose0 = np.asarray(c.get_state()["tcp_pose"], dtype=float)
    flipped = other_sign(pose0[3:])
    # arccos cannot resolve angles near zero below about 1e-6 deg, so a tighter
    # bound fails on float noise alone (21 Oct 2026, 1.2e-6 deg).
    assert angle_deg(rotvec_to_R(flipped), rotvec_to_R(pose0[3:])) < 1e-4
    q0 = joints_deg(c)

    assert c.move_to_pose([*pose0[:3], *flipped]) is not False
    time.sleep(SETTLE_S)

    dq = joints_deg(c) - q0
    assert np.max(np.abs(dq)) < JOINT_TOL_DEG, f"the arm moved: {np.round(dq, 1)}"


def test_the_same_orientation_written_the_other_way_does_not_move_with_servol(at_down):
    """
    The same check on the servo path, which is the one Architecture B uses. Held
    for one second at the control rate, then stopped.
    """
    c = at_down
    pose0 = np.asarray(c.get_state()["tcp_pose"], dtype=float)
    target = [*pose0[:3], *other_sign(pose0[3:])]
    q0 = joints_deg(c)

    period = c.servo_time
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        c.servo_to_pose(target)
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
    c.servo_stop()
    time.sleep(SETTLE_S)

    dq = joints_deg(c) - q0
    assert np.max(np.abs(dq)) < JOINT_TOL_DEG, f"the arm moved: {np.round(dq, 1)}"
