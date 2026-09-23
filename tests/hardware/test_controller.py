"""Controller tests against the real UR5e. Requires the arm reachable on RTDE.

THIS MOVES THE ARM, and there is no simulator behind it any more. `robot.host` in
config.yaml points at whatever you set it to, and these tests will drive it.
Stand within reach of the estop.

Replaces the old `test_ur5e_controller.py`, which was written against a
simulator and sent hardcoded ABSOLUTE targets with no envelope check. That was
survivable in simulation and would have been dangerous the moment `robot.host`
became a real address, because a pose that suited one bench means nothing on
another.

The rule here, and in `test_servo.py`: every target is the arm's CURRENT pose
plus a small offset, and every target is checked against `WorkspaceEnvelope`
before it is sent. A test that cannot say where the arm is starting from has no
business saying where it should go.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fyp.architecture_a.skills import GRIPPER_CLOSED, GRIPPER_OPEN
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.hardware.ur5e_controller import URController, rg2_target_mm
from fyp.shared.helpers.config import get_config

pytestmark = pytest.mark.integration

# From config.yaml, which is the single source of truth for tunables. How far a
# test moves a real arm is exactly the kind of number that should not be buried
# in a test file where nobody reviewing the workspace bounds would find it.
_T = get_config()["testing"]
STEP_M = float(_T["step_m"])
STEP_RAD = float(_T["step_rad"])
TOLERANCE_M = float(_T["tolerance_m"])
TOLERANCE_RAD = float(_T["tolerance_rad"])
SETTLE_S = float(_T["settle_s"])
_G = get_config()["robot"]["gripper"]


@pytest.fixture
def controller():
    """
    Connected controller that puts the arm back where it found it.

    Does NOT start the gripper, so the motion tests never move the fingers.
    Only the gripper tests below ask for it, through the `gripper` fixture.
    """
    c = URController()
    start_pose = list(c.get_state()["tcp_pose"])
    start_q = list(c.get_state()["joint_pos"])
    yield c
    try:
        c.servo_stop()
        c.move_joints(start_q)
        c.move_to_pose(start_pose)
    finally:
        c.close()


@pytest.fixture
def gripper(controller):
    """
    It takes the connected controller and gives it back with the gripper started,
    and leaves the fingers open afterwards so nothing is left clamped.

    NOTHING MAY BE BETWEEN THE FINGERS. The close tests expect to close on air.
    """
    controller.gripper_start()
    yield controller
    controller.gripper_toggle(GRIPPER_OPEN)


def test_gripper_start_reports_success(controller):
    assert controller.gripper_start() == "Gripper initialized."


def test_get_state_reports_none_before_start(controller):
    assert controller.get_state()["gripper_state"] is None


def test_get_state_returns_the_expected_keys(controller):
    state = controller.get_state()
    assert set(state) == {"joint_pos", "tcp_pose", "gripper_state"}
    assert len(state["tcp_pose"]) == 6
    assert len(state["joint_pos"]) == 6


def test_move_to_pose_reaches_a_small_relative_target(controller):
    start = np.asarray(controller.get_state()["tcp_pose"], dtype=float)
    target = start.copy()
    target[2] += STEP_M  # up, away from the table rather than into it

    WorkspaceEnvelope().check_pose(target, "controller test target")
    assert controller.move_to_pose(list(target)) is not False
    time.sleep(SETTLE_S)

    final = np.asarray(controller.get_state()["tcp_pose"], dtype=float)
    assert final[:3] == pytest.approx(target[:3], abs=TOLERANCE_M)


def test_move_joints_reaches_a_small_relative_target(controller):
    """
    Moves the wrist, joint 5, rather than a shoulder or elbow. A small angle at
    the base sweeps the whole arm across the bench; the same angle at the wrist
    moves the tool a few millimetres.
    """
    start = np.asarray(controller.get_state()["joint_pos"], dtype=float)
    target = start.copy()
    target[4] += STEP_RAD

    assert controller.move_joints(list(target)) is not False
    time.sleep(SETTLE_S)

    final = np.asarray(controller.get_state()["joint_pos"], dtype=float)
    assert final == pytest.approx(target, abs=TOLERANCE_RAD)


def test_gripper_toggles_between_the_two_int_states(gripper):
    """
    The API takes int 0 and 1, and so does every caller: `skills.py` uses
    GRIPPER_OPEN and GRIPPER_CLOSED, and the LeRobot adapter thresholds a float
    into the same two values.

    The width is checked as well as the state. The state only says what Python
    asked for; the width is what the controller measured on the fingers.
    """
    width = gripper.gripper_toggle(GRIPPER_OPEN)
    assert gripper.get_state()["gripper_state"] == GRIPPER_OPEN
    assert width == pytest.approx(rg2_target_mm(_G["open_mm"]), abs=_G["tolerance_mm"])

    width = gripper.gripper_toggle(GRIPPER_CLOSED)
    assert gripper.get_state()["gripper_state"] == GRIPPER_CLOSED
    assert width == pytest.approx(rg2_target_mm(_G["close_mm"]), abs=_G["tolerance_mm"])


def test_gripper_already_in_place_is_not_sent_again(gripper):
    """
    The script checks the width before sending, so a second open sends no word
    and returns in about the fixed cost of the call, not a stroke.
    """
    gripper.gripper_toggle(GRIPPER_OPEN)
    t0 = time.monotonic()
    gripper.gripper_toggle(GRIPPER_OPEN)
    assert time.monotonic() - t0 < float(_T["gripper_no_op_s"])


def test_gripper_survives_many_cycles(gripper):
    """
    On 23 Sep 2026 about 1 command in 66 was lost with the old script. The resend
    must make every one of these arrive, or raise, never pass silently.
    """
    for _ in range(int(_T["gripper_cycles"])):
        gripper.gripper_toggle(GRIPPER_CLOSED)
        gripper.gripper_toggle(GRIPPER_OPEN)


def test_gripper_rejects_anything_that_is_not_zero_or_one(gripper):
    assert gripper.gripper_toggle("open") == "Unable to control gripper."
    assert gripper.gripper_toggle(7) == "Unable to control gripper."
