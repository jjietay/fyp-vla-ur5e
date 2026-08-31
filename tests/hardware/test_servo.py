"""Smoke test for the non-blocking servo path. Requires a UR controller.

THIS MOVES THE ARM. `robot.host` in config.yaml decides which arm, and there is
no simulator behind it any more, so a run here is real metal moving.

Kept deliberately timid, and unlike the older controller tests it commands
NOTHING absolute. Every target is the arm's current pose plus a small offset, so
the test is safe from whatever pose the arm happens to be in and cannot fling it
across the workspace because a hardcoded number suited a different setup. Each
target is also checked against `WorkspaceEnvelope` before it is sent, which is
the same clamp Architecture B applies in the adapter.

What this exists to catch: `servoL` is the one code path Architecture B depends
on that had never been sent to anything. A wrong argument order or a wrong
lookahead does not raise, it just tracks badly, and finding that out in the lab
with a policy attached means debugging two new things at once.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.hardware.ur5e_controller import URController
from fyp.shared.helpers.config import get_config

pytestmark = pytest.mark.integration

# From config.yaml. See the note in test_controller.py.
_T = get_config()["testing"]
SETTLE_S = float(_T["settle_s"])
STEP_M = float(_T["step_m"])
STOPPED_M = float(_T["stopped_tolerance_m"])
# Servo tracking is looser than a completed moveL, so this one doubles the
# configured positional tolerance rather than using it directly.
TOLERANCE_M = float(_T["tolerance_m"]) * 2


@pytest.fixture
def controller():
    """
    It takes nothing and gives you a connected controller that puts the arm back
    where it found it.

    The restore matters more than it looks. Without it a failing test leaves the
    arm 2 cm from where the next test expects it, and the second failure hides
    the first.
    """
    c = URController()
    start = list(c.get_state()["tcp_pose"])
    yield c
    try:
        c.servo_stop()
        c.move_to_pose(start)
    finally:
        c.close()


def _servo_towards(controller, target, seconds: float) -> None:
    """
    It takes a target pose and gives you the arm servoed at it for a while.

    Paced on the wall clock at `servo.time`, NOT with `period_start`/
    `wait_period`. Those hold the RTDE control period, which is 500 Hz, and using
    them here would issue targets 25 times faster than the rate `servo.time`
    declares.
    """
    period = controller.servo_time
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        controller.servo_to_pose(target)
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
    controller.servo_stop()
    time.sleep(SETTLE_S)


def test_servo_converges_on_a_small_relative_target(controller):
    """Re-issuing one target at the control rate should walk the arm to it."""
    start = np.asarray(controller.get_state()["tcp_pose"], dtype=float)
    target = start.copy()
    target[2] += STEP_M  # up, away from the table rather than towards it

    WorkspaceEnvelope().check_pose(target, "servo test target")
    _servo_towards(controller, list(target), seconds=2.0)

    final = np.asarray(controller.get_state()["tcp_pose"], dtype=float)
    assert final[:3] == pytest.approx(target[:3], abs=TOLERANCE_M)


def test_servo_stop_leaves_the_arm_still(controller):
    """
    After `servo_stop` the arm must stop, not coast towards the last target.

    This is the failure that would not show up as an exception anywhere. The
    controller keeps driving to the last servo target until told otherwise, so an
    episode that ends by closing the socket leaves an arm that is still moving.
    """
    start = np.asarray(controller.get_state()["tcp_pose"], dtype=float)
    target = start.copy()
    target[2] += STEP_M

    WorkspaceEnvelope().check_pose(target, "servo test target")
    _servo_towards(controller, list(target), seconds=0.5)

    settled = np.asarray(controller.get_state()["tcp_pose"], dtype=float)
    time.sleep(1.0)
    later = np.asarray(controller.get_state()["tcp_pose"], dtype=float)

    assert later[:3] == pytest.approx(settled[:3], abs=STOPPED_M), (
        "arm kept moving a second after servo_stop")


def test_servo_does_not_block(controller):
    """
    One `servo_to_pose` must return in well under a control period.

    The whole reason this method exists. `move_to_pose` returns when the arm
    arrives, and a policy loop built on that spends every cycle waiting.
    """
    start = np.asarray(controller.get_state()["tcp_pose"], dtype=float)
    target = start.copy()
    target[2] += STEP_M / 2
    WorkspaceEnvelope().check_pose(target, "servo test target")

    t0 = time.monotonic()
    controller.servo_to_pose(list(target))
    elapsed = time.monotonic() - t0

    controller.servo_stop()
    assert elapsed < controller.servo_time, (
        f"servo_to_pose took {elapsed * 1000:.1f} ms, which is not far short of the "
        f"{controller.servo_time * 1000:.0f} ms control period. It is behaving like a "
        f"blocking call.")
