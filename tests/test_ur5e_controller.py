"""Integration tests for the real UR5e controller. Requires URSim running.

Renamed from pytest_controller.py: pytest's default collection pattern is
`test_*.py`, so under the old name these tests were never picked up by a bare
`pytest` run — only by invoking the file directly.

NOTE `test_gripper_state` below is a PRE-EXISTING failure, not something the
restructure introduced. It passes strings ("open"/"close") to gripper_toggle,
which takes int 0/1 and returns "Unable to control gripper." for anything else;
get_state likewise reports 0/1, never "open". It is left exactly as written so
the discrepancy stays visible rather than being quietly papered over — decide
whether the API or the test is wrong, then fix that one.
"""

import time

import pytest

from fyp.shared.hardware.ur5e_controller import URController


@pytest.fixture
def controller():
    c = URController()
    yield c
    c.close()


def test_start_gripper(controller):
    results = controller.gripper_start()
    assert results == "Gripper initialized."


def test_get_state_returns_expected_keys(controller):
    controller.gripper_start()
    state = controller.get_state()
    assert set(state.keys()) == {"joint_pos", "tcp_pose", "gripper_state"}


def test_move_to_pose(controller):
    controller.gripper_start()
    target_pose = [0.0, -0.2329, 0.7294, -0.000, 2.221, -2.221]
    output = controller.move_to_pose(target_pose)
    assert output == True
    time.sleep(5)
    final_pose = controller.get_state()["tcp_pose"]
    assert final_pose == pytest.approx(target_pose, abs=0.01)


def test_move_to_q(controller):
    controller.gripper_start()
    target_position = [0, -1.5708, 0.1745, -1.5708, 0, 0]
    output = controller.move_joints(target_position)
    assert output == True
    time.sleep(5)
    final_position = controller.get_state()["joint_pos"]
    assert final_position == pytest.approx(target_position, abs=0.01)


def test_gripper_state(controller):
    controller.gripper_start()
    controller.gripper_toggle("open")
    assert controller.get_state()["gripper_state"] == "open"

    controller.gripper_toggle("close")
    assert controller.get_state()["gripper_state"] == "close"

    text_output = controller.gripper_toggle("this is not supposed to work")
    assert text_output == "Unable to control gripper."
