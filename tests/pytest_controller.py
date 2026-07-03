import pytest
from fyp.controller import URController
import time

@pytest.fixture
def controller():
    c = URController()
    yield c
    c.close()

def test_start_gripper(controller):
    results =  controller.gripper_start()
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

