"""Unit tests for the action space shared by training and inference.

The one thing worth testing before there is a robot or a checkpoint. Everything
else in Architecture B fails loudly: a wrong camera name raises a KeyError, an
unreachable arm refuses the connection. A rotation convention that disagrees
between the converter and the adapter fails silently, produces a dataset that
trains and a policy that turns the wrong way, and by then the evidence is hours
of lab time in the past.

No hardware, no lerobot, no torch. This runs anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

from fyp.architecture_b.demos.lerobot_export import (ACTION_NAMES, STATE_NAMES,
                                                     apply_delta_action,
                                                     build_state,
                                                     compute_delta_action)
from fyp.shared.helpers.rotations import euler_to_rotvec, rotvec_to_euler, rotvec_to_R

RNG = np.random.default_rng(0)


def random_pose() -> np.ndarray:
    """It takes nothing and gives you a plausible TCP pose, position plus rotvec."""
    xyz = RNG.uniform(-0.5, 0.5, size=3)
    # Kept away from +/- pi/2 pitch, where the Euler decomposition is singular and
    # the round trip is not expected to hold. Real grasp poses do not live there.
    rpy = RNG.uniform(-1.0, 1.0, size=3)
    return np.concatenate([xyz, euler_to_rotvec(rpy)])


# ------------------------------------------------------------- the contract

def test_names_are_the_expected_order():
    """The wire order. robot_client unpacks the action tensor by index."""
    assert STATE_NAMES == ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21",
                           "gripper")
    assert ACTION_NAMES == ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def test_state_vector_matches_state_names():
    vector = build_state(random_pose(), 1)
    assert len(vector) == len(STATE_NAMES)
    assert vector.dtype == np.float32


def test_state_columns_are_the_first_two_columns_of_R():
    """The names promise R[i, j] in column order, so check each one by name."""
    pose = random_pose()
    R = rotvec_to_R(pose[3:])
    state = dict(zip(STATE_NAMES, build_state(pose, 0)))
    for i in range(3):
        for j in range(2):
            assert state[f"r{i}{j}"] == pytest.approx(R[i, j], abs=1e-6)


def test_state_is_continuous_through_the_roll_flip():
    """
    The reason the state is not Euler. With the gripper pointing down, a tilt of
    a hundredth of a radian either side moves Euler roll from about +pi to about
    -pi. The 6D state of the same two poses must stay close, because the arm
    barely moved.
    """
    xyz = [0.3, -0.3, 0.3]
    a = np.concatenate([xyz, euler_to_rotvec([np.pi - 0.005, 0.0, 0.3])])
    b = np.concatenate([xyz, euler_to_rotvec([-np.pi + 0.005, 0.0, 0.3])])

    # First confirm the two poses really do sit either side of the flip.
    assert abs(rotvec_to_euler(a[3:])[0] - rotvec_to_euler(b[3:])[0]) > 6.0

    assert np.max(np.abs(build_state(a, 1) - build_state(b, 1))) < 0.02


def test_state_columns_are_unit_and_orthogonal():
    """Two columns of a rotation matrix, so the policy never sees an impossible state."""
    state = build_state(random_pose(), 1)
    c0, c1 = np.asarray(state[3:6], dtype=float), np.asarray(state[6:9], dtype=float)
    assert np.linalg.norm(c0) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(c1) == pytest.approx(1.0, abs=1e-6)
    assert float(c0 @ c1) == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------- round trip

@pytest.mark.parametrize("trial", range(20))
def test_delta_round_trip_recovers_the_next_pose(trial):
    """
    compute_delta_action then apply_delta_action must give back where we started.

    This is the test that catches a left/right multiplication mistake in the
    rotation delta. Position would still be right, so the failure looks like a
    calibration problem rather than like a maths problem.
    """
    pose_t, pose_next = random_pose(), random_pose()

    action = compute_delta_action(pose_t, pose_next, gripper_next=1)
    recovered = apply_delta_action(pose_t, action)

    # Tolerance is float32 sized on purpose. `compute_delta_action` casts to
    # float32 because that is what the dataset stores, so about 1e-7 of the
    # round trip error is the storage format rather than the arithmetic. A
    # tighter tolerance here would fail on correct code.
    np.testing.assert_allclose(recovered[:3], pose_next[:3], atol=1e-6)
    # Compared as rotation matrices, because a rotation vector and the same
    # rotation turned the long way round are equal rotations with different
    # numbers, and comparing the vectors would fail on a correct result.
    np.testing.assert_allclose(rotvec_to_R(recovered[3:]),
                               rotvec_to_R(pose_next[3:]), atol=1e-6)


def test_zero_action_does_not_move_the_arm():
    """An idle policy must hold position, not drift."""
    pose = random_pose()
    recovered = apply_delta_action(pose, np.zeros(7))

    np.testing.assert_allclose(recovered[:3], pose[:3], atol=1e-12)
    np.testing.assert_allclose(rotvec_to_R(recovered[3:]),
                               rotvec_to_R(pose[3:]), atol=1e-12)


def test_translation_is_base_frame_not_tool_frame():
    """
    The delta position adds in the base frame, so it does not depend on how the
    tool is turned. Getting this wrong makes the arm move correctly at one
    orientation and diagonally at every other one.
    """
    xyz = np.array([0.1, -0.2, 0.3])
    upright = np.concatenate([xyz, euler_to_rotvec([0.0, 0.0, 0.0])])
    turned = np.concatenate([xyz, euler_to_rotvec([0.0, 0.0, 1.2])])

    step = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(apply_delta_action(upright, step)[:3],
                               apply_delta_action(turned, step)[:3], atol=1e-12)


def test_gripper_element_is_ignored_by_the_pose_maths():
    """The gripper is an absolute target and is commanded separately."""
    pose = random_pose()
    opened = apply_delta_action(pose, np.array([0.01, 0, 0, 0, 0, 0, 1.0]))
    closed = apply_delta_action(pose, np.array([0.01, 0, 0, 0, 0, 0, 0.0]))

    np.testing.assert_allclose(opened, closed, atol=1e-12)
