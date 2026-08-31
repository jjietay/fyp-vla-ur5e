"""Unit tests for Architecture B's inference logic, with no arm and no lerobot.

These cover the code that actually runs when a policy drives the robot. Until the
adapter was split, none of it was reachable from a test: it lived inside a module
that imports `lerobot`, which the FYP venv deliberately does not have.

What is tested here is the sequence that decides where the arm goes. Clamp,
apply the delta in the base frame, check the envelope, threshold the gripper.
Each of those has a failure mode that raises nothing:

  a clamp that clips per axis instead of scaling changes the DIRECTION
  a delta applied on the wrong side of the rotation is right at the identity only
  a check performed before the clamp passes a pose that is then modified
  a gripper threshold on the wrong side of 0.5 inverts every grasp

The real config is used throughout, so these run against the clamps and bounds
that ship rather than against invented ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from fyp.architecture_b.demos.lerobot_export import ACTION_NAMES, STATE_NAMES
from fyp.architecture_b.policy_action import PolicyActionTranslator
from fyp.shared.hardware.safety import SafetyViolation
from fyp.shared.helpers.config import get_config
from fyp.shared.helpers.rotations import euler_to_rotvec, rotvec_to_R

CFG = get_config()
W = CFG["workspace"]

# Middle of the real workspace, so a small delta in any direction stays inside.
CENTRE = np.array([(W["x"][0] + W["x"][1]) / 2,
                   (W["y"][0] + W["y"][1]) / 2,
                   (W["z"][0] + W["z"][1]) / 2])
POSE = np.concatenate([CENTRE, euler_to_rotvec([0.0, 0.0, 0.0])])


@pytest.fixture
def t():
    return PolicyActionTranslator(CFG)


def action(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, gripper=0.0):
    """It takes named deltas and gives you the 7-vector in wire order."""
    return np.array([dx, dy, dz, droll, dpitch, dyaw, gripper], dtype=float)


# ------------------------------------------------------------- the contract

def test_features_use_the_dataset_names(t):
    """
    LeRobot builds `observation.state` by looking each DATASET feature name up in
    what the robot returns, and unpacks the action tensor by the ORDER of
    `action_features`. Both come from the converter's tuples, not from retyping.
    """
    assert list(t.action_features) == list(ACTION_NAMES)
    for name in STATE_NAMES:
        assert t.observation_features[name] is float


def test_observation_carries_the_camera_under_its_configured_name(t):
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = t.observation(POSE, 1, rgb)

    assert set(obs) == set(STATE_NAMES) | {CFG["architecture_b"]["camera"]}
    assert obs[t.camera] is rgb


def test_observation_state_matches_the_pose_it_was_given(t):
    obs = t.observation(POSE, 1, None)
    assert [obs["x"], obs["y"], obs["z"]] == pytest.approx(list(CENTRE))
    assert obs["gripper"] == 1.0


# -------------------------------------------------------------- the clamp

def test_an_oversized_step_is_scaled_not_clipped(t):
    """
    Scaling preserves the direction the policy asked for. Clipping each axis
    separately bends the movement towards the box corners, which is a silent
    change of direction rather than of speed.
    """
    big = action(dx=1.0, dy=1.0, dz=0.0)
    out = t.clamp(big)

    assert np.linalg.norm(out[:3]) == pytest.approx(t.max_step_m)
    # same direction, different length
    assert out[0] == pytest.approx(out[1])


def test_a_small_step_passes_through_untouched(t):
    small = action(dx=t.max_step_m / 10)
    assert t.clamp(small) == pytest.approx(small)


def test_rotation_is_clamped_independently_of_translation(t):
    out = t.clamp(action(dx=10.0, droll=10.0))
    assert np.linalg.norm(out[:3]) == pytest.approx(t.max_step_m)
    assert np.linalg.norm(out[3:6]) == pytest.approx(t.max_step_rad)


def test_the_gripper_element_survives_clamping(t):
    """It is an absolute target, not a delta, so scaling it would be wrong."""
    assert t.clamp(action(dx=10.0, gripper=1.0))[6] == 1.0


# ------------------------------------------------------------- the gripper

@pytest.mark.parametrize("value,expected", [
    (0.0, 0), (0.49, 0), (0.5, 1), (0.51, 1), (1.0, 1),
])
def test_gripper_threshold(t, value, expected):
    assert t.gripper(value) == expected


# ------------------------------------------------------------- translate

def test_translate_moves_the_pose_by_the_requested_delta(t):
    step = t.max_step_m / 2
    target, gripper, clamped = t.translate(POSE, action(dx=step, gripper=1.0))

    assert target[:3] == pytest.approx(POSE[:3] + [step, 0, 0])
    assert gripper == 1
    assert clamped[0] == pytest.approx(step)


def test_translate_returns_the_clamped_action_not_the_request(t):
    """
    LeRobot records what was sent. Returning the request would hide every clamp
    that fired, and a policy driving into a clamp on every step would look
    perfectly healthy in the logs.
    """
    _, _, clamped = t.translate(POSE, action(dx=10.0))
    assert np.linalg.norm(clamped[:3]) == pytest.approx(t.max_step_m)


def test_translate_accepts_the_dict_lerobot_hands_over(t):
    """`robot_client` builds a dict keyed by `action_features`, so both work."""
    step = t.max_step_m / 2
    as_dict = dict(zip(ACTION_NAMES, action(dx=step)))
    from_dict, _, _ = t.translate(POSE, as_dict)
    from_array, _, _ = t.translate(POSE, action(dx=step))
    assert from_dict == pytest.approx(from_array)


def test_a_target_outside_the_workspace_raises(t):
    """
    Must raise rather than clamp into the box. A policy steering out of the
    workspace is a policy doing something wrong, and quietly pinning it to the
    wall would hide that while the arm pressed against its own limit.
    """
    at_edge = np.concatenate([[W["x"][1], CENTRE[1], CENTRE[2]],
                              euler_to_rotvec([0.0, 0.0, 0.0])])
    with pytest.raises(SafetyViolation):
        t.translate(at_edge, action(dx=t.max_step_m))


def test_the_check_happens_after_the_clamp(t):
    """
    Order matters. An enormous step that lands outside the box must be clamped
    first and then pass, because the arm will only ever be given the clamped one.
    Checking before clamping would reject a step the arm was never going to take.
    """
    target, _, _ = t.translate(POSE, action(dx=100.0))
    assert W["x"][0] <= target[0] <= W["x"][1]


def test_orientation_delta_is_applied_in_the_base_frame(t):
    """
    Same rotation delta from two differently turned starts must produce the same
    CHANGE. Applying it on the right instead is correct at the identity and wrong
    everywhere else, which is why it survives a casual test.
    """
    turned = np.concatenate([CENTRE, euler_to_rotvec([0.0, 0.0, 1.2])])
    step = action(droll=t.max_step_rad / 2)

    from_upright, _, _ = t.translate(POSE, step)
    from_turned, _, _ = t.translate(turned, step)

    delta_a = rotvec_to_R(from_upright[3:]) @ rotvec_to_R(POSE[3:]).T
    delta_b = rotvec_to_R(from_turned[3:]) @ rotvec_to_R(turned[3:]).T
    np.testing.assert_allclose(delta_a, delta_b, atol=1e-9)


def test_a_zero_action_holds_position(t):
    target, gripper, _ = t.translate(POSE, action())
    assert target[:3] == pytest.approx(POSE[:3])
    assert gripper == 0


# ------------------------------------------------ the two architectures agree

def test_b_can_never_move_faster_than_a_is_allowed_to(t):
    """
    The clamp is derived from `workspace.max_speed`, not written down, so this
    holds whatever the config says. It exists because the per-step distance was
    once a literal, and 0.05 m at 20 Hz is 1.0 m/s against Architecture A's 0.25.

    `shared/` exists so both architectures are constrained identically. A speed
    ceiling that applies to one of them is the same class of mistake as two
    different safety envelopes.
    """
    saturated = t.clamp(action(dx=1000.0, droll=1000.0))

    linear = np.linalg.norm(saturated[:3]) / t.control_period_s
    angular = np.linalg.norm(saturated[3:6]) / t.control_period_s

    assert linear <= CFG["workspace"]["max_speed"] + 1e-12
    assert angular <= CFG["workspace"]["max_angular_speed"] + 1e-12


def test_the_step_ceiling_is_one_control_period(t):
    """The rule itself, so a refactor cannot quietly replace it with a constant."""
    assert t.max_step_m == pytest.approx(
        CFG["workspace"]["max_speed"] * CFG["robot"]["servo"]["time"])
    assert t.max_step_rad == pytest.approx(
        CFG["workspace"]["max_angular_speed"] * CFG["robot"]["servo"]["time"])
