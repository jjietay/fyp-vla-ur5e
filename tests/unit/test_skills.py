"""Unit tests for the motion primitives, with a fake controller and no arm.

`skills.py` is the largest thing in Architecture A that moves the arm, and until
now it had no test at all. Everything it does can be checked without hardware,
because it only ever talks to the controller through five methods, so a fake that
records what it was asked to do exercises the whole file.

What is worth testing here is not "does pick pick". That needs an object. It is
the set of properties that make a skill safe and inspectable:

  every commanded pose passed the envelope
  the gripper opens and closes at the right point in the sequence
  a refused pose is a failure and not a silently skipped waypoint
  a skill that cannot run does not move the arm first and fail second

Each of those is a bug that would otherwise be found with a real arm and a real
object on the table.
"""

from __future__ import annotations

import numpy as np
import pytest

from fyp.architecture_a.skills import GRIPPER_CLOSED, GRIPPER_OPEN, Skills
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.hardware.ur5e_controller import GripperError, URController
from fyp.shared.helpers.config import get_config

# The REAL config. Testing skills against invented workspace bounds and invented
# approach heights proves they work on numbers the arm will never use, and the
# approach height in particular decides whether a waypoint clears the table.
CFG = get_config()
W = CFG["workspace"]

# Derived, so editing config.yaml moves the target with it. Low in the box, so
# that adding retreat_height still leaves it inside.
TARGET = [(W["x"][0] + W["x"][1]) / 2,
          (W["y"][0] + W["y"][1]) / 2,
          W["z"][0] + 0.05]


@pytest.fixture(autouse=True)
def no_settling(monkeypatch):
    """
    It takes nothing and gives you skills that do not sleep.

    `pour` holds the tilt for `hold_s` seconds. Patching the sleep rather than
    passing a smaller number keeps every number under test real and neutralises
    only the waiting, which has no bearing on whether the logic is correct.
    """
    monkeypatch.setattr("fyp.architecture_a.skills.time.sleep", lambda _s: None)


class FakeController:
    """
    It takes nothing and gives you a controller that records instead of moving.

    `refuse_after` reproduces the one controller behaviour that is easy to ignore:
    `moveL` returning False rather than raising. A skill that does not check it
    carries on to the next waypoint as though the arm had arrived.

    `gripper_fails_on` makes that gripper state raise `GripperError`, as the real
    controller does when the RG2 does not respond.

    `held_mm` is the width a close stops at, which is what the real controller
    returns. 0 is a close on nothing; the default is something held.
    """

    def __init__(self, refuse_after: int | None = None, gripper_fails_on: int | None = None,
                 joint_pos=None, held_mm=40.0):
        self.held_mm = held_mm
        self.calls: list[tuple] = []
        self.joint_pos = [0.0] * 6 if joint_pos is None else [float(v) for v in joint_pos]
        self.refuse_after = refuse_after
        self.gripper_fails_on = gripper_fails_on
        self._moves = 0

    def move_to_pose(self, pose, speed=None, acc=None):
        self.calls.append(("move_to_pose", list(pose), speed, acc))
        self._moves += 1
        if self.refuse_after is not None and self._moves > self.refuse_after:
            return False
        return True

    def move_joints(self, q, speed=None, acc=None):
        self.calls.append(("move_joints", list(q), speed, acc))
        return True

    def gripper_toggle(self, state):
        self.calls.append(("gripper_toggle", state))
        if state == self.gripper_fails_on:
            raise GripperError("NO_RESPONSE, width 99.8 mm")
        return 99.8 if state == GRIPPER_OPEN else self.held_mm

    def get_state(self):
        self.calls.append(("get_state",))
        return {"joint_pos": list(self.joint_pos), "tcp_pose": [0.0] * 6,
                "gripper_state": GRIPPER_OPEN}

    @property
    def sequence(self) -> list[str]:
        """Just the call names, for asserting on ordering."""
        return [c[0] for c in self.calls]

    @property
    def poses(self) -> list[list[float]]:
        return [c[1] for c in self.calls if c[0] == "move_to_pose"]


@pytest.fixture
def ctrl():
    return FakeController()


@pytest.fixture
def skills(ctrl):
    return Skills(ctrl, cfg=CFG)


@pytest.fixture
def env():
    return WorkspaceEnvelope(CFG)


# ------------------------------------------------------- the safety property

@pytest.mark.parametrize("run", [
    lambda s: s.pick(TARGET),
    lambda s: s.place(TARGET),
    lambda s: s.open_drawer(TARGET),
    lambda s: s.pour(TARGET, [TARGET[0] - 0.2, TARGET[1], TARGET[2]],
                     hold_s=0.0, steps=3),
])
def test_every_commanded_pose_is_inside_the_envelope(skills, ctrl, env, run):
    """
    The one property that must hold for all of them, including intermediate
    waypoints. An approach pose that clips the table breaks the arm exactly as
    thoroughly as a grasp pose that does.
    """
    result = run(skills)
    assert result.ok, result.message
    assert ctrl.poses, "skill commanded nothing at all"
    for pose in ctrl.poses:
        assert env.contains(pose[:3]), f"{pose[:3]} escaped the envelope"


@pytest.mark.parametrize("run", [
    lambda s: s.pick(TARGET),
    lambda s: s.place(TARGET),
    lambda s: s.open_drawer(TARGET),
])
def test_speed_never_exceeds_the_configured_ceiling(skills, ctrl, run):
    run(skills)
    for _, _, speed, acc in [c for c in ctrl.calls if c[0] == "move_to_pose"]:
        assert speed <= W["max_speed"]
        assert acc <= W["max_acc"]


# -------------------------------------------------------------------- pick

def test_pick_opens_then_descends_then_closes(skills, ctrl):
    """
    Ordering, which no amount of position checking would catch. Closing before
    the descent brings the gripper down shut, and it pushes the object away
    rather than picking it up.
    """
    assert skills.pick(TARGET).ok
    assert ctrl.sequence == ["gripper_toggle", "move_to_pose", "move_to_pose",
                             "gripper_toggle", "move_to_pose"]
    grips = [c[1] for c in ctrl.calls if c[0] == "gripper_toggle"]
    assert grips == [GRIPPER_OPEN, GRIPPER_CLOSED]


def test_pick_approaches_from_directly_above(skills, ctrl):
    """
    A diagonal approach sweeps the gripper through the space beside the target,
    which is exactly where the other objects are.
    """
    assert skills.pick(TARGET).ok
    approach, grasp = ctrl.poses[0], ctrl.poses[1]
    assert approach[:2] == pytest.approx(grasp[:2])
    assert approach[2] > grasp[2]


def test_pick_retreats_higher_than_it_approached(skills, ctrl):
    assert skills.pick(TARGET).ok
    assert ctrl.poses[-1][2] > ctrl.poses[0][2]


def test_grasp_offset_lowers_the_grasp_point(ctrl):
    skills = Skills(ctrl, cfg=CFG)
    assert skills.pick(TARGET, grasp_offset=0.01).ok
    assert ctrl.poses[1][2] == pytest.approx(TARGET[2] - 0.01)


def test_a_grasp_the_offset_would_put_under_the_floor_happens_at_the_floor(ctrl):
    """
    13 Oct 2026: a short orange read its centre 13 mm above the floor, the 28 mm
    offset asked for a grasp below it, and every pick was refused. The offset
    means "as low as is safe", so the grasp goes to the floor instead.
    """
    skills = Skills(ctrl, cfg=CFG)
    low = [TARGET[0], TARGET[1], W["z"][0] + 0.013]
    assert skills.pick(low, grasp_offset=0.028).ok
    assert ctrl.poses[1][2] == pytest.approx(W["z"][0])


def test_place_whose_retreat_is_over_the_ceiling_retreats_to_the_ceiling(ctrl):
    """13 Oct 2026: a box at z 0.258 put the retreat over the 0.38 ceiling."""
    skills = Skills(ctrl, cfg=CFG)
    high = [TARGET[0], TARGET[1], W["z"][1] - 0.13]
    assert skills.place(high).ok
    assert ctrl.poses[-1][2] == pytest.approx(W["z"][1])


def test_place_whose_approach_is_over_the_ceiling_approaches_from_the_ceiling(ctrl):
    """13 Oct 2026: a box at z 0.278 asked for an approach at 0.398, over 0.38."""
    skills = Skills(ctrl, cfg=CFG)
    result = skills.place([TARGET[0], TARGET[1], W["z"][1] - 0.10], approach_height=0.12)
    assert result.ok
    assert ctrl.poses[0][2] == pytest.approx(W["z"][1])
    assert ctrl.poses[0][2] > ctrl.poses[1][2]      # still comes DOWN to the drop


def test_an_unfinishable_place_is_refused_before_anything_moves(ctrl):
    """
    A drop point over the ceiling cannot be reached however the approach is
    clamped. Refusing before the first motion means the gripper never opens over
    the wrong spot.
    """
    skills = Skills(ctrl, cfg=CFG)
    result = skills.place([TARGET[0], TARGET[1], W["z"][1] + 0.01])
    assert not result.ok
    assert ctrl.poses == []
    assert "gripper_toggle" not in ctrl.sequence


def test_a_close_that_never_happened_is_not_a_hold():
    """
    13 Oct 2026: the fingers stayed open and the pick reported success. A close
    that ends at the open width means the command never ran.
    """
    ctrl = FakeController(held_mm=99.0)
    result = Skills(ctrl, cfg=CFG).pick(TARGET)
    assert not result.ok
    assert "did not happen" in result.message


def test_a_centre_below_the_floor_is_still_refused(ctrl):
    """A centre under the floor is a bad detection, not a short object."""
    skills = Skills(ctrl, cfg=CFG)
    result = skills.pick([TARGET[0], TARGET[1], W["z"][0] - 0.01])
    assert not result.ok
    assert "z" in result.message


# ------------------------------------------------------------------- place

def test_place_releases_above_the_surface(skills, ctrl):
    """
    Pressing the object into the table before opening either jams the fingers or
    flips the object.
    """
    assert skills.place(TARGET, release_clearance=0.01).ok
    drop = ctrl.poses[1]
    assert drop[2] == pytest.approx(TARGET[2] + 0.01)


def test_place_opens_only_after_arriving(skills, ctrl):
    assert skills.place(TARGET).ok
    assert ctrl.sequence.index("gripper_toggle") > ctrl.sequence.index("move_to_pose")


# ------------------------------------------------------------------ failure

def test_target_outside_the_envelope_fails_without_moving(skills, ctrl):
    """
    Fails first, moves never. A skill that commands the approach and then
    discovers the grasp is out of bounds has already put the arm somewhere
    unexpected.
    """
    result = skills.pick([TARGET[0], TARGET[1], W["z"][1] + 1.0])
    assert not result.ok
    assert "z" in result.message
    assert ctrl.poses == []


def test_a_refused_pose_is_a_failure_not_a_skipped_waypoint():
    """
    `moveL` returns False when the controller rejects a target. Ignoring that
    return is an hour spent wondering why the arm skipped a waypoint.
    """
    ctrl = FakeController(refuse_after=1)
    result = Skills(ctrl, cfg=CFG).pick(TARGET)
    assert not result.ok
    assert "refused" in result.message


def test_pour_stops_when_the_grasp_failed(ctrl):
    """
    A skill reports failure by RETURNING `ok=False`, not by raising, so a caller
    that ignores the return value carries on as if it worked. `pour` calls `pick`
    first, and without this check it would tilt an empty gripper over the glass
    and report a pour that never happened, which scores as a spill.

    `refuse_after=1` lets the approach pose through and refuses the descent, which
    is where a real grasp fails.
    """
    ctrl = FakeController(refuse_after=1)
    target = [TARGET[0] - 0.2, TARGET[1], TARGET[2]]
    result = Skills(ctrl, cfg=CFG).pour(TARGET, target, hold_s=0.0, steps=3)

    assert not result.ok
    assert "grasp" in result.message
    # It must not have gone on to the receptacle after the grasp failed.
    assert all(pose[0] != pytest.approx(target[0]) for pose in ctrl.poses)


def test_pour_reports_the_waypoints_of_the_skills_it_calls(skills, ctrl):
    """
    `pour` is the only skill built out of other skills, so its trace is the only
    one that can silently lose poses. The trace is the evidence for the
    per-stage-diagnosability claim, so it has to cover the whole motion.
    """
    result = skills.pour(TARGET, [TARGET[0] - 0.2, TARGET[1], TARGET[2]],
                         hold_s=0.0, steps=3)
    assert result.ok, result.message
    assert result.waypoints == ctrl.poses


def test_open_drawer_rejects_a_bad_axis_without_touching_the_arm(skills, ctrl):
    result = skills.open_drawer(TARGET, pull_axis="sideways")
    assert not result.ok
    assert ctrl.calls == []


# -------------------------------------------------------------- open_drawer

def test_open_drawer_stands_off_then_pulls_the_named_way(skills, ctrl):
    assert skills.open_drawer(TARGET, pull_distance=0.20, pull_axis="-x").ok
    approach, handle, pulled = ctrl.poses[0], ctrl.poses[1], ctrl.poses[2]
    # Stand off on the far side of the handle from the pull, then pull through it.
    assert approach[0] > handle[0]
    assert pulled[0] == pytest.approx(TARGET[0] - 0.20)


# ---------------------------------------------------------------- inspection

def test_skills_touch_only_the_shared_controller_api(skills, ctrl):
    """
    The rule that keeps the comparison honest. A skill reaching past the
    controller for a ur_rtde handle cannot be given to Architecture B, and the
    experiment starts measuring plumbing.
    """
    for run in (lambda s: s.pick(TARGET), lambda s: s.place(TARGET),
                lambda s: s.open_drawer(TARGET)):
        run(skills)
    allowed = {"move_to_pose", "move_joints", "gripper_toggle", "get_state"}
    assert set(ctrl.sequence) <= allowed


def test_result_records_every_waypoint_it_commanded(skills, ctrl):
    """The trace is the architecture's selling point, so it has to be complete."""
    result = skills.pick(TARGET)
    assert result.waypoints == ctrl.poses


def test_home_fails_cleanly_when_home_q_is_not_configured(ctrl):
    """
    Without `robot.home_q`, `home` cannot know where to go. It must say so rather
    than move the arm somewhere arbitrary.

    The config handed in is the real one with `home_q` removed. `home` reads the
    config it was given, so this stays true now that the real config has one.
    """
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["robot"].pop("home_q", None)
    result = Skills(ctrl, cfg=cfg).home()
    assert not result.ok
    assert "home_q" in result.message
    assert ctrl.calls == []


HOME_Q = [float(v) for v in CFG["robot"]["home_q"]]


def test_home_moves_the_joints_to_the_configured_home_q():
    """From near home: one joint move to exactly the configured angles, checked first."""
    near = [q + 0.05 for q in HOME_Q]
    ctrl = FakeController(joint_pos=near)
    result = Skills(ctrl, cfg=CFG).home()
    assert result.ok
    assert ctrl.sequence == ["get_state", "move_joints"]
    assert ctrl.calls[-1][1] == pytest.approx(HOME_Q)


def test_home_refuses_to_spin_the_wrist_a_full_turn():
    """
    Wrist 3 at home_q minus 360 degrees is the same physical pose, and a joint
    move from there would turn the wrist a full circle. `home` must refuse.
    """
    import math
    spun = list(HOME_Q)
    spun[5] -= 2 * math.pi
    ctrl = FakeController(joint_pos=spun)
    result = Skills(ctrl, cfg=CFG).home()
    assert not result.ok
    assert "joint 5" in result.message
    assert "move_joints" not in ctrl.sequence


# ------------------------------------------------- the fake cannot drift

def test_fake_controller_matches_the_real_one():
    """
    The fake stands in for `URController`, so its signatures have to match.

    Without this, changing an argument on the real controller leaves the fake
    behind and every test in this file keeps passing against an interface that no
    longer exists. That is worse than no test, because it reads as coverage.

    Compared by name and parameter list, not by behaviour. The whole point of the
    fake is that it behaves differently.
    """
    import inspect

    for name in ("move_to_pose", "move_joints", "gripper_toggle", "get_state"):
        real = inspect.signature(getattr(URController, name))
        fake = inspect.signature(getattr(FakeController, name))
        assert list(real.parameters) == list(fake.parameters), (
            f"{name}: URController takes {list(real.parameters)}, "
            f"FakeController takes {list(fake.parameters)}")


# ------------------------------------------------- a gripper that does not respond

@pytest.mark.parametrize("run", [
    lambda s: s.pick(TARGET),
    lambda s: s.place(TARGET),
    lambda s: s.pour(TARGET, TARGET),
    lambda s: s.open_drawer(TARGET),
])
def test_a_gripper_failure_is_a_failed_result_not_a_crash(run):
    """
    `GripperError` from the controller must come back as `ok=False` with the
    reason, like every other skill failure. If it escaped, the planner would
    never see it and the run would stop with the arm wherever it was.
    """
    for state in (GRIPPER_OPEN, GRIPPER_CLOSED):
        ctrl = FakeController(gripper_fails_on=state)
        result = run(Skills(ctrl, cfg=CFG))
        if ("gripper_toggle", state) not in ctrl.calls:
            continue  # this skill never commands that state
        assert not result.ok
        assert "gripper did not respond" in result.message


def test_pick_stops_moving_after_a_failed_close():
    """No lift after a close that failed: the arm stays at the grasp pose."""
    ctrl = FakeController(gripper_fails_on=GRIPPER_CLOSED)
    Skills(ctrl, cfg=CFG).pick(TARGET)
    assert ctrl.sequence[-1] == "gripper_toggle"


# ------------------------------------------------- a close on nothing

def test_a_close_on_nothing_is_a_failed_pick():
    """
    The fingers travelling all the way in is a normal return from the gripper,
    so only the width can say the pick missed. A pick that reports success here
    sends the planner on to place an object the arm is not holding.
    """
    result = Skills(FakeController(held_mm=0.0), cfg=CFG).pick(TARGET)
    assert not result.ok
    assert "grasp missed" in result.message


def test_a_held_object_is_reported_with_its_width(skills):
    result = skills.pick(TARGET)
    assert result.ok, result.message
    assert "holding 40 mm" in result.message


def test_a_missed_grasp_opens_and_backs_off_above_the_target():
    """
    After a miss the arm must not stay at the grasp pose, where the gripper hides
    the object from the fixed camera during the re-perception that follows.
    """
    ctrl = FakeController(held_mm=0.0)
    Skills(ctrl, cfg=CFG).pick(TARGET)
    grippers = [c[1] for c in ctrl.calls if c[0] == "gripper_toggle"]
    assert grippers[-1] == GRIPPER_OPEN
    assert ctrl.sequence[-1] == "move_to_pose"
    assert ctrl.poses[-1][2] > TARGET[2]


def test_the_threshold_is_the_configured_one():
    """A width just above the threshold is a hold, just below it is a miss."""
    limit = float(CFG["architecture_a"]["skills"]["min_grasp_width_mm"])
    assert Skills(FakeController(held_mm=limit + 0.1), cfg=CFG).pick(TARGET).ok
    assert not Skills(FakeController(held_mm=limit - 0.1), cfg=CFG).pick(TARGET).ok


def test_no_width_is_not_taken_as_a_grasp():
    """No number is no evidence, and guessing "held" drops things from a height."""
    result = Skills(FakeController(held_mm=None), cfg=CFG).pick(TARGET)
    assert not result.ok
    assert "cannot be confirmed" in result.message


def test_pour_stops_when_the_grasp_missed():
    """Nothing is tilted over the target when the container was never picked up."""
    ctrl = FakeController(held_mm=0.0)
    result = Skills(ctrl, cfg=CFG).pour(TARGET, [TARGET[0] - 0.2, TARGET[1], TARGET[2]],
                                        hold_s=0.0, steps=3)
    assert not result.ok
    assert "grasp missed" in result.message
    down = np.asarray(CFG["architecture_a"]["skills"]["down_rotvec"], dtype=float)
    assert all(np.allclose(p[3:], down) for p in ctrl.poses), "it tilted anyway"


def test_open_drawer_does_not_pull_when_it_missed_the_handle():
    ctrl = FakeController(held_mm=0.0)
    result = Skills(ctrl, cfg=CFG).open_drawer(TARGET, pull_axis="-x")
    assert not result.ok
    assert "grasp missed" in result.message
    assert all(p[0] >= TARGET[0] - 1e-9 for p in ctrl.poses[2:]), "it pulled along -x"
