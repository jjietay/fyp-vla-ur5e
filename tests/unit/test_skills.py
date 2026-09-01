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
from fyp.shared.hardware.ur5e_controller import URController
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

    `settle_s` is a real 0.4 s in config.yaml, and this file actuates the gripper
    dozens of times. Patching the sleep rather than overriding the config keeps
    every number under test real and neutralises only the waiting, which is the
    one thing that has no bearing on whether the logic is correct.
    """
    monkeypatch.setattr("fyp.architecture_a.skills.time.sleep", lambda _s: None)


class FakeController:
    """
    It takes nothing and gives you a controller that records instead of moving.

    `refuse_after` reproduces the one controller behaviour that is easy to ignore:
    `moveL` returning False rather than raising. A skill that does not check it
    carries on to the next waypoint as though the arm had arrived.
    """

    def __init__(self, refuse_after: int | None = None):
        self.calls: list[tuple] = []
        self.refuse_after = refuse_after
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

    def get_state(self):
        self.calls.append(("get_state",))
        return {"joint_pos": [0.0] * 6, "tcp_pose": [0.0] * 6, "gripper_state": GRIPPER_OPEN}

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


def test_home_fails_cleanly_when_home_q_is_not_configured(skills, ctrl):
    """
    `robot.home_q` is not in config.yaml today, so `home` cannot work. It must say
    so rather than move the arm somewhere arbitrary.

    Note this reads the real config rather than the injected one, because `home`
    calls `get_config()` directly. That inconsistency is deliberate to record: if
    `home_q` is ever added, this test starts failing and the discrepancy surfaces.
    """
    result = skills.home()
    assert not result.ok
    assert "home_q" in result.message
    assert ctrl.calls == []


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
