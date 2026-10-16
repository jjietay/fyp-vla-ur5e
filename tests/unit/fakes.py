"""Fakes that more than one unit test module uses.

A fake goes here only when a second test module needs it. One copy per fake
means one place to update when the real interface changes, and
`test_fake_controller_matches_the_real_one` in test_skills.py keeps
FakeController honest against URController for every user.

Imported as `from fakes import ...`: tests/unit has no __init__.py, so pytest
puts this directory on sys.path.
"""
from __future__ import annotations

from fyp.architecture_a.skills import GRIPPER_OPEN
from fyp.shared.hardware.ur5e_controller import GripperError


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
