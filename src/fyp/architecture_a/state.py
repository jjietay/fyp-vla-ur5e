"""state.py

It takes each look at the table and gives you what the planner is shown as
"State now": the object in the gripper, the objects in view with IDs that stay
the same from look to look, and the objects no longer in view.

Code keeps this, not the model. The model only reads it.

## IDs are matched by position, never by query

A new detection takes the ID of the nearest known object within `tolerance`
(`verify.tolerance_mm`). The query only breaks a tie. Tier 2 is the reason: after
the user says "orange juice", the planner looks again with "orange juice bottle",
and the bottle that answers is the same `bottle_2` it saw as "bottle". A match by
query would give it a new ID and lose that link.

## Nothing out of view carries a position

The held object is shown by ID only: the gripper hides it from the fixed camera,
and its place is the gripper. An object no longer in view is shown by ID and the
turn it was last seen, with no xyz. A stale position is something the planner
can act on, and in Tier 3 a person moves things during the task. `visible_xyz`
is what `validate_call` checks a pick against, so a skill aimed at something out
of view is refused until a new look finds it. The positions are still kept here,
unshown, so an object that comes back to the same place gets its old ID back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tracked:
    """One object A knows about, in view or not."""
    id: str
    name: str
    confidence: float
    xyz: list[float]
    last_seen: int
    visible: bool = True
    # True when the last look did not search for this object's name, so "not
    # seen" would claim more than the camera showed.
    looked_for: bool = True


class WorldState:
    """
    It takes the matching tolerance in metres and gives you the state of one run,
    updated after every look.

    One per run. IDs from an earlier instruction mean nothing in a new one.
    """

    def __init__(self, tolerance_m: float):
        self.tolerance = float(tolerance_m)
        self.objects: dict[str, Tracked] = {}
        self.held: str | None = None
        self.out_of_reach: list[dict] = []
        self._count: dict[str, int] = {}

    # -------------------------------------------------------------- looking

    def update(self, in_reach: list[dict], out_of_reach: list[dict], queries: list[str],
               turn: int, gripper=None) -> None:
        """
        It takes one look ({name, confidence, xyz} rows in reach, out-of-reach
        rows, the queries it searched for, the planner turn) and gives you the
        state brought up to date.

        `gripper` is the TCP position. While something is held, the detection
        nearest the TCP and within tolerance is the held object, whatever name
        found it. It stays "in gripper" and does not become a new object on the
        table, or a look with other words ("fruit" for the held "orange") would
        offer the planner the object it is already holding.
        """
        rows = [dict(r) for r in in_reach]
        if self.held is not None and gripper is not None:
            near = [r for r in rows if _dist(r["xyz"], gripper) <= self.tolerance]
            if near:
                rows.remove(min(near, key=lambda r: _dist(r["xyz"], gripper)))

        known = [t for t in self.objects.values() if t.id != self.held]
        # Closest pairs first, one to one, so two bottles side by side cannot both
        # claim the same old ID. Same name sorts first at equal distance.
        pairs = sorted(((_dist(r["xyz"], t.xyz), r["name"] != t.name, i, t.id)
                        for i, r in enumerate(rows) for t in known
                        if _dist(r["xyz"], t.xyz) <= self.tolerance))
        taken_rows: set[int] = set()
        taken_ids: set[str] = set()
        for _, _, i, tid in pairs:
            if i in taken_rows or tid in taken_ids:
                continue
            taken_rows.add(i)
            taken_ids.add(tid)
            self._see(self.objects[tid], rows[i], turn)

        for t in known:
            if t.id not in taken_ids:
                t.visible = False
                t.looked_for = t.name in queries
        for i, r in enumerate(rows):
            if i not in taken_rows:
                t = Tracked(id=self._new_id(r["name"]), name=r["name"],
                            confidence=r["confidence"], xyz=list(r["xyz"]), last_seen=turn)
                self.objects[t.id] = t
        self.out_of_reach = [dict(o) for o in out_of_reach]

    def nothing_seen(self, turn: int, queries: list[str]) -> None:
        """It takes a look that found nothing usable and gives you every table
        object marked out of view."""
        self.update([], [], queries, turn)

    @staticmethod
    def _see(t: Tracked, row: dict, turn: int) -> None:
        # The latest name wins: after "orange juice bottle" finds bottle_2, the
        # planner should read the words it just chose.
        t.name, t.confidence, t.xyz = row["name"], row["confidence"], list(row["xyz"])
        t.last_seen, t.visible, t.looked_for = turn, True, True

    def _new_id(self, name: str) -> str:
        base = "_".join(name.split()) or "object"
        self._count[base] = self._count.get(base, 0) + 1
        return f"{base}_{self._count[base]}"

    # ------------------------------------------------------------ the gripper

    def grab(self, object_id: str) -> None:
        """It takes the ID a pick took and gives you it marked as held."""
        self.held = object_id

    def release(self, at=None, turn: int = 0) -> None:
        """
        It takes where the camera saw the released object (or None) and gives you
        the gripper marked empty.

        Placed at the seen point, the object keeps its ID on the next look. Not
        seen, it is out of view, with no position to act on.
        """
        if self.held is None:
            return
        t = self.objects[self.held]
        self.held = None
        if at is not None:
            t.xyz, t.visible, t.last_seen = [float(v) for v in at], True, turn
        else:
            t.visible = False

    # --------------------------------------------------------------- reading

    def target(self, xyz) -> Tracked | None:
        """It takes a position and gives you the object in view nearest it within
        tolerance, or None."""
        near = [t for t in self.objects.values()
                if t.visible and t.id != self.held and _dist(t.xyz, xyz) <= self.tolerance]
        return min(near, key=lambda t: _dist(t.xyz, xyz), default=None)

    def visible_xyz(self) -> list[list[float]]:
        """It takes nothing and gives you the positions a pick may aim at."""
        return [t.xyz for t in self.objects.values() if t.visible and t.id != self.held]

    def render(self) -> dict:
        """It takes nothing and gives you the state as the planner reads it."""
        shown = [t for t in self.objects.values() if t.id != self.held]
        return {
            "in_gripper": self.held,
            "objects": [{"id": t.id, "name": t.name, "confidence": t.confidence,
                         "xyz": t.xyz} for t in shown if t.visible],
            "not_in_view": [{"id": t.id, "name": t.name, "last_seen_turn": t.last_seen,
                             "why": "not seen" if t.looked_for else "not looked for"}
                            for t in shown if not t.visible],
            "out_of_reach": self.out_of_reach,
        }


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
