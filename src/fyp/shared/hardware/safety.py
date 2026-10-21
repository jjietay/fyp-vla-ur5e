"""safety.py

It takes a pose the software wants to command and gives you either the same pose
back or an exception explaining why the arm must not go there.

This is the last thing between generated output and a physical machine, and it
is shared on purpose. Architecture A produces poses from an LLM plan and
Architecture B produces them from a neural network; neither source is trustworthy
enough to reach `moveL` unchecked. A single envelope also keeps the comparison
honest, because both architectures are constrained identically.

Nothing here talks to the robot. It is pure predicate logic over numbers so it
can be unit tested without hardware, which is the only way you find out the
bounds are wrong before rather than after.
"""
from __future__ import annotations

import numpy as np

from fyp.shared.helpers.config import get_config


class SafetyViolation(Exception):
    """Raised when a commanded pose or velocity is outside the configured envelope."""


class WorkspaceEnvelope:
    """
    It takes the workspace block from config and gives you a checker that accepts
    or rejects candidate poses.

    Horizontally the limit is a ring sector around the base axis: between
    `min_radius` and `max_radius`, and between the two bearings in `angle_deg`.
    Vertically it is the `z` range. The arm's reach is round and the table sits
    in an arc in front of it, so a sector follows both with three numbers a tape
    measure gives you; a box either left half its area off the table or cut the
    reach short (21 Oct 2026, after the arm was moved). The real UR5e enforces its
    own joint and singularity limits, so this only has to stop the obvious
    catastrophes: driving into the table, reaching off the bench, swinging into
    the camera, or commanding a speed nobody intended.

    The `x` and `y` box stays as a loose backstop around the sector, so a typo in
    one sector number is still caught. `min_radius` and `angle_deg` are optional;
    without them the limit is the box and `max_radius` alone, as before.

    Why `max_radius` exists at all: the horizontal distance from the base axis. A box has corners, and the arm's reach is round. On 23 Sep 2026 a
    target 0.81 m out was inside the box and still out of reach, because its
    approach and retreat poses could not be reached. The controller refuses such
    a pose itself, but only after the skill has started, so the arm stops half
    way through a pick. Rejecting it here stops the skill before it moves.

    The radius is horizontal, not spherical, because the table is close to the
    base height and a tape measure on the table gives exactly this number.
    """

    def __init__(self, cfg: dict | None = None):
        ws = (cfg or get_config())["workspace"]
        self.x = tuple(float(v) for v in ws["x"])
        self.y = tuple(float(v) for v in ws["y"])
        self.z = tuple(float(v) for v in ws["z"])
        self.max_radius = float(ws["max_radius"])
        self.min_radius = float(ws.get("min_radius", 0.0))
        # The bearing of the POINT, atan2(y, x), not the base joint angle. They
        # differ by a few degrees because the shoulder is offset from the base axis.
        self.angle_deg = tuple(float(v) for v in ws.get("angle_deg", (-180.0, 180.0)))
        self.max_joint_step_deg = float(ws["max_joint_step_deg"])
        self.max_speed = float(ws["max_speed"])
        self.max_acc = float(ws["max_acc"])

    def contains_xy(self, x, y):
        """
        It takes base-frame x and y, scalars or arrays, and gives you whether each
        is inside the horizontal limit: the box and the ring sector.

        Vectorised so a script filtering thousands of depth points uses exactly
        the same test as a commanded pose.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        r = np.hypot(x, y)
        a = np.degrees(np.arctan2(y, x))
        return ((self.x[0] <= x) & (x <= self.x[1])
                & (self.y[0] <= y) & (y <= self.y[1])
                & (self.min_radius <= r) & (r <= self.max_radius)
                & (self.angle_deg[0] <= a) & (a <= self.angle_deg[1]))

    def contains(self, xyz) -> bool:
        """It takes a base-frame point and gives you whether it is inside the envelope."""
        x, y, z = (float(v) for v in np.asarray(xyz, dtype=float).reshape(3))
        return bool(self.z[0] <= z <= self.z[1] and self.contains_xy(x, y))

    def check_point(self, xyz, what: str = "point") -> np.ndarray:
        """
        It takes a base-frame point and gives it back, or raises with the axis
        that failed and by how much.

        The message names the offending axis on purpose. "Out of workspace" tells
        you nothing at 2am; "z=-0.031 below min 0.005" tells you the calibration
        is off by three centimetres.
        """
        p = np.asarray(xyz, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            raise SafetyViolation(f"{what} is not finite: {p.tolist()}")
        for value, (lo, hi), axis in zip(p, (self.x, self.y, self.z), "xyz"):
            if value < lo:
                raise SafetyViolation(
                    f"{what} {axis}={value:.3f} is below min {lo:.3f} "
                    f"(by {lo - value:.3f} m)")
            if value > hi:
                raise SafetyViolation(
                    f"{what} {axis}={value:.3f} is above max {hi:.3f} "
                    f"(by {value - hi:.3f} m)")
        # After the box on purpose. A point that fails both is reported by axis,
        # which is the more specific message.
        radius = float(np.hypot(p[0], p[1]))
        if radius > self.max_radius:
            raise SafetyViolation(
                f"{what} radius={radius:.3f} is above max {self.max_radius:.3f} "
                f"(by {radius - self.max_radius:.3f} m)")
        if radius < self.min_radius:
            raise SafetyViolation(
                f"{what} radius={radius:.3f} is below min {self.min_radius:.3f} "
                f"(by {self.min_radius - radius:.3f} m)")
        angle = float(np.degrees(np.arctan2(p[1], p[0])))
        lo, hi = self.angle_deg
        if angle < lo:
            raise SafetyViolation(
                f"{what} angle={angle:.1f} deg is below min {lo:.1f} (by {lo - angle:.1f} deg)")
        if angle > hi:
            raise SafetyViolation(
                f"{what} angle={angle:.1f} deg is above max {hi:.1f} (by {angle - hi:.1f} deg)")
        return p

    def check_pose(self, pose, what: str = "pose") -> list[float]:
        """
        It takes a 6-element UR pose and gives it back as a plain list, or raises.

        Only the translation is bounded. Orientation is left free because a
        rotation vector has no meaningful box, and the controller rejects
        unreachable orientations itself.
        """
        p = np.asarray(pose, dtype=float).reshape(-1)
        if p.size != 6:
            raise SafetyViolation(f"{what} must have 6 elements, got {p.size}")
        if not np.all(np.isfinite(p)):
            raise SafetyViolation(f"{what} is not finite: {p.tolist()}")
        self.check_point(p[:3], what)
        return [float(v) for v in p]

    def check_joint_move(self, current_q, target_q, what: str = "joint move") -> list[float]:
        """
        It takes the joint angles now and a joint target, and gives the target back
        as a plain list, or raises naming the joint that would turn too far.

        A joint move takes every joint the direct way from its current angle to
        its target angle, and UR joints turn +/-360 degrees. So one physical
        wrist angle has two joint values, for example 253 and -107 degrees, and a
        target written with the other value turns the wrist a full circle, with
        the gripper and its cable on it. The box and radius checks cannot see
        this, because the TCP ends in the right place. Refusing any joint that
        must turn more than `max_joint_step_deg` stops it.

        Refuses rather than wraps the target to the nearer value. Wrapping would
        silently change which way the cable is wound, and a person should see
        that and jog the arm instead.
        """
        now = np.asarray(current_q, dtype=float).reshape(-1)
        target = np.asarray(target_q, dtype=float).reshape(-1)
        if now.size != 6 or target.size != 6:
            raise SafetyViolation(f"{what} needs 6 joint angles, got {now.size} and {target.size}")
        if not (np.all(np.isfinite(now)) and np.all(np.isfinite(target))):
            raise SafetyViolation(f"{what} has a non-finite joint angle")
        change = np.degrees(target - now)
        worst = int(np.argmax(np.abs(change)))
        if abs(change[worst]) > self.max_joint_step_deg:
            raise SafetyViolation(
                f"{what} would turn joint {worst} by {change[worst]:+.1f} deg, over the "
                f"{self.max_joint_step_deg:.0f} deg limit. Jog the arm nearer on the pendant")
        return [float(v) for v in target]

    def clamp_motion(self, speed: float | None, acc: float | None) -> tuple[float, float]:
        """
        It takes a requested speed and acceleration and gives you values that are
        certainly within the configured ceiling.

        Clamps rather than raises. A plan asking to move too fast is a tuning
        mistake, not an emergency, and silently slowing down is the safe failure.
        Going out of bounds in space is different, and that does raise.
        """
        s = self.max_speed if speed is None else min(float(speed), self.max_speed)
        a = self.max_acc if acc is None else min(float(acc), self.max_acc)
        if s <= 0 or a <= 0:
            raise SafetyViolation(f"speed and acc must be positive, got {s}, {a}")
        return s, a

    def describe(self) -> str:
        """It takes nothing and gives you the envelope as one line, for prompts and logs."""
        return (f"x[{self.x[0]:.3f},{self.x[1]:.3f}] "
                f"y[{self.y[0]:.3f},{self.y[1]:.3f}] "
                f"z[{self.z[0]:.3f},{self.z[1]:.3f}] "
                f"radius[{self.min_radius:.3f},{self.max_radius:.3f}] "
                f"angle[{self.angle_deg[0]:.0f},{self.angle_deg[1]:.0f}] deg, metres, base frame")
