"""skills.py

It takes a base-frame position and gives you the arm actually doing something
there: picking, placing, pouring, or opening a drawer.

These are the only things Architecture A can do. The LLM planner does not
generate motion, it selects from this list and fills in the arguments, which is
exactly what makes the architecture inspectable: every motion the robot performs
traces back to a function you can read.

Two rules hold everywhere in this file.

  Only the shared controller API. move_to_pose, move_joints, gripper_toggle,
  get_state, and nothing else. If a skill reaches past the controller for a
  ur_rtde handle, Architecture B cannot be given the same treatment and the
  comparison starts measuring plumbing instead of architectures.

  Every pose goes through the safety envelope before it is commanded. No
  exceptions, including intermediate waypoints, because an approach pose that
  clips the table breaks the arm just as thoroughly as a grasp pose that does.

Poses are UR format: [x, y, z, rx, ry, rz], metres and rotation vector, base
frame. Orientation stays a rotation vector throughout rather than being
converted to Euler and back, because every conversion is a chance to introduce a
convention bug and nothing here needs Euler.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope
from fyp.shared.hardware.ur5e_controller import GripperError
from fyp.shared.helpers.config import get_config

GRIPPER_CLOSED = 0
GRIPPER_OPEN = 1


class SkillError(Exception):
    """Raised when a skill cannot complete. Carries the reason, not just a failure."""


class GraspMissed(SkillError):
    """
    Raised when the fingers closed on nothing. Its own type because the skill
    answers it differently from a gripper that did not respond: it backs off so
    the object can be seen again, instead of stopping where it is.
    """


@dataclass
class SkillResult:
    """What a skill did, in enough detail to appear in a trace and a report."""
    skill: str
    ok: bool
    message: str
    waypoints: list[list[float]] = field(default_factory=list)


class Skills:
    """
    It takes a controller and gives you the motion primitives the planner can call.

    Holds the controller and the tuning parameters so a skill call site stays
    short enough for an LLM to fill in correctly. Every parameter that matters
    is exposed as an argument with a configured default, because the planner
    will want to vary approach height for a tall bottle versus a flat block.
    """

    def __init__(self, controller, cfg: dict | None = None,
                 envelope: WorkspaceEnvelope | None = None):
        cfg = cfg or get_config()
        s = cfg["architecture_a"]["skills"]
        self.ctrl = controller
        self.env = envelope or WorkspaceEnvelope(cfg)
        self.approach_height = float(s["approach_height"])
        self.retreat_height = float(s["retreat_height"])
        self.grasp_depth_offset = float(s["grasp_depth_offset"])
        self.speed = float(s["speed"])
        self.acc = float(s["acc"])
        self.down_rotvec = [float(v) for v in s["down_rotvec"]]
        self.min_grasp_width_mm = float(s["min_grasp_width_mm"])
        # From the same cfg as everything else, not a fresh get_config(), so a
        # test that hands in its own config really controls where home is.
        home_q = cfg.get("robot", {}).get("home_q")
        self.home_q = None if home_q is None else [float(v) for v in home_q]

    # ---------------------------------------------------------------- helpers

    def _pose(self, xyz, rotvec=None) -> list[float]:
        """It takes a position and gives you a full checked pose with a tool orientation."""
        p = np.asarray(xyz, dtype=float).reshape(3)
        r = self.down_rotvec if rotvec is None else [float(v) for v in np.asarray(rotvec).reshape(3)]
        return self.env.check_pose([*p, *r])

    def _move(self, pose, result: SkillResult, speed=None, acc=None) -> None:
        """
        It takes a checked pose and gives you the arm there, or raises.

        `moveL` returns False when the controller rejects the target, which is
        easy to ignore and then spend an hour wondering why the arm skipped a
        waypoint. Treat a falsy return as a hard failure.
        """
        s, a = self.env.clamp_motion(speed if speed is not None else self.speed,
                                     acc if acc is not None else self.acc)
        ok = self.ctrl.move_to_pose(pose, s, a)
        result.waypoints.append(list(pose))
        if ok is False:
            raise SkillError(f"controller refused pose {['%.3f' % v for v in pose]}")

    def _gripper(self, state: int):
        """
        It takes an open or closed state and gives you the gripper there, with the
        final finger width in mm, or raises SkillError.

        No sleep afterwards. `gripper_toggle` returns only after the fingers have
        stopped, so there is nothing left to wait for.

        `GripperError` becomes `SkillError` here, in one place, so every skill
        reports a gripper that did not respond as `ok=False` with the reason,
        like any other failure. Left as it was, it would escape the skill and stop
        the whole run with the arm wherever it happened to be.
        """
        try:
            return self.ctrl.gripper_toggle(state)
        except GripperError as e:
            raise SkillError(f"gripper did not respond: {e}") from e

    def _grasp(self) -> float:
        """
        It takes nothing and gives you the fingers closed on something, with the
        width they stopped at, or raises GraspMissed when they closed on nothing.

        A close that misses is not a gripper error. The fingers did exactly what
        they were told and travelled all the way in, so `gripper_toggle` returns
        normally and only the width tells a miss from a hold. Without this check a
        missed pick lifts an empty gripper, reports success, and the planner goes
        on to place nothing.

        The width is the one `gripper_toggle` returns, not the RG2 grip-detected
        input, which is unreliable on this cell.
        """
        width = self._gripper(GRIPPER_CLOSED)
        try:
            width = float(width)
        except (TypeError, ValueError):
            # No number means no evidence either way, and "probably holding it" is
            # the guess that drops things from a height.
            raise SkillError(f"gripper returned no width ({width!r}), "
                             "so the grasp cannot be confirmed") from None
        if width < self.min_grasp_width_mm:
            raise GraspMissed(f"grasp missed: fingers closed to {width:.1f} mm, below "
                             f"{self.min_grasp_width_mm:.1f} mm, so nothing is held")
        return width

    def _back_off(self, xyz, result: SkillResult, speed=None) -> str:
        """
        It takes the target of a missed grasp and gives you the fingers open and
        the arm above it, plus a note when that did not work.

        Backing off before reporting matters because the pipeline re-perceives the
        scene straight after a failure. Left at the grasp pose, the gripper sits
        between the fixed camera and the object it just missed, the object is not
        seen, and the planner re-plans against a scene with the target missing.

        Never raises. The miss is the failure being reported, and a back-off that
        also failed is added to that message rather than replacing it.
        """
        try:
            self._gripper(GRIPPER_OPEN)
            self._move(self._above(xyz, self.retreat_height), result, speed=speed)
        except (SkillError, SafetyViolation) as e:
            return f"; back-off failed, arm left at the grasp pose: {e}"
        return ""

    def _above(self, xyz, height: float) -> list[float]:
        """It takes a target and gives you the pose directly above it."""
        p = np.asarray(xyz, dtype=float).reshape(3).copy()
        p[2] += height
        return self._pose(p)

    # ----------------------------------------------------------------- skills

    def pick(self, xyz, approach_height: float | None = None,
             grasp_offset: float | None = None, speed: float | None = None) -> SkillResult:
        """
        It takes a base-frame position and gives you the object at it held in the
        gripper.

        Approach from directly above rather than diagonally. A diagonal approach
        sweeps the gripper through the space beside the target, which is where
        the other objects are.
        """
        r = SkillResult("pick", True, "")
        h = self.approach_height if approach_height is None else float(approach_height)
        off = self.grasp_depth_offset if grasp_offset is None else float(grasp_offset)

        grasp = np.asarray(xyz, dtype=float).reshape(3).copy()
        grasp[2] -= off

        try:
            self._gripper(GRIPPER_OPEN)
            self._move(self._above(xyz, h), r, speed=speed)
            self._move(self._pose(grasp), r, speed=speed)
            try:
                width = self._grasp()
            except GraspMissed as e:
                r.ok, r.message = False, str(e) + self._back_off(xyz, r, speed)
                return r
            self._move(self._above(xyz, self.retreat_height), r, speed=speed)
        except (SkillError, SafetyViolation) as e:
            r.ok, r.message = False, str(e)
            return r

        r.message = f"picked at {np.round(grasp, 3).tolist()}, holding {width:.0f} mm"
        return r

    def place(self, xyz, approach_height: float | None = None,
              release_clearance: float = 0.01, speed: float | None = None) -> SkillResult:
        """
        It takes a base-frame position and gives you whatever is held released there.

        `release_clearance` stops the gripper pressing the object into the
        surface before opening, which either jams the fingers or flips the object.
        """
        r = SkillResult("place", True, "")
        h = self.approach_height if approach_height is None else float(approach_height)

        drop = np.asarray(xyz, dtype=float).reshape(3).copy()
        drop[2] += float(release_clearance)

        try:
            self._move(self._above(xyz, h), r, speed=speed)
            self._move(self._pose(drop), r, speed=speed)
            self._gripper(GRIPPER_OPEN)
            self._move(self._above(xyz, self.retreat_height), r, speed=speed)
        except (SkillError, SafetyViolation) as e:
            r.ok, r.message = False, str(e)
            return r

        r.message = f"placed at {np.round(drop, 3).tolist()}"
        return r

    def pour(self, source_xyz, target_xyz, tilt_rad: float = 1.9,
             pour_height: float = 0.15, hold_s: float = 3.0,
             steps: int = 12) -> SkillResult:
        """
        It takes a container position and a receptacle position, and gives you the
        contents of one transferred into the other.

        The tilt is interpolated over `steps` rather than commanded in one move,
        for two reasons. A single large reorientation makes the wrist take an
        unpredictable path, and liquid responds to the rate of tilt, not the
        final angle, so a slew you cannot control is a spill you cannot prevent.

        Tilt is applied about the tool x-axis by composing rotation vectors
        through their rotation matrices. Adding rotation vectors component-wise
        looks like it works and does not.

        `pick` and `place` are called for their sub-motions, and their results are
        CHECKED rather than discarded. A skill reports failure by returning
        `ok=False` instead of raising, so ignoring the return value of the grasp
        lets everything below it run on an empty gripper: the arm tilts nothing
        over the glass, returns nothing to the table, and reports a pour that
        happened. Their waypoints are absorbed as well, so the trace still lists
        every pose the arm was sent to.
        """
        from fyp.shared.helpers.rotations import R_to_rotvec, rotvec_to_R

        r = SkillResult("pour", True, "")
        try:
            grasp = self.pick(source_xyz)
            r.waypoints += grasp.waypoints
            if not grasp.ok:
                raise SkillError(f"could not grasp the container: {grasp.message}")

            above = self._above(target_xyz, float(pour_height))
            self._move(above, r)

            R0 = rotvec_to_R(np.asarray(self.down_rotvec, dtype=float))
            for i in range(1, int(steps) + 1):
                a = float(tilt_rad) * i / steps
                R_tilt = np.array([[1, 0, 0],
                                   [0, np.cos(a), -np.sin(a)],
                                   [0, np.sin(a), np.cos(a)]], dtype=float)
                rv = R_to_rotvec(R0 @ R_tilt)
                self._move(self._pose(above[:3], rv), r, speed=self.speed * 0.5)

            time.sleep(float(hold_s))

            for i in range(int(steps) - 1, -1, -1):
                a = float(tilt_rad) * i / steps
                R_tilt = np.array([[1, 0, 0],
                                   [0, np.cos(a), -np.sin(a)],
                                   [0, np.sin(a), np.cos(a)]], dtype=float)
                rv = R_to_rotvec(R0 @ R_tilt)
                self._move(self._pose(above[:3], rv), r, speed=self.speed * 0.5)

            back = self.place(source_xyz)
            r.waypoints += back.waypoints
            if not back.ok:
                raise SkillError(f"could not return the container: {back.message}")
        except (SkillError, SafetyViolation) as e:
            r.ok, r.message = False, str(e)
            return r

        r.message = f"poured into {np.round(np.asarray(target_xyz, float), 3).tolist()}"
        return r

    def open_drawer(self, handle_xyz, pull_distance: float = 0.20,
                    pull_axis: str = "-x", speed: float | None = None) -> SkillResult:
        """
        It takes a handle position and gives you the drawer pulled open.

        `pull_axis` is the base-frame direction the drawer travels, which depends
        entirely on how the unit is oriented on the bench and must be measured,
        not guessed. Pulling the wrong way drags the whole drawer unit across the
        table or stalls the arm against a closed drawer.
        """
        r = SkillResult("open_drawer", True, "")
        axes = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0)}
        if pull_axis not in axes:
            r.ok, r.message = False, f"pull_axis must be one of {sorted(axes)}, got {pull_axis!r}"
            return r
        idx, sign = axes[pull_axis]

        approach = np.asarray(handle_xyz, dtype=float).reshape(3).copy()
        approach[idx] -= sign * 0.10          # stand off along the pull axis
        pulled = np.asarray(handle_xyz, dtype=float).reshape(3).copy()
        pulled[idx] += sign * float(pull_distance)

        try:
            self._gripper(GRIPPER_OPEN)
            self._move(self._pose(approach), r, speed=speed)
            self._move(self._pose(handle_xyz), r, speed=speed)
            try:
                self._grasp()
            except GraspMissed as e:
                # Pulling with empty fingers sweeps the gripper through the space in
                # front of the drawer and reports a drawer that never moved.
                r.ok, r.message = False, str(e) + self._back_off(handle_xyz, r, speed)
                return r
            self._move(self._pose(pulled), r, speed=speed)
            self._gripper(GRIPPER_OPEN)
            self._move(self._above(pulled, self.retreat_height), r, speed=speed)
        except (SkillError, SafetyViolation) as e:
            r.ok, r.message = False, str(e)
            return r

        r.message = f"opened {pull_distance:.3f} m along {pull_axis}"
        return r

    def home(self) -> SkillResult:
        """
        It takes nothing and gives you the arm back at a known joint configuration.

        Joint space, not Cartesian. A Cartesian move to "home" from an arbitrary
        pose can route through a singularity; a joint move cannot.

        The joint move is checked first, with the same limit `go_home.py` uses. A
        joint move with no size check can turn wrist 3 a full circle when the
        wrist sits at the other joint value of the same physical angle.
        """
        r = SkillResult("home", True, "")
        if self.home_q is None:
            r.ok, r.message = False, "no robot.home_q in config; measure it and add it"
            return r
        try:
            now = self.ctrl.get_state()["joint_pos"]
            self.env.check_joint_move(now, self.home_q, "home")
        except SafetyViolation as e:
            r.ok, r.message = False, str(e)
            return r
        ok = self.ctrl.move_joints(self.home_q, self.speed, self.acc)
        if ok is False:
            r.ok, r.message = False, "controller refused the home joint target"
            return r
        r.message = "at home"
        return r
