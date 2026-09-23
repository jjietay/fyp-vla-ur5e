"""ur5e_controller.py

Real UR5e backend, driven over ur_rtde.

Public API:

- __init__ connects via ur_rtde
- move_to_pose(pose, speed, acc) using moveL, BLOCKING
- move_joints(q, speed, acc) using moveJ, BLOCKING
- servo_to_pose(pose, ...) using servoL, NON-BLOCKING, for a policy loop
- servo_stop(acc) ends a servo stream
- period_start() / wait_period(t) hold an exact control cadence
- get_state() returns a dict: joint_pos, tcp_pose, gripper_state
  (gripper_state is None until gripper_start() has run)
- gripper_start() checks the RG2 answers and reads where its fingers are
- gripper_toggle(state = 0 or 1, where 0 is closed and 1 is open), BLOCKING,
  returns the final finger width in mm
- gripper_move(width_mm) any width, returns (status, width) without judging it,
  for bench tools
- close()

The gripper is OPT-IN. Connecting touches no output. Only gripper_start() and
gripper_toggle() send anything to the tool connector, so a motion-only run
never moves the fingers.

## How the OnRobot RG2 is driven

The RG2 sits on the wrist tool connector with no Compute Box, so there is no
network interface to it. The RG URCap 1.10.1 drives it by shifting a 17 bit word
out over the two tool digital outputs, and reads back a ready signal on a tool
input and the finger width on a tool analog input. `docs/rg2.script` is the
pendant program this is copied from, and `scripts/check_gripper.py` is the bench
test that confirmed it on 23 Sep 2026.

Each half bit is one `sync()`, 2 ms. RTDE cannot toggle a pin that fast with
reliable timing from Python, so the whole exchange runs ON THE CONTROLLER as
URScript sent with `sendCustomScriptFunction`. That call stops the ur_rtde
control script, runs ours, and uploads the control script again, so every
gripper call costs about 0.2 s before the fingers move and returns only after
they stop. A full stroke is about 1.5 s. Any open servo stream ends too.

The check happens on the controller, not in Python, for two reasons found on
the bench. First, 1 command in about 66 was lost: the fingers did not move and
the ready signal still read 1, so Python saw nothing wrong. Second, ur_rtde has
no getter for the tool analog input, so Python cannot read the finger width.
The URScript therefore reads the width itself, watches for the busy signal after
sending, sends once more if the fingers never started, and hands the result
back through RTDE output registers:

    output_double_register[r]      final finger width, mm
    output_int_register[r + 1]     GripperStatus
    output_int_register[r]         call id, written LAST

Python waits for its own call id to appear before it reads the other two, so a
result left over from an earlier call is never mistaken for this one. `r` is
`robot.gripper.result_register`.

The grip detected input is NOT used. On the bench it read 1 right after closing
on an object and 0 a moment later with the object still held.
"""

import math
import rtde_control
import rtde_receive
import rtde_io
import time
from enum import IntEnum
from pathlib import Path

from fyp.shared.helpers.config import get_config, DEFAULT_CONFIG_PATH


class GripperError(RuntimeError):
    """Raised when the RG2 did not do what it was told, with the reason and the width."""


class GripperStatus(IntEnum):
    """
    What the URScript reports back in the status register.

    The numbers are written into the URScript as literals, so they are part of
    the wire format between this file and the controller. Do not renumber them.
    """
    NO_RESPONSE = 0   # sent max_sends times and the fingers never started
    MOVED = 1         # busy appeared and ready came back
    AT_TARGET = 2     # nothing sent, or motion finished late, width within tolerance
    TIMEOUT = 3       # busy appeared and ready did not come back within 3 s
    NOT_READY = 4     # the gripper never reported ready, so nothing was sent
    READ = 5          # read only, nothing was sent


def rg2_word(width_mm: float, force_n: float) -> int:
    """
    It takes a target finger width in mm and a grip force in N and gives you the
    17 bit word the RG2 expects, with the URCap's own clamping applied.

    The offsets and limits are the RG2 protocol as the URCap `RG2()` function
    applies them (docs/rg2.script), not tuning, so they are written here and not
    in config.yaml. Changing one would send a different word, not a better one.
    """
    w = min(max(float(width_mm) + 9.2, 0.0), 110.0)
    f = min(max(float(force_n), 3.0), 40.0)
    return math.floor(w) * 4 + math.floor(f / 2) * 444 + 32768


def rg2_target_mm(width_mm: float) -> float:
    """
    It takes a requested width and gives you the width the RG2 will really go
    to, which is what the arrival check must compare against.

    The word carries whole millimetres of (width + 9.2), so a request for 50 mm
    is really 49.8 mm and a request for 0 mm is really -0.2 mm.
    """
    return math.floor(min(max(float(width_mm) + 9.2, 0.0), 110.0)) - 9.2


def rg2_urscript(word: int, target_mm: float, tolerance_mm: float, max_sends: int,
                 call_id: int, register: int, read_only: bool = False) -> str:
    """
    It takes the word to send and how to judge arrival, and gives you the
    URScript body that sends it, checks it, and reports the result.

    Built as text here so the text can be unit tested. Nothing can run URScript
    away from the arm, so the tests check its structure and its numbers instead.

    `read_only` sends nothing and only reports the current width. It is how
    `gripper_start` learns where the fingers are.

    Timings are in `sync()` cycles of 2 ms. 1500 is 3 s, longer than a full
    stroke. 20 is the URCap's own wait for the busy signal to appear after a
    word is sent. The width line is the URCap's formula for tool analog input 0
    (`get_analog_in(2)`), copied as it is.
    """
    sends = 0 if read_only else int(max_sends)
    status = GripperStatus.READ if read_only else GripperStatus.NO_RESPONSE
    width = ("rg_zscale = (get_analog_in(2) - 0.026) / 2.5900013\n"
             "{i}rg_width = floor((5.0 + 110 * sin(rg_zscale * 1.57079633 - 0.08726646))"
             " * 10) / 10.0 - 9.2")
    return f"""
  set_tool_voltage(24)
  set_tool_output_mode(0)
  set_tool_digital_output_mode(0, 1)
  set_tool_digital_output_mode(1, 1)
  set_tool_analog_input_domain(0, 1)
  rg_target = {float(target_mm):.3f}
  rg_status = {int(status)}
  rg_sends = 0
  rg_done = False
  while rg_sends < {sends} and rg_done == False:
    rg_t = 0
    while get_digital_in(9) == False and rg_t < 1500:
      rg_t = rg_t + 1
      sync()
    end
    {width.format(i="    ")}
    if get_digital_in(9) == False:
      rg_status = {int(GripperStatus.NOT_READY)}
      rg_done = True
    elif norm(rg_width - rg_target) <= {float(tolerance_mm):.3f}:
      rg_status = {int(GripperStatus.AT_TARGET)}
      rg_done = True
    else:
      rg_data = {int(word)}
      rg_msb = 65536
      rg_i = 0
      while rg_i < 17:
        set_digital_out(8, True)
        if rg_data >= rg_msb:
          rg_data = rg_data - rg_msb
          set_digital_out(9, False)
        else:
          set_digital_out(9, True)
        end
        sync()
        set_digital_out(8, False)
        sync()
        rg_data = rg_data * 2
        rg_i = rg_i + 1
      end
      rg_sends = rg_sends + 1
      rg_t = 0
      while get_digital_in(9) == True and rg_t < 20:
        rg_t = rg_t + 1
        sync()
      end
      if get_digital_in(9) == False:
        rg_t = 0
        while get_digital_in(9) == False and rg_t < 1500:
          rg_t = rg_t + 1
          sync()
        end
        if get_digital_in(9) == True:
          rg_status = {int(GripperStatus.MOVED)}
        else:
          rg_status = {int(GripperStatus.TIMEOUT)}
        end
        rg_done = True
      end
    end
  end
  if rg_done == False and rg_status == {int(GripperStatus.NO_RESPONSE)}:
    rg_t = 0
    while get_digital_in(9) == False and rg_t < 1500:
      rg_t = rg_t + 1
      sync()
    end
  end
  {width.format(i="  ")}
  if rg_done == False and rg_status == {int(GripperStatus.NO_RESPONSE)}:
    if norm(rg_width - rg_target) <= {float(tolerance_mm):.3f}:
      rg_status = {int(GripperStatus.AT_TARGET)}
    end
  end
  write_output_float_register({int(register)}, rg_width)
  write_output_integer_register({int(register) + 1}, rg_status)
  write_output_integer_register({int(register)}, {int(call_id)})
"""


class URController:
    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH, ):
        self.cfg = get_config(path)
        robot_ip = self.cfg["robot"]["host"]
        self.default_speed = self.cfg["robot"]["motion"]["default_speed"]
        self.default_acc = self.cfg["robot"]["motion"]["default_acc"]
        servo = self.cfg["robot"]["servo"]
        self.servo_time = float(servo["time"])
        self.servo_lookahead = float(servo["lookahead_time"])
        self.servo_gain = float(servo["gain"])
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_io = rtde_io.RTDEIOInterface(robot_ip)
        self._reset_gripper()

    # -------------------------------------------------------------- gripper

    def _reset_gripper(self):
        """
        It takes nothing and gives you a controller whose gripper is not started.

        Separate from `__init__` so the unit tests can build a controller on fake
        RTDE interfaces without connecting to anything.
        """
        g = self.cfg["robot"]["gripper"]
        self._grip_open_mm = float(g["open_mm"])
        self._grip_close_mm = float(g["close_mm"])
        self._grip_force_n = float(g["force_n"])
        self._grip_tolerance_mm = float(g["tolerance_mm"])
        self._grip_max_sends = int(g["max_sends"])
        self._grip_register = int(g["result_register"])
        self._grip_result_timeout_s = float(g["result_timeout_s"])
        self._grip_call_id = None
        self._gripper_state = None
        self.gripper_width_mm = None

    def _gripper_ready(self) -> bool:
        """It takes nothing and gives you the RG2 ready signal, tool input 1 (RTDE bit 17)."""
        return bool((int(self.rtde_r.getActualDigitalInputBits()) >> 17) & 1)

    def _rg2_run(self, width_mm: float, read_only: bool = False,
                 force_n: float | None = None):
        """
        It takes a target width and gives you (GripperStatus, final width in mm)
        once the fingers have stopped, as the controller measured them.

        Raises GripperError only when no result arrives at all. Judging the
        result is the caller's job, because what counts as success differs
        between opening and closing.
        """
        self._grip_call_id += 1
        call_id = self._grip_call_id
        reg = self._grip_register
        script = rg2_urscript(
            word=rg2_word(width_mm, self._grip_force_n if force_n is None else force_n),
            target_mm=rg2_target_mm(width_mm),
            tolerance_mm=self._grip_tolerance_mm,
            max_sends=self._grip_max_sends,
            call_id=call_id, register=reg, read_only=read_only)

        if not self.rtde_c.sendCustomScriptFunction("fyp_rg2", script):
            raise GripperError("the controller did not run the gripper script. "
                               "Is the pendant in Remote Control mode?")

        # The script has ended when the call returns, but RTDE samples at 500 Hz,
        # so the registers can lag by a cycle or two. Wait for this call's id
        # rather than sleeping a fixed time.
        deadline = time.monotonic() + self._grip_result_timeout_s
        while int(self.rtde_r.getOutputIntRegister(reg)) != call_id:
            if time.monotonic() > deadline:
                raise GripperError(
                    f"no result from the gripper script in output_int_register {reg} "
                    f"after {self._grip_result_timeout_s} s")
            time.sleep(0.002)

        status = GripperStatus(int(self.rtde_r.getOutputIntRegister(reg + 1)))
        width = float(self.rtde_r.getOutputDoubleRegister(reg))
        self.gripper_width_mm = width
        return status, width

    def gripper_start(self):
        """
        It takes nothing and gives you a started gripper, with `gripper_state`
        read from where the fingers really are.

        Moves nothing. It checks the ready signal, then runs the script in read
        only mode to learn the width. Fully open counts as open (1) and anything
        else counts as closed (0). That is the careful reading for Architecture B:
        a policy that wants the gripper open will then send an open, instead of
        the adapter assuming it is open and never sending one.
        """
        if not self._gripper_ready():
            raise GripperError("the RG2 does not report ready on tool input 1. "
                               "Check the tool cable and that the RG2 LED is on.")
        # Start counting from whatever id is in the register now. It survives
        # between sessions, so starting at 1 could match an old result at once.
        self._grip_call_id = int(self.rtde_r.getOutputIntRegister(self._grip_register))
        _, width = self._rg2_run(self._grip_open_mm, read_only=True)
        near_open = abs(width - rg2_target_mm(self._grip_open_mm)) <= self._grip_tolerance_mm
        self._gripper_state = 1 if near_open else 0
        return "Gripper initialized."

    def move_to_pose(self, pose, speed : float | None = None, acc : float | None = None):
        speed = speed if speed is not None else self.default_speed
        acc = acc if acc is not None else self.default_acc
        return self.rtde_c.moveL(pose, speed, acc)

    def move_joints(self, q, speed : float | None = None, acc : float | None = None):
        speed = speed if speed is not None else self.default_speed
        acc = acc if acc is not None else self.default_acc
        return self.rtde_c.moveJ(q, speed, acc)

    def servo_to_pose(self, pose, time_s: float | None = None,
                      lookahead_time: float | None = None, gain: float | None = None):
        """
        It takes a target TCP pose and gives you one servo step towards it,
        returning immediately instead of waiting for the motion to finish.

        This is the call a learned policy needs. `moveL` blocks until the arm
        arrives, so a 20 Hz policy loop built on it spends every cycle waiting
        and the arm stops between chunks. `servoL` hands the controller a target
        and returns, so the loop re-issues a fresh target every control period
        and the motion stays continuous.

        Meant to be called repeatedly at a fixed rate. One call on its own does
        almost nothing visible, and stopping the calls leaves the servo stream
        open, so pair it with `servo_stop`.

        `time_s` is the control period and must match the rate you actually call
        this at, or the arm tracks a target it thinks is further away in time
        than it is. `lookahead_time` in 0.03 to 0.2 smooths the path at the cost
        of lag. `gain` in 100 to 2000 decides how hard the controller chases the
        target, and too high oscillates.

        No workspace check happens here, matching `move_to_pose`. The caller
        checks, because only the caller knows whether the pose came from a plan
        or from a policy.
        """
        # speed and acceleration are accepted by servoL and ignored by current
        # UR firmware, so they are passed as zero rather than as the defaults,
        # which would read as if they did something.
        return self.rtde_c.servoL(
            list(pose),
            0.0,
            0.0,
            self.servo_time if time_s is None else float(time_s),
            self.servo_lookahead if lookahead_time is None else float(lookahead_time),
            self.servo_gain if gain is None else float(gain),
        )

    def servo_stop(self, acc: float | None = None):
        """
        It takes a deceleration and gives you a stopped arm, ending the servo stream.

        Not optional after `servo_to_pose`. The controller keeps driving towards
        the last target until it is told to stop, so an episode that just stops
        calling leaves the arm creeping.
        """
        acc = self.default_acc if acc is None else float(acc)
        return self.rtde_c.servoStop(acc)

    def period_start(self):
        """
        It takes nothing and gives you a timestamp marking the start of an RTDE
        control period.

        Pair with `wait_period`. Sleeping for a fixed interval instead lets the
        work inside the loop add to the period, so the real rate drifts below the
        intended one.

        CAREFUL, the period is the RTDE CONTROL period, not `servo.time`. ur_rtde
        paces these to the frequency the control interface was opened with, which
        is 500 Hz on an e-series by default, so this pair holds about 2 ms and NOT
        the 50 ms a 20 Hz policy runs at. Pacing a 20 Hz loop with them would run
        it 25 times too fast.

        Architecture B does not use these, because LeRobot's `robot_client` paces
        itself from its own `--fps`. They are here for a hand written servo loop,
        such as the smoke test in `tests/hardware/`.
        """
        return self.rtde_c.initPeriod()

    def wait_period(self, t_start):
        """
        It takes the timestamp from `period_start` and gives you the remainder of
        the RTDE control period, slept away. See `period_start` for which period
        that is, because it is not the servo period.
        """
        return self.rtde_c.waitPeriod(t_start)

    def gripper_move(self, width_mm: float, force_n: float | None = None):
        """
        It takes any finger width in mm, and optionally a force, and gives you
        (GripperStatus, final width in mm) once the fingers have stopped.

        For bench tools such as `scripts/check_gripper.py`, which need to report
        what happened rather than stop on it. It judges nothing and raises only
        when no result comes back at all. Skills and Architecture B use
        `gripper_toggle`, which decides what counts as a failure.
        """
        if self._grip_call_id is None:
            raise RuntimeError("gripper_move() called before gripper_start()")
        return self._rg2_run(width_mm, force_n=force_n)

    def gripper_toggle(self, state: int):
        """
        It takes 1 for open or 0 for closed and gives you the fingers there,
        returning the final width in mm once they have stopped.

        Raises GripperError when the gripper did not respond, or when an open
        did not reach the open width. A close that stops early is NOT an error,
        because stopping on the object is the whole point of a close. The
        returned width tells the caller whether anything is held: a close that
        ends near `close_mm` closed on nothing.
        """
        # Raise rather than return a message. A skill that closes a gripper that
        # was never started would otherwise carry on as if it had grasped.
        if self._grip_call_id is None:
            raise RuntimeError("gripper_toggle() called before gripper_start()")
        if isinstance(state, bool) or state not in (0, 1):
            return "Unable to control gripper."

        target = self._grip_open_mm if state == 1 else self._grip_close_mm
        status, width = self.gripper_move(target)

        if status not in (GripperStatus.MOVED, GripperStatus.AT_TARGET):
            raise GripperError(f"gripper {'open' if state == 1 else 'close'} failed: "
                               f"{status.name}, width {width:.1f} mm")
        if state == 1 and abs(width - rg2_target_mm(target)) > self._grip_tolerance_mm:
            raise GripperError(f"gripper open stopped at {width:.1f} mm, expected "
                               f"{rg2_target_mm(target):.1f} mm. Something is in the way.")

        self._gripper_state = state
        return width

    def get_state(self):
        return {
        "joint_pos": self.rtde_r.getActualQ(),
        "tcp_pose": self.rtde_r.getActualTCPPose(),
        # None, not a guess, when the gripper was never started. After that it is
        # the last state the controller CONFIRMED, since Python cannot read the
        # finger width between calls.
        "gripper_state": self._gripper_state,
        }

    def close(self):
        # The fingers are left where they are. Opening here would drop whatever
        # the arm is holding.
        self.rtde_c.stopScript()
