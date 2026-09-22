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
- gripper_start(default pin_power = 1, default pin_control = 2)
- gripper_toggle(state = 0 or 1, where 0 is closed and 1 is open)
- close()

The gripper is OPT-IN. Connecting touches no digital output. Only gripper_start()
writes to the pins, so a motion-only run cannot switch a control box output that
may be wired to someone else's equipment in the shared lab. The pin code itself
does not fit the lab's OnRobot gripper yet, which sits on the tool connector.
"""

import rtde_control
import rtde_receive
import rtde_io
import time
from pathlib import Path

from fyp.shared.helpers.config import get_config, DEFAULT_CONFIG_PATH


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
        self.pin_power = None
        self.pin_control = None

    def gripper_start(self, pin_power : int | None = None, pin_control : int | None = None):

        grip = self.cfg["robot"]["gripper"]
        self.pin_power = pin_power if pin_power is not None else grip["power_pin"]
        self.pin_control = pin_control if pin_control is not None else grip["control_pin"]
        self.rtde_io.setStandardDigitalOut(self.pin_power, True)
        time.sleep(0.1)
        if self.rtde_r.getDigitalOutState(self.pin_power):
            return "Gripper initialized."
        else:
            print("Gripper unable to turn on.")

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

    def gripper_toggle(self, state: int):
        # Raise rather than return a message. A skill that closes a gripper that
        # was never started would otherwise carry on as if it had grasped.
        if self.pin_control is None:
            raise RuntimeError("gripper_toggle() called before gripper_start()")
        if state == 1:
            self.rtde_io.setStandardDigitalOut(self.pin_control, False)
            time.sleep(0.1)
        elif state == 0:
            self.rtde_io.setStandardDigitalOut(self.pin_control, True)
            time.sleep(0.1)
        else:
            return "Unable to control gripper."

    def get_state(self):
        return {
        "joint_pos": self.rtde_r.getActualQ(),
        "tcp_pose": self.rtde_r.getActualTCPPose(),
        # None, not a guess, when the gripper was never started. Passing None to
        # getDigitalOutState raises a TypeError, which broke every motion-only caller.
        "gripper_state": None if self.pin_control is None
                         else (0 if self.rtde_r.getDigitalOutState(self.pin_control) else 1)
        }

    def close(self):
        if self.pin_power is not None:
            self.rtde_io.setStandardDigitalOut(self.pin_power, False)
        self.rtde_c.stopScript()
