"""Unit tests for the RG2 gripper path in `URController`, with no arm.

Two halves. The first checks the URScript text and the word, because nothing can
run URScript away from the controller and a wrong number there is silent. The
second runs the Python side against fake RTDE interfaces that play back what the
controller would report, which covers every decision `gripper_toggle` makes.

What these cannot show is that the URScript behaves on the real controller.
`tests/hardware/test_controller.py` does that.
"""

from __future__ import annotations

import re

import pytest

from fyp.shared.hardware.ur5e_controller import (GripperError, GripperStatus,
                                                 URController, rg2_target_mm,
                                                 rg2_urscript, rg2_word)
from fyp.shared.helpers.config import get_config

CFG = get_config()
G = CFG["robot"]["gripper"]
REG = int(G["result_register"])
OPEN = rg2_target_mm(G["open_mm"])
CLOSE = rg2_target_mm(G["close_mm"])
TOL = float(G["tolerance_mm"])


# ------------------------------------------------------------- the word

def test_word_matches_the_value_confirmed_on_the_bench():
    """50 mm at 20 N gave 51 mm between the pads on 23 Sep 2026, from this word."""
    assert rg2_word(50, 20) == 37444


def test_word_clamps_like_the_urcap():
    assert rg2_word(500, 500) == rg2_word(100.8, 40)
    assert rg2_word(-50, 0) == rg2_word(-9.2, 3)


def test_word_fits_in_17_bits():
    assert 0 < rg2_word(200, 200) < 2 ** 17


def test_target_is_what_the_word_really_asks_for():
    """Whole millimetres of width + 9.2, so 50 is really 49.8."""
    assert rg2_target_mm(50) == pytest.approx(49.8)
    assert rg2_target_mm(0) == pytest.approx(-0.2)


# ------------------------------------------------------------ the script

def script(read_only=False, call_id=7):
    return rg2_urscript(rg2_word(0, 20), rg2_target_mm(0), TOL, int(G["max_sends"]),
                        call_id, REG, read_only=read_only)


def test_script_blocks_are_balanced():
    """
    Every `while`, `if` and `def` needs an `end`. A missing one is a syntax error
    that only the controller would report, in the lab.
    """
    lines = [ln.strip() for ln in script().splitlines()]
    opens = sum(1 for ln in lines if re.match(r"^(while|if) ", ln))
    ends = sum(1 for ln in lines if ln == "end")
    assert opens == ends


def test_script_writes_the_call_id_last():
    """
    Python reads the width and status once it sees its id. If the id were
    written first, Python could read the registers before the other two landed.
    """
    lines = [ln.strip() for ln in script(call_id=7).splitlines() if ln.strip()]
    assert lines[-1] == f"write_output_integer_register({REG}, 7)"
    assert f"write_output_float_register({REG}, rg_width)" in lines
    assert f"write_output_integer_register({REG + 1}, rg_status)" in lines


def test_script_sends_the_word_and_retries_from_config():
    text = script()
    assert f"rg_data = {rg2_word(0, 20)}" in text
    assert f"while rg_sends < {int(G['max_sends'])} and rg_done == False:" in text


def test_read_only_script_sends_nothing():
    text = script(read_only=True)
    assert "while rg_sends < 0 and" in text
    assert f"rg_status = {int(GripperStatus.READ)}" in text


def test_status_numbers_in_the_script_match_the_enum():
    """The numbers are the wire format. The enum and the script must agree."""
    text = script()
    for s in (GripperStatus.NOT_READY, GripperStatus.AT_TARGET,
              GripperStatus.MOVED, GripperStatus.TIMEOUT):
        assert f"rg_status = {int(s)}" in text


# --------------------------------------------------- the Python decisions

class FakeRTDE:
    """
    It takes a list of (status, width) results and gives you fake control and
    receive interfaces that report them, one per gripper script, in order.

    The call id is read out of the script text, which also checks that the
    script carries the id Python is waiting for. `stale_polls` makes the id
    register lag behind by that many reads, as RTDE can.
    """

    def __init__(self, results, ready=True, start_id=100, stale_polls=0, runs=True):
        self.results = list(results)
        self.ready = ready
        self.runs = runs
        self.stale_polls = stale_polls
        self.ints = {REG: start_id, REG + 1: -1}
        self.doubles = {REG: -1.0}
        self.scripts = []
        self._pending = None
        self._lag = 0

    # control
    def sendCustomScriptFunction(self, name, text):
        self.scripts.append(text)
        if not self.runs:
            return False
        call_id = int(re.findall(rf"write_output_integer_register\({REG}, (\d+)\)", text)[-1])
        status, width = self.results.pop(0)
        self._pending = (call_id, int(status), float(width))
        self._lag = self.stale_polls
        return True

    # receive
    def getActualDigitalInputBits(self):
        return (1 << 17) if self.ready else 0

    def getOutputIntRegister(self, reg):
        if self._pending is not None:
            if self._lag > 0:
                self._lag -= 1
            else:
                call_id, status, width = self._pending
                self.ints[REG], self.ints[REG + 1], self.doubles[REG] = call_id, status, width
                self._pending = None
        return self.ints[reg]

    def getOutputDoubleRegister(self, reg):
        return self.doubles[reg]

    def getActualQ(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return [0.0] * 6


def make(results, **kw) -> tuple[URController, FakeRTDE]:
    """It takes the results to play back and gives you a controller on fake RTDE."""
    fake = FakeRTDE(results, **kw)
    c = URController.__new__(URController)
    c.cfg = CFG
    c.rtde_c = c.rtde_r = fake
    c._reset_gripper()
    return c, fake


def test_state_is_none_before_start():
    c, _ = make([])
    assert c.get_state()["gripper_state"] is None


def test_toggle_before_start_raises():
    c, _ = make([])
    with pytest.raises(RuntimeError, match="before gripper_start"):
        c.gripper_toggle(1)


def test_start_reads_open_fingers_as_open():
    c, fake = make([(GripperStatus.READ, OPEN)])
    assert c.gripper_start() == "Gripper initialized."
    assert c.get_state()["gripper_state"] == 1
    assert "while rg_sends < 0 and" in fake.scripts[0]


def test_start_reads_anything_not_fully_open_as_closed():
    c, _ = make([(GripperStatus.READ, 40.0)])
    c.gripper_start()
    assert c.get_state()["gripper_state"] == 0


def test_start_raises_when_the_gripper_is_not_ready():
    c, fake = make([], ready=False)
    with pytest.raises(GripperError, match="ready"):
        c.gripper_start()
    assert fake.scripts == []


def test_start_counts_on_from_the_id_already_in_the_register():
    """An id left over from an earlier session must not be reused, or it matches at once."""
    c, fake = make([(GripperStatus.READ, OPEN)], start_id=5000)
    c.gripper_start()
    assert f"write_output_integer_register({REG}, 5001)" in fake.scripts[0]


def test_a_close_on_an_object_is_success_and_reports_the_width():
    c, _ = make([(GripperStatus.READ, OPEN), (GripperStatus.MOVED, 62.0)])
    c.gripper_start()
    assert c.gripper_toggle(0) == pytest.approx(62.0)
    assert c.get_state()["gripper_state"] == 0
    assert c.gripper_width_mm == pytest.approx(62.0)


def test_already_at_target_is_success():
    c, _ = make([(GripperStatus.READ, OPEN), (GripperStatus.AT_TARGET, OPEN)])
    c.gripper_start()
    assert c.gripper_toggle(1) == pytest.approx(OPEN)


@pytest.mark.parametrize("status", [GripperStatus.NO_RESPONSE, GripperStatus.TIMEOUT,
                                    GripperStatus.NOT_READY])
def test_a_failed_move_raises_and_keeps_the_old_state(status):
    """The lost command from the bench is NO_RESPONSE. It must not pass silently."""
    c, _ = make([(GripperStatus.READ, OPEN), (status, OPEN)])
    c.gripper_start()
    with pytest.raises(GripperError, match=status.name):
        c.gripper_toggle(0)
    assert c.get_state()["gripper_state"] == 1


def test_an_open_that_stops_short_raises():
    c, _ = make([(GripperStatus.READ, CLOSE), (GripperStatus.MOVED, OPEN - 3 * TOL)])
    c.gripper_start()
    with pytest.raises(GripperError, match="in the way"):
        c.gripper_toggle(1)


def test_python_waits_for_its_own_call_id():
    """RTDE can lag the script by a cycle or two. The old id must not be read as the result."""
    c, _ = make([(GripperStatus.READ, OPEN), (GripperStatus.MOVED, CLOSE)], stale_polls=3)
    c.gripper_start()
    assert c.gripper_toggle(0) == pytest.approx(CLOSE)


def test_no_result_at_all_raises():
    c, fake = make([(GripperStatus.READ, OPEN)])
    c.gripper_start()
    fake.results.append((GripperStatus.MOVED, CLOSE))
    fake.stale_polls = 10 ** 9
    with pytest.raises(GripperError, match="no result"):
        c.gripper_toggle(0)


def test_a_script_the_controller_refuses_raises():
    c, fake = make([(GripperStatus.READ, OPEN)])
    c.gripper_start()
    fake.runs = False
    with pytest.raises(GripperError, match="Remote Control"):
        c.gripper_toggle(0)


@pytest.mark.parametrize("bad", ["open", 7, True, -1])
def test_anything_that_is_not_zero_or_one_is_refused(bad):
    c, fake = make([(GripperStatus.READ, OPEN)])
    c.gripper_start()
    assert c.gripper_toggle(bad) == "Unable to control gripper."
    assert len(fake.scripts) == 1


def test_gripper_move_reports_a_failure_without_raising():
    """The bench path. check_gripper.py needs to print a lost command, not stop on it."""
    c, _ = make([(GripperStatus.READ, OPEN), (GripperStatus.NO_RESPONSE, OPEN)])
    c.gripper_start()
    status, width = c.gripper_move(50)
    assert status is GripperStatus.NO_RESPONSE
    assert width == pytest.approx(OPEN)


def test_gripper_move_uses_the_force_it_is_given():
    c, fake = make([(GripperStatus.READ, OPEN), (GripperStatus.MOVED, 49.8)])
    c.gripper_start()
    c.gripper_move(50, force_n=30)
    assert f"rg_data = {rg2_word(50, 30)}" in fake.scripts[-1]


def test_gripper_move_before_start_raises():
    c, _ = make([])
    with pytest.raises(RuntimeError, match="before gripper_start"):
        c.gripper_move(50)
