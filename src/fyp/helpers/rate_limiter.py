"""Drift-free fixed-rate loop helper.

Holds a loop at a target frequency using deadline-based sleeping: the next
deadline advances by a fixed period rather than being measured from "now", so
per-iteration overshoot does not accumulate into a drifting rate.

Currently unused. The sim server gates recording on wall-clock time directly
(see hardware/sim/server.py::_maybe_record); this class is kept for the real
robot's teleop loop, where a genuine fixed-rate loop is needed.
"""

import time


class RateControl:
    def __init__(self, hz: float = 20.0):
        self.dt = 1.0 / hz
        self._next_tick: float | None = None

    def start(self) -> None:
        self._next_tick = time.monotonic()

    def wait(self) -> None:
        if self._next_tick is None:
            raise RuntimeError("Call start() before wait().")
        self._next_tick += self.dt
        sleep_time = self._next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
