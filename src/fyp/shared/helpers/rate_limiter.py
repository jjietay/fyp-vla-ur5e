""" rate_limiter.py

This is a Drift-free fixed-rate loop helper.

Holds a loop at a target frequency using deadline-based sleeping: the next
deadline advances by a fixed period rather than being measured from "now", so
per-iteration overshoot does not accumulate into a drifting rate.

Currently unused. It exists for the hardware teleop recorder, where moveJ and
moveL block and state has to be polled from a separate thread on a real clock.
Counting iterations instead of gating on time is how a recorder silently ends up
capturing at a quarter of its configured rate.
"""

import time


class RateControl:
    """
    This class's main purpose is to ensure a specific rate is followed for control loop.
    """
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
