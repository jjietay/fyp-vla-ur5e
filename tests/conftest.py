"""Shared pytest configuration, applying to both test directories.

The layout splits on ONE question: does this test need a device plugged in.

    unit/       needs nothing but Python. Runs on a laptop, on the lab machine,
                and in CI, always, and is expected to be green at all times.
    hardware/   needs the real UR5e answering on the RTDE port at `robot.host`.
                THESE MOVE THE ARM.

Not split by what the test covers, because that answer changes as the project
grows and the directory would have to be reorganised each time. "Can I run this
right now" does not change, and it is the question you actually ask.

The `integration` marker is what enforces it. It exists so `pytest tests` is
meaningful away from the cell: before it, every controller test errored on a
refused TCP connection, the suite was permanently red, and it stopped being
usable as a "did I break something" check, which is exactly when you most want
one.

Hardware tests are skipped, not deleted. They run the moment the arm is
reachable, and `--integration` forces them to run and fail loudly if it is not.

There is no simulator behind this any more. The skip used to mean "the simulator
is not running" and now means "you are not at the arm", so a hardware test that
runs is a test moving real metal. Every one of them commands relative offsets from the
current pose and checks the envelope first, and anything added here must do the
same.

The marker is applied per module by `pytestmark` rather than per test, so adding
a file to `hardware/` and forgetting the marker is the one mistake that turns the
suite red on a laptop again. Check for it when adding a file there.
"""
from __future__ import annotations

import socket

import pytest

from fyp.shared.helpers.config import get_config



def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False,
                     help="run integration tests even if no controller answers")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs the real UR5e reachable at robot.host; MOVES THE ARM")


def _controller_reachable(host: str, port: int, timeout: float = 0.4) -> bool:
    """It takes a host and gives you whether something is listening on the RTDE port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    robot = get_config()["robot"]
    host, port = robot["host"], int(robot["rtde_port"])
    if _controller_reachable(host, port):
        return
    skip = pytest.mark.skip(
        reason=f"no UR5e on {host}:{port}. These tests move a real arm. Check "
               f"robot.host in config.yaml, or pass --integration to run anyway. "
               f"See docs/commands.md.")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
