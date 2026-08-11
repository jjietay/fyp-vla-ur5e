"""Shared pytest configuration.

The `integration` marker exists so `pytest tests` is meaningful on a machine with
no robot attached. Before this, every controller test errored on a refused TCP
connection, so the suite was permanently red and stopped being usable as a "did I
break something" check, which is exactly when you most want one.

Integration tests are skipped, not deleted. They still run the moment URSim or the
real arm is reachable, and `--integration` forces them to run and fail loudly if
it is not.
"""
from __future__ import annotations

import socket

import pytest

from fyp.shared.helpers.config import get_config

RTDE_PORT = 30004


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False,
                     help="run integration tests even if no controller answers")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs a reachable UR controller (URSim or the real arm)")


def _controller_reachable(host: str, port: int = RTDE_PORT, timeout: float = 0.4) -> bool:
    """It takes a host and gives you whether something is listening on the RTDE port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    host = get_config()["robot"]["host"]
    if _controller_reachable(host):
        return
    skip = pytest.mark.skip(
        reason=f"no UR controller on {host}:{RTDE_PORT}. Start URSim, or pass --integration "
               f"to run anyway. See docs/commands.md.")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
