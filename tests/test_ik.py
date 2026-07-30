"""Regression tests for helpers/ik.py.

The one that matters is test_solve_ik_restores_qpos. solve_ik used to leave
data.qpos sitting at the solution, which made move_to_pose teleport: move_joints
read qpos as its start point, saw a zero delta, and returned without moving or
updating data.ctrl.

Needs MuJoCo, so these run in the FYP venv, not the lerobot one.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from fyp.helpers.ik import solve_ik
from fyp.hardware.sim.mujoco_controller import URControllerMuJoCo


@pytest.fixture
def ctrl():
    return URControllerMuJoCo()


def _target_mat(ctrl):
    return ctrl.data.site_xmat[ctrl._tcp_site_id].reshape(3, 3).copy()


def test_solve_ik_restores_qpos_on_success(ctrl):
    before = ctrl.data.qpos.copy()
    target = ctrl.data.site_xpos[ctrl._tcp_site_id].copy()
    target[0] += 0.05

    q_sol, ok = solve_ik(ctrl.model, ctrl.data, ctrl._tcp_site_id,
                         target, _target_mat(ctrl), **ctrl._ik_cfg)

    assert ok, "5 cm move should be reachable from home"
    np.testing.assert_allclose(ctrl.data.qpos, before, atol=1e-12)
    assert np.abs(q_sol - before[:6]).max() > 1e-6, "solution should differ from start"


def test_solve_ik_restores_qpos_on_failure(ctrl):
    before = ctrl.data.qpos.copy()
    unreachable = np.array([10.0, 10.0, 10.0])

    _q, ok = solve_ik(ctrl.model, ctrl.data, ctrl._tcp_site_id,
                      unreachable, _target_mat(ctrl), **ctrl._ik_cfg)

    assert not ok
    np.testing.assert_allclose(ctrl.data.qpos, before, atol=1e-12)


def test_move_to_pose_actually_moves(ctrl):
    """The bug this whole file exists for: the arm has to travel, not teleport."""
    start_tcp = np.array(ctrl.get_state()["tcp_pose"])
    target = start_tcp.copy()
    target[0] += 0.05

    ctrl.move_to_pose(target)

    end_tcp = np.array(ctrl.get_state()["tcp_pose"])
    moved = np.linalg.norm(end_tcp[:3] - start_tcp[:3])
    assert moved > 0.03, f"TCP only moved {moved * 1000:.1f} mm, expected ~50 mm"

    # ctrl must track the commanded pose, or the actuators drag the arm back
    np.testing.assert_allclose(ctrl.data.ctrl[:6], ctrl.data.qpos[:6], atol=5e-3)


def test_solve_ik_reaches_the_target(ctrl):
    target = ctrl.data.site_xpos[ctrl._tcp_site_id].copy()
    target[2] -= 0.08
    mat = _target_mat(ctrl)

    q_sol, ok = solve_ik(ctrl.model, ctrl.data, ctrl._tcp_site_id,
                         target, mat, **ctrl._ik_cfg)
    assert ok

    # replay the solution and confirm the site really lands there
    ctrl.data.qpos[:6] = q_sol
    mujoco.mj_forward(ctrl.model, ctrl.data)
    reached = ctrl.data.site_xpos[ctrl._tcp_site_id].copy()
    assert np.linalg.norm(reached - target) < 1e-3
