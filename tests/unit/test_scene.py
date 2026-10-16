"""Perception at its own interface: a rendered frame in, in-reach and out-of-reach objects out.

The frame is the orange on the table from `test_end_to_end`, so the real
mask_ray localiser and a real T_base_cam run on real geometry. Only OWLv2 and
SAM are faked, at the same two seams the end-to-end test uses.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from fyp.architecture_a.perception.scene import Perception
from fyp.architecture_a.trace import Stage, Trace
from fyp.shared.helpers.config import get_config

from test_end_to_end import R_ORANGE, FakeCamera, _render, _t_base_cam

T = _t_base_cam()
CFG0 = get_config()
TABLE_Z = float(CFG0["architecture_a"]["calibration"]["table_z"])
REACHABLE = np.array([0.45, -0.35, TABLE_Z + R_ORANGE])


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG0)
    c["architecture_a"]["trace"]["dir"] = str(tmp_path / "traces")
    return c


def _look(monkeypatch, centre, detected=True):
    """It takes the orange's centre and gives you the frame, with OWLv2 and SAM faked."""
    mask, depth = _render(T, np.asarray(centre, dtype=float), TABLE_Z)
    vs, us = np.nonzero(mask)
    box = [float(us.min()), float(vs.min()), float(us.max()), float(vs.max())]
    monkeypatch.setattr("fyp.architecture_a.perception.scene.detect",
                        lambda img, queries, model: [(queries[0], 0.9, box)] if detected else [])
    monkeypatch.setattr("fyp.architecture_a.perception.segmenter.segment",
                        lambda img, boxes, model: [(mask, 0.95) for _ in boxes])
    return FakeCamera(mask, depth).capture()


def test_the_object_is_found_where_it_is_in_base_frame(cfg, monkeypatch):
    frame = _look(monkeypatch, REACHABLE)
    seen = Perception(cfg, T).see(frame, ["orange"])

    (row,) = seen.in_reach
    assert row["name"] == "orange"
    assert np.linalg.norm(np.array(row["xyz"]) - REACHABLE) < 0.01
    (raw,) = seen.located
    assert np.linalg.norm(raw.p_base - REACHABLE) < 0.01
    assert seen.out_of_reach == []


def test_a_point_just_under_the_floor_is_lifted_at_runtime_but_raw_stays(cfg, monkeypatch):
    # Floor 1 cm above the orange's centre: the push "overshot" by 1 cm.
    floor = REACHABLE[2] + 0.01
    cfg["workspace"]["z"] = [floor, 0.38]
    frame = _look(monkeypatch, REACHABLE)
    seen = Perception(cfg, T).see(frame, ["orange"])

    (row,) = seen.in_reach
    assert row["xyz"][2] == pytest.approx(floor, abs=1e-4)
    # Calibration must see the localiser's own z, not the clamped one.
    assert seen.located[0].p_base[2] == pytest.approx(REACHABLE[2], abs=0.01)
    assert seen.located[0].p_base[2] < floor - 0.005


def test_a_point_far_under_the_floor_is_refused_not_lifted(cfg, monkeypatch):
    cfg["workspace"]["z"] = [REACHABLE[2] + 0.1, 0.38]   # 100 mm, beyond max_below_floor_mm
    frame = _look(monkeypatch, REACHABLE)
    seen = Perception(cfg, T).see(frame, ["orange"], strict=False)

    assert seen.in_reach == []
    assert "z" in seen.out_of_reach[0]["reason"]
    assert seen.located[0].p_base[2] == pytest.approx(REACHABLE[2], abs=0.01)


def test_an_object_out_of_reach_is_named_with_a_reason_and_no_position(cfg, monkeypatch):
    cfg["workspace"]["max_radius"] = 0.30   # the orange sits about 0.57 m from the base
    frame = _look(monkeypatch, REACHABLE)
    seen = Perception(cfg, T).see(frame, ["orange"], strict=False)

    assert seen.in_reach == []
    (out,) = seen.out_of_reach
    assert out["name"] == "orange"
    assert "radius" in out["reason"]
    assert "xyz" not in out
    # Still measured: the calibration scripts read this, not the reach split.
    assert np.linalg.norm(seen.located[0].p_base - REACHABLE) < 0.01


def test_strict_perception_fails_in_the_stage_that_found_nothing_usable(cfg, monkeypatch):
    cases = {}

    # Nothing detected.
    frame = _look(monkeypatch, REACHABLE, detected=False)
    cases[Stage.DETECTION] = (frame, cfg)

    # Detected, but no depth on the object.
    frame = _look(monkeypatch, REACHABLE)
    blind = type(frame)(rgb=frame.rgb, depth=np.full_like(frame.depth, np.nan),
                        intrinsics=frame.intrinsics)
    cfg_box = copy.deepcopy(cfg)
    cfg_box["architecture_a"]["perception"]["localiser"] = "box_depth"
    cases[Stage.DEPTH] = (blind, cfg_box)

    # Found, but out of reach.
    cfg_far = copy.deepcopy(cfg)
    cfg_far["workspace"]["max_radius"] = 0.30
    cases[Stage.TRANSFORM] = (frame, cfg_far)

    for stage, (fr, c) in cases.items():
        if stage is Stage.DETECTION:
            _look(monkeypatch, REACHABLE, detected=False)
        else:
            _look(monkeypatch, REACHABLE)
        trace = Trace("test", cfg=c)
        with pytest.raises(RuntimeError):
            Perception(c, T).see(fr, ["orange"], trace)
        assert trace.failed_stage is stage, stage


def test_a_successful_look_records_detection_depth_and_transform(cfg, monkeypatch):
    frame = _look(monkeypatch, REACHABLE)
    trace = Trace("test", cfg=cfg)
    Perception(cfg, T).see(frame, ["orange"], trace)

    assert [e.stage for e in trace.events] == [Stage.DETECTION, Stage.DEPTH, Stage.TRANSFORM]
    assert trace.events[0].data["found"] == ["orange"]
    assert trace.events[1].data["valid"] == 1
    assert trace.events[2].data["objects"][0]["name"] == "orange"


def test_non_strict_nothing_detected_gives_an_empty_result(cfg, monkeypatch):
    frame = _look(monkeypatch, REACHABLE, detected=False)
    seen = Perception(cfg, T).see(frame, ["orange"], strict=False)
    assert seen.in_reach == seen.out_of_reach == seen.located == []
