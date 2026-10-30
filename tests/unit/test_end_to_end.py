"""One spoken instruction through the real harness and the real Architecture A, with no arm.

The rest of the unit suite proves that each module works alone. This proves they
connect: harness -> ArchitectureA.run -> planner -> perceive -> perception ->
planner -> validation -> skills -> envelope -> controller, and the outcome back into a
trial record. A new pipeline file with the wrong signature, the wrong return
shape, or a motion that skips the envelope fails here, on a laptop, instead of
at the bench.

Fakes sit only at the edges, where the real thing needs a device, a GPU model or
the network: the microphone (`ears`), the `ask` callback, the camera, the
controller, the Anthropic client, OWLv2 and SAM. Everything between them is the
production code. The camera frame is a rendered orange on the table, so the real
mask_ray localiser and the real T_base_cam transform run on real geometry.
"""
from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from fyp.architecture_a.pipeline import ArchitectureA
from fyp.evaluation.harness import Harness
from fyp.evaluation.suite import Tier
from fyp.shared.hardware.camera import Frame
from fyp.shared.hardware.intrinsics import CameraIntrinsics
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.helpers.config import PROJECT_ROOT, get_config

from fakes import FakeController

INSTRUCTION = "pick up the orange"
INTR = CameraIntrinsics(fx=640.0, fy=640.0, cx=640.0, cy=360.0, width=1280, height=720)
R_ORANGE = 0.027


# ------------------------------------------------------------------ the scene

def _t_base_cam() -> np.ndarray:
    """It takes nothing and gives you a T_base_cam placed like the cell's camera."""
    o = np.array([0.725, 0.651, 0.863])
    z = np.array([0.04, -0.824, -0.565]); z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    y = -(up - (up @ z) * z); y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(y, z), y, z], axis=1)
    T[:3, 3] = o
    return T


def _render(T, centre, table_z):
    """It takes the camera, a sphere centre and the table height, and gives you
    (mask, depth): the sphere's pixels, and z-depth on the sphere or the table."""
    R, o = T[:3, :3], T[:3, 3]
    uu, vv = np.meshgrid(np.arange(INTR.width), np.arange(INTR.height))
    # One ray per pixel, in base frame, scaled so the camera z of each is 1:
    # then the ray parameter at a hit IS the z-depth the camera reports.
    db = np.stack([(uu - INTR.cx) / INTR.fx, (vv - INTR.cy) / INTR.fy,
                   np.ones_like(uu, dtype=float)], axis=-1) @ R.T
    oc = o - centre
    a, b, c = (db ** 2).sum(-1), 2 * (db @ oc), oc @ oc - R_ORANGE ** 2
    disc = b * b - 4 * a * c
    hit = disc > 0
    on_sphere = (-b - np.sqrt(np.where(hit, disc, 0))) / (2 * a)
    depth = np.where(hit, on_sphere, (table_z - o[2]) / db[..., 2])
    return hit, depth.astype(np.float32)


# Centre of a held orange above the TCP: the TCP is at the fingertips, and the
# fingers close a little under the middle of the orange.
HELD_ABOVE_TCP = 0.02


class Scene:
    """
    It takes the camera, the table and where the orange starts, and gives you
    where the orange is now.

    `mode` is what the orange does when the fake gripper closes on it:
    "follows" (it is held, and stays where it is released), "stays" (it never
    leaves the table, as on 13 Oct 2026), "vanishes" (the camera stops seeing
    it once it is held, as when the gripper hides it) or "sticks" (once held,
    it stays at the gripper even after the fingers open).
    """

    def __init__(self, T, table_z, start):
        self.T, self.table_z = T, table_z
        self.resting = np.asarray(start, dtype=float)
        self.controller = None
        self.mode = "follows"
        self._was_held = False
        self.mask = None

    def centre(self):
        held = self.controller is not None and self.controller.holding
        if self.mode == "stays":
            return self.resting
        if self.mode == "sticks" and (held or self._was_held):
            self._was_held = True
            return np.asarray(self.controller.tcp_pose[:3]) + [0.0, 0.0, HELD_ABOVE_TCP]
        if held:
            self._was_held = True
            if self.mode == "vanishes":
                return None
            return np.asarray(self.controller.tcp_pose[:3]) + [0.0, 0.0, HELD_ABOVE_TCP]
        if self._was_held:
            # Released: it sits on the table under where the gripper let go.
            x, y = self.controller.tcp_pose[:2]
            self.resting = np.array([x, y, self.table_z + R_ORANGE])
            self._was_held = False
        return self.resting


class FakeCamera:
    """It takes a rendered scene and gives you the same frame on every capture."""

    def __init__(self, mask, depth):
        self.mask = mask
        rgb = np.full((INTR.height, INTR.width, 3), 200, np.uint8)
        rgb[mask] = (255, 140, 0)
        self.frame = Frame(rgb=rgb, depth=depth, intrinsics=INTR)

    def capture(self, depth_frames: int = 1) -> Frame:
        return self.frame


class SceneCamera:
    """It takes a scene and gives you a frame of it as it is at each capture."""

    def __init__(self, scene):
        self.scene = scene

    def capture(self, depth_frames: int = 1) -> Frame:
        c = self.scene.centre()
        if c is None:
            mask = np.zeros((INTR.height, INTR.width), bool)
            _, depth = _render(self.scene.T, np.array([9.0, 9.0, 9.0]), self.scene.table_z)
        else:
            mask, depth = _render(self.scene.T, c, self.scene.table_z)
        self.scene.mask = mask
        rgb = np.full((INTR.height, INTR.width, 3), 200, np.uint8)
        rgb[mask] = (255, 140, 0)
        return Frame(rgb=rgb, depth=depth, intrinsics=INTR)


# ------------------------------------------------------------- the other edges

class FakeEars:
    """It takes a transcript and gives you a recogniser that always heard it."""

    def __init__(self, text):
        self.text = text

    def hear(self, prompt):
        return SimpleNamespace(text=self.text)


class FakeAsk:
    """It takes an answer and gives you an `ask` callback that records the question."""

    def __init__(self, answer):
        self.answer = answer
        self.questions: list[str] = []

    def __call__(self, question, options):
        self.questions.append(question)
        return self.answer


def _tool(name, args, i=0):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=f"toolu_{name}_{i}")


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _state_in(text) -> dict:
    """It takes a tool_result text and gives you the state in it."""
    return json.loads(text.split("State now:\n", 1)[1])


def _seen(kwargs) -> list[dict]:
    """It takes a planner request and gives you the objects in view in the
    latest state the planner was shown."""
    texts = [r["content"] for m in kwargs["messages"]
             if m["role"] == "user" and isinstance(m["content"], list)
             for r in m["content"] if "State now:" in r["content"]]
    return _state_in(texts[-1])["objects"]


class FakeLLM:
    """
    It takes a planner script and gives you an Anthropic client that plays it.

    Each planner turn takes the next entry of `script`: a function from the
    request to the reply's content blocks. Requests are kept in `calls`, so a
    test can see what the model was shown.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        # The code under test calls `client.messages.create(...)`. Pointing
        # `messages` back at this object lets one class stand in for both.
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.script.pop(0)(kwargs)
        return SimpleNamespace(content=content,
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def look(kwargs):
    return [_text("Find the orange."), _tool("perceive", {"queries": ["orange"]}, len(kwargs["messages"]))]


def pick_what_was_seen(kwargs):
    return [_tool("pick", {"xyz": _seen(kwargs)[0]["xyz"]}, len(kwargs["messages"]))]


def finish(kwargs):
    return [_text("Done: the orange is in the gripper.")]


# ---------------------------------------------------------------------- setup

@pytest.fixture
def cell(tmp_path, monkeypatch):
    """
    It takes a temporary directory and gives you everything a run needs, with
    every write redirected under it.

    The REAL config is used for the envelope, skills, planner and perception, so
    the run is checked against the limits the arm actually uses. Only the paths
    change: `data/` has no backup, and a test must never write into it.
    """
    cfg = copy.deepcopy(get_config())
    cfg["architecture_a"]["trace"]["dir"] = str(tmp_path / "traces")

    T = _t_base_cam()

    # Low in the reachable box, resting on the table.
    table_z = float(cfg["architecture_a"]["calibration"]["table_z"])
    centre = np.array([0.45, -0.35, table_z + R_ORANGE])
    scene = Scene(T, table_z, centre)

    def detect(img, queries, model):
        # OWLv2 boxes whatever orange the last frame showed, or nothing.
        vs, us = np.nonzero(scene.mask)
        if not len(us):
            return []
        return [(queries[0], 0.9, [float(us.min()), float(vs.min()),
                                   float(us.max()), float(vs.max())])]

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("fyp.architecture_a.perception.scene.detect", detect)
    monkeypatch.setattr("fyp.architecture_a.perception.segmenter.segment",
                        lambda img, boxes, model: [(scene.mask, 0.95) for _ in boxes])
    # The settle pause before each check has nothing to wait for here.
    monkeypatch.setattr("fyp.architecture_a.pipeline.time.sleep", lambda _s: None)

    return SimpleNamespace(cfg=cfg, tmp=tmp_path, T=T, camera=SceneCamera(scene),
                           centre=centre, scene=scene)


def _run_one_trial(cell, script, answer="the orange", controller_kw=None):
    llm = FakeLLM(script)
    controller, ask = FakeController(**(controller_kw or {})), FakeAsk(answer)
    cell.scene.controller = controller
    agent = ArchitectureA(cell.camera, controller, cfg=cell.cfg, ask=ask,
                          calibration=cell.T, planner_client=llm)
    # The operator presses Enter at every prompt: success is the operator's to
    # type, so these tests check what A did, never a typed verdict.
    harness = Harness(agent, "A", FakeEars(INSTRUCTION), cfg=cell.cfg,
                      out_dir=cell.tmp / "evaluation", n_trials=1, operator=lambda _p: "")
    records = harness.run_tier(Tier(0))
    cell.last_llm = llm
    return records, controller, ask, llm


# Not walked: tool state that changes on its own, and simulation/, which is an
# unrelated archive this project never touches.
_NOT_THE_PROJECT = {".git", ".venv", "simulation", "__pycache__", ".pytest_cache", ".mypy_cache"}


def _repo_snapshot() -> dict[str, int]:
    """It takes nothing and gives you every project file with its modification time."""
    snap = {}
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in _NOT_THE_PROJECT]
        for f in files:
            p = os.path.join(root, f)
            snap[p] = os.stat(p).st_mtime_ns
    return snap


# ---------------------------------------------------------------------- tests

def test_a_spoken_instruction_reaches_the_arm_through_the_whole_chain(cell):
    before = _repo_snapshot()
    records, controller, _, llm = _run_one_trial(cell, [look, pick_what_was_seen, finish])

    assert len(records) == 1
    rec = records[0]
    assert rec.failed_stage is None, rec
    assert rec.transcript == INSTRUCTION
    assert rec.run_id and not rec.asked_question

    # The planner saw the orange where it was rendered, so perception connects.
    seen = _state_in(_fed_back(llm)[0])["objects"]
    assert seen[0]["name"] == "orange"
    assert np.linalg.norm(np.array(seen[0]["xyz"]) - cell.centre) < 0.01

    env = WorkspaceEnvelope(cell.cfg)
    assert controller.poses, "the pick reached the controller as no motion at all"
    assert all(env.contains(p[:3]) for p in controller.poses), controller.poses
    # A joint target has no base-frame position to hold against the envelope
    # without forward kinematics, so a pick must send none.
    assert not [c for c in controller.calls if c[0] == "move_joints"]

    # The harness wrote its record, and nothing in the repo changed.
    assert (cell.tmp / "evaluation" / "A_tier0.csv").is_file()
    assert _repo_snapshot() == before


def test_a_clarifying_question_is_asked_and_recorded(cell):
    question = [_tool("ask_user", {"question": "Which orange do you mean?"})]
    records, _, ask, _ = _run_one_trial(cell, [look, lambda kw: question, pick_what_was_seen, finish])

    assert ask.questions == ["Which orange do you mean?"]
    assert records[0].asked_question
    assert records[0].question_text == "Which orange do you mean?"


def _too_far(cell):
    w = cell.cfg["workspace"]
    return [w["x"][1] + 0.1, (w["y"][0] + w["y"][1]) / 2, w["z"][0] + 0.05]


def test_a_plan_outside_the_envelope_never_reaches_the_arm(cell):
    # The model repeats the bad call until the turn limit, so the plan stays rejected.
    bad = lambda kw: [_tool("pick", {"xyz": _too_far(cell)}, len(kw["messages"]))]
    records, controller, _, _ = _run_one_trial(
        cell, [look] + [bad] * (int(cell.cfg["architecture_a"]["planner"]["max_turns"]) - 1))

    assert records[0].failed_stage == "planning"
    assert not [c for c in controller.calls if c[0] in ("move_to_pose", "move_joints")]


def test_a_rejected_plan_is_corrected_by_the_model_and_then_runs(cell):
    bad = lambda kw: [_tool("pick", {"xyz": _too_far(cell)}, len(kw["messages"]))]
    records, controller, _, llm = _run_one_trial(cell, [look, bad, pick_what_was_seen, finish])

    assert records[0].failed_stage is None
    assert controller.poses, "the corrected pick never reached the controller"
    # The model was shown why its first call failed.
    # The request holds the live message list, so index it: [-1] is a later turn.
    # 0 instruction, 1 look, 2 its result, 3 the bad pick, 4 the rejection.
    feedback = llm.calls[2]["messages"][4]["content"][0]
    assert feedback["is_error"] and feedback["content"].startswith("REJECTED:")


def _fed_back(llm) -> list[str]:
    """It takes the client and gives you every tool_result text the planner got,
    in order. The requests share one live message list, so read the last."""
    return [r["content"] for m in llm.calls[-1]["messages"]
            if m["role"] == "user" and isinstance(m["content"], list)
            for r in m["content"] if r.get("type") == "tool_result"]


def test_the_first_look_is_traced_as_the_planners_choice_of_words(cell):
    _run_one_trial(cell, [look, pick_what_was_seen, finish])

    lines = [json.loads(line) for line in next((cell.tmp / "traces").glob("*.jsonl")).open()]
    grounding = [r for r in lines if r.get("stage") == "grounding"]
    assert grounding and grounding[0]["data"]["queries"] == ["orange"]
    looks = [r for r in lines if r["type"] == "look"]
    assert looks[0]["found"]["orange"]
    assert _fed_back(cell.last_llm)[0].startswith("OK: looked for ['orange']")


def test_after_a_pick_the_object_is_in_the_gripper_by_id(cell):
    """Names alone hid a pick that lifted nothing (13 Oct 2026)."""
    _run_one_trial(cell, [look, pick_what_was_seen, finish])

    state = _state_in(_fed_back(cell.last_llm)[1])
    assert state["in_gripper"] == "orange_1"
    assert state["objects"] == []


# ------------------------------------------- the check after every motion

def test_a_pick_that_left_the_object_on_the_table_is_a_failed_step(cell):
    """
    13 Oct 2026: the gripper closed on the orange, the width said "held", and
    the orange never left the table. After the pick the camera finds it where it
    was, not at the gripper, so the step fails and the planner is told why.
    """
    cell.scene.mode = "stays"
    _run_one_trial(cell, [look, pick_what_was_seen, finish])

    result = _fed_back(cell.last_llm)[1]
    assert result.startswith("FAILED"), result
    assert "not at the gripper" in result
    state = _state_in(result)
    assert state["in_gripper"] is None
    assert [o["id"] for o in state["objects"]] == ["orange_1"]


def test_a_pick_with_the_object_at_the_gripper_is_verified(cell):
    _run_one_trial(cell, [look, pick_what_was_seen, finish])

    result = _fed_back(cell.last_llm)[1]
    assert result.startswith("OK") and "verified: orange_1 is at the gripper" in result


def test_an_object_the_camera_cannot_see_after_the_pick_is_unverified_not_failed(cell):
    """The gripper can hide the orange; failing every hidden pick hides the good ones."""
    cell.scene.mode = "vanishes"
    _run_one_trial(cell, [look, pick_what_was_seen, finish])

    result = _fed_back(cell.last_llm)[1]
    assert result.startswith("OK") and "unverified: orange_1 not seen" in result
    assert _state_in(result)["in_gripper"] == "orange_1"


def place_in_front(kwargs):
    xyz = [0.40, -0.30, float(get_config()["workspace"]["z"][0]) + 0.03]
    return [_tool("place", {"xyz": xyz}, len(kwargs["messages"]))]


def test_a_place_that_left_the_object_in_the_gripper_is_a_failed_step(cell):
    cell.scene.mode = "sticks"
    _run_one_trial(cell, [look, pick_what_was_seen, place_in_front, finish])

    pick, place = _fed_back(cell.last_llm)[1:3]
    assert pick.startswith("OK")
    assert place.startswith("FAILED"), place
    assert "still at the gripper" in place


def test_a_place_that_released_the_object_is_verified(cell):
    _run_one_trial(cell, [look, pick_what_was_seen, place_in_front, finish])

    place = _fed_back(cell.last_llm)[2]
    assert place.startswith("OK") and "verified: orange_1 released" in place
    state = _state_in(place)
    assert state["in_gripper"] is None
    assert [o["id"] for o in state["objects"]] == ["orange_1"]


# ------------------------------------------- the operator's verdict per step

class FakeJudge:
    """It takes the answers to give and gives you a `judge` callback that records
    what it was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked: list[str] = []

    def __call__(self, step):
        self.asked.append(step)
        return self.answers.pop(0)


def _agent(cell, script, judge):
    controller = FakeController()
    cell.scene.controller = controller
    llm = FakeLLM(script)
    agent = ArchitectureA(cell.camera, controller, cfg=cell.cfg, ask=FakeAsk("yes"),
                          calibration=cell.T, planner_client=llm, judge=judge)
    return agent, llm


def test_the_operator_judges_each_step_and_a_never_hears_the_answer(cell):
    """
    The operator's y/n is the ground truth for each step, beside A's own check.
    It must not reach the planner: an operator who tells A "that pick failed" is
    a perfect sensor Architecture B does not have.
    """
    judge = FakeJudge(False)
    agent, llm = _agent(cell, [look, pick_what_was_seen, finish], judge)

    outcome = agent.run(INSTRUCTION)

    assert len(judge.asked) == 1 and judge.asked[0].startswith("pick")
    assert [(j["tool"], j["a_ok"], j["operator_ok"]) for j in outcome.judged] == \
        [("pick", True, False)]
    fed = _fed_back(llm)
    assert fed[1].startswith("OK"), "A's own result reached the planner unchanged"
    assert not any("operator" in t.lower() for t in fed)


# --------------------------------------------------------------- the contract

def _architecture_a(cell):
    controller = FakeController()
    cell.scene.controller = controller
    return ArchitectureA(cell.camera, controller, cfg=cell.cfg, ask=FakeAsk("yes"),
                         calibration=cell.T,
                         planner_client=FakeLLM([look, pick_what_was_seen, finish]))


AGENTS = {
    "A": _architecture_a,
    "B": None,   # add B's factory here once it has a run() wrapper
}


@pytest.mark.parametrize("name", AGENTS)
def test_every_agent_returns_what_the_harness_reads(cell, name):
    """The harness reads `ok`, `failed_stage`, `asked`, `run_id` and `summary`
    and nothing else. Each agent must give all of them, the same way, or the
    harness grows a special case and stops being architecture-agnostic."""
    if AGENTS[name] is None:
        pytest.skip(f"no Architecture {name} agent yet")
    outcome = AGENTS[name](cell).run(INSTRUCTION)

    assert isinstance(outcome.ok, bool)
    assert outcome.failed_stage is None or isinstance(outcome.failed_stage, str)
    assert isinstance(outcome.asked, list)
    assert isinstance(outcome.run_id, str) and outcome.run_id
    assert isinstance(outcome.summary, str)


def test_the_planners_max_per_query_reaches_the_detector(cell):
    """Two bottles need two boxes for one query, and only the planner knows to ask."""
    look_for_two = lambda kw: [_tool("perceive", {"queries": ["orange"], "max_per_query": {"orange": 2}},
                                     len(kw["messages"]))]
    _run_one_trial(cell, [look_for_two, pick_what_was_seen, finish])

    lines = [json.loads(line) for line in next((cell.tmp / "traces").glob("*.jsonl")).open()]
    detection = [r for r in lines if r.get("stage") == "detection"]
    assert detection[0]["data"]["per_query"] == {"orange": 2}
    assert _fed_back(cell.last_llm)[0].startswith("OK: looked for ['orange']; found orange x")
    assert "of up to 2" in _fed_back(cell.last_llm)[0]


def test_a_trial_carries_the_cost_of_every_api_call(cell):
    """Three planning calls (perceive, pick, done), 10 in and 5 out each. Grounding
    is the perceive call, so no model call sits outside the planner's tally."""
    records, _, _, llm = _run_one_trial(cell, [look, pick_what_was_seen, finish])
    rec = records[0]
    assert rec.api_calls == len(llm.calls) == 3
    assert (rec.input_tokens, rec.output_tokens) == (30, 15)
    p = cell.cfg["architecture_a"]["planner"]["price_usd_per_mtok"]
    assert abs(rec.api_cost_usd - (30 * p["input"] + 15 * p["output"]) / 1e6) < 1e-9
