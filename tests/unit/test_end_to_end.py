"""One spoken instruction through the real harness and the real Architecture A, with no arm.

The rest of the unit suite proves that each module works alone. This proves they
connect: harness -> ArchitectureA.run -> grounding -> perception -> planner ->
validation -> skills -> envelope -> controller, and the outcome back into a
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
from types import SimpleNamespace

import numpy as np
import pytest

from fyp.architecture_a.pipeline import ArchitectureA
from fyp.architecture_a.skills import GRIPPER_OPEN
from fyp.evaluation.harness import Harness
from fyp.evaluation.suite import Tier
from fyp.shared.hardware.camera import Frame
from fyp.shared.hardware.intrinsics import CameraIntrinsics
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.helpers.config import PROJECT_ROOT, get_config

INSTRUCTION = "pick up the orange"
INTR = CameraIntrinsics(fx=640.0, fy=640.0, cx=640.0, cy=360.0, width=1280, height=720)
R_ORANGE = 0.027


# ------------------------------------------------------------------ the scene

def _camera() -> np.ndarray:
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
    db = np.stack([(uu - INTR.cx) / INTR.fx, (vv - INTR.cy) / INTR.fy,
                   np.ones_like(uu, dtype=float)], axis=-1) @ R.T
    oc = o - centre
    a, b, c = (db ** 2).sum(-1), 2 * (db @ oc), oc @ oc - R_ORANGE ** 2
    disc = b * b - 4 * a * c
    hit = disc > 0
    on_sphere = (-b - np.sqrt(np.where(hit, disc, 0))) / (2 * a)
    depth = np.where(hit, on_sphere, (table_z - o[2]) / db[..., 2])
    return hit, depth.astype(np.float32)


class FakeCamera:
    """It takes a rendered scene and gives you the same frame on every capture."""

    def __init__(self, mask, depth):
        self.mask = mask
        rgb = np.full((INTR.height, INTR.width, 3), 200, np.uint8)
        rgb[mask] = (255, 140, 0)
        self.frame = Frame(rgb=rgb, depth=depth, intrinsics=INTR)

    def capture(self, depth_frames: int = 1) -> Frame:
        return self.frame


# ------------------------------------------------------------- the other edges

class FakeController:
    """It takes nothing and gives you a controller that records instead of moving.
    A close stops at 40 mm, which is something held."""

    def __init__(self):
        self.calls: list[tuple] = []

    def move_to_pose(self, pose, speed=None, acc=None):
        self.calls.append(("move_to_pose", list(pose)))
        return True

    def move_joints(self, q, speed=None, acc=None):
        self.calls.append(("move_joints", list(q)))
        return True

    def gripper_toggle(self, state):
        self.calls.append(("gripper_toggle", state))
        return 99.8 if state == GRIPPER_OPEN else 40.0

    def get_state(self):
        return {"joint_pos": [0.0] * 6, "tcp_pose": [0.0] * 6, "gripper_state": GRIPPER_OPEN}

    @property
    def poses(self) -> list[list[float]]:
        return [c[1] for c in self.calls if c[0] == "move_to_pose"]


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


def _seen(kwargs) -> list[dict]:
    """It takes a planner request and gives you the object table the planner was shown."""
    first = kwargs["messages"][0]["content"]
    return json.loads(first.split("Detected objects:\n", 1)[1].split("\n\n", 1)[0])


class FakeLLM:
    """
    It takes a planner script and gives you an Anthropic client that plays it.

    Grounding is always answered with `look_for(["orange"])`. Each planner turn
    then takes the next entry of `script`: a function from the request to the
    reply's content blocks. Requests are kept in `calls`, so a test can see what
    the model was shown.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["tools"][0]["name"] == "look_for":
            content = [_tool("look_for", {"queries": ["orange"]})]
        else:
            content = self.script.pop(0)(kwargs)
        return SimpleNamespace(content=content,
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


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
    cfg["paths"]["calibration_dir"] = str(tmp_path / "calibration")
    cfg["architecture_a"]["trace"]["dir"] = str(tmp_path / "traces")

    T = _camera()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "T_base_cam.json").write_text(
        json.dumps({"T_base_cam": T.tolist()}))

    # Low in the reachable box, resting on the table.
    table_z = float(cfg["architecture_a"]["calibration"]["table_z"])
    centre = np.array([0.45, -0.35, table_z + R_ORANGE])
    mask, depth = _render(T, centre, table_z)
    vs, us = np.nonzero(mask)
    box = [float(us.min()), float(vs.min()), float(us.max()), float(vs.max())]

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("fyp.architecture_a.pipeline.detect",
                        lambda img, queries, model: [(queries[0], 0.9, box)])
    monkeypatch.setattr("fyp.architecture_a.perception.segmenter.segment",
                        lambda img, boxes, model: [(mask, 0.95) for _ in boxes])

    def install(script):
        llm = FakeLLM(script)
        monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: llm)
        return llm

    return SimpleNamespace(cfg=cfg, tmp=tmp_path, camera=FakeCamera(mask, depth),
                           centre=centre, install=install)


def _run_one_trial(cell, script, answer="the orange"):
    llm = cell.install(script)
    controller, ask = FakeController(), FakeAsk(answer)
    agent = ArchitectureA(cell.camera, controller, cfg=cell.cfg, ask=ask)
    harness = Harness(agent, "A", FakeEars(INSTRUCTION), cfg=cell.cfg,
                      out_dir=cell.tmp / "evaluation", n_trials=1)
    records = harness.run_tier(Tier(0), interactive=False)
    return records, controller, ask, llm


def _files_under(*dirs):
    return {p for d in dirs for p in (PROJECT_ROOT / d).rglob("*") if p.is_file()}


# ---------------------------------------------------------------------- tests

def test_a_spoken_instruction_reaches_the_arm_through_the_whole_chain(cell):
    before = _files_under("data", "logs")
    records, controller, _, llm = _run_one_trial(cell, [pick_what_was_seen, finish])

    assert len(records) == 1
    rec = records[0]
    assert rec.success and rec.failed_stage is None, rec
    assert rec.transcript == INSTRUCTION
    assert rec.run_id and not rec.asked_question

    # The planner saw the orange where it was rendered, so perception connects.
    seen = _seen(llm.calls[1])
    assert seen[0]["name"] == "orange"
    assert np.linalg.norm(np.array(seen[0]["xyz"]) - cell.centre) < 0.01

    env = WorkspaceEnvelope(cell.cfg)
    assert controller.poses, "the pick reached the controller as no motion at all"
    assert all(env.contains(p[:3]) for p in controller.poses), controller.poses

    # The harness wrote its record, and nothing was written into data/ or logs/.
    assert (cell.tmp / "evaluation" / "A_tier0.csv").is_file()
    assert _files_under("data", "logs") == before


def test_a_clarifying_question_is_asked_and_recorded(cell):
    question = [_tool("ask_user", {"question": "Which orange do you mean?"})]
    records, _, ask, _ = _run_one_trial(cell, [lambda kw: question, pick_what_was_seen, finish])

    assert ask.questions == ["Which orange do you mean?"]
    assert records[0].asked_question
    assert records[0].question_text == "Which orange do you mean?"
    assert records[0].success


def test_a_plan_outside_the_envelope_never_reaches_the_arm(cell):
    w = cell.cfg["workspace"]
    too_far = [w["x"][1] + 0.1, (w["y"][0] + w["y"][1]) / 2, w["z"][0] + 0.05]
    records, controller, _, _ = _run_one_trial(
        cell, [lambda kw: [_tool("pick", {"xyz": too_far})]])

    assert not records[0].success
    assert records[0].failed_stage == "planning"
    assert not [c for c in controller.calls if c[0] in ("move_to_pose", "move_joints")]


# --------------------------------------------------------------- the contract

def _architecture_a(cell):
    cell.install([pick_what_was_seen, finish])
    return ArchitectureA(cell.camera, FakeController(), cfg=cell.cfg, ask=FakeAsk("yes"))


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
