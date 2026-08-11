"""Unit tests for the parts of Architecture A that need no robot and no API key.

Deliberately covers the safety envelope and plan validation hardest. Those two
are the only things standing between generated output and a physical arm, and
they are the only parts of the pipeline whose failure is expensive rather than
merely annoying.
"""
from __future__ import annotations

import numpy as np
import pytest

from fyp.architecture_a.planner import PlanRejected, validate_call
from fyp.architecture_a.tools import TOOL_NAMES, validate_schemas_against_skills
from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope

CFG = {"workspace": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 0.6],
                     "max_speed": 0.25, "max_acc": 0.5}}


@pytest.fixture
def env():
    return WorkspaceEnvelope(CFG)


# ------------------------------------------------------------------- safety

def test_point_inside_is_accepted(env):
    assert env.contains([0.0, 0.0, 0.3])


@pytest.mark.parametrize("bad", [
    [0.6, 0.0, 0.3],    # x too far
    [0.0, -0.6, 0.3],   # y too far
    [0.0, 0.0, -0.01],  # below the table
    [0.0, 0.0, 0.7],    # above the ceiling
])
def test_point_outside_is_rejected(env, bad):
    assert not env.contains(bad)
    with pytest.raises(SafetyViolation):
        env.check_point(bad)


def test_violation_names_the_axis_and_the_overshoot(env):
    with pytest.raises(SafetyViolation, match=r"z=-0\.010 is below min 0\.000"):
        env.check_point([0.0, 0.0, -0.01])


def test_nan_is_rejected(env):
    with pytest.raises(SafetyViolation):
        env.check_point([0.0, np.nan, 0.3])


def test_pose_must_have_six_elements(env):
    with pytest.raises(SafetyViolation, match="6 elements"):
        env.check_pose([0.0, 0.0, 0.3])


def test_orientation_is_not_bounded(env):
    # rotation vectors have no meaningful box; only translation is checked
    assert env.check_pose([0.0, 0.0, 0.3, 99.0, -99.0, 12.0])


def test_speed_is_clamped_not_rejected(env):
    speed, acc = env.clamp_motion(10.0, 10.0)
    assert speed == 0.25 and acc == 0.5


def test_zero_speed_is_an_error(env):
    with pytest.raises(SafetyViolation):
        env.clamp_motion(0.0, 0.1)


# ---------------------------------------------------------- plan validation

def test_unknown_tool_is_rejected(env):
    with pytest.raises(PlanRejected, match="unknown tool"):
        validate_call("teleport", {"xyz": [0, 0, 0.1]}, env)


def test_out_of_workspace_argument_is_rejected(env):
    with pytest.raises(PlanRejected, match="below min"):
        validate_call("pick", {"xyz": [0.0, 0.0, -0.5]}, env)


def test_missing_required_argument_is_rejected(env):
    with pytest.raises(PlanRejected, match="missing required"):
        validate_call("pour", {"source_xyz": [0.1, 0.1, 0.1]}, env)


def test_unexpected_argument_is_rejected(env):
    with pytest.raises(PlanRejected, match="unexpected"):
        validate_call("pick", {"xyz": [0.1, 0.1, 0.1], "velocity": 3}, env)


def test_malformed_position_is_rejected(env):
    with pytest.raises(PlanRejected, match="not a 3-element position"):
        validate_call("pick", {"xyz": "over there"}, env)


def test_bad_enum_is_rejected(env):
    with pytest.raises(PlanRejected, match="must be one of"):
        validate_call("open_drawer", {"handle_xyz": [0.1, 0.1, 0.1], "pull_axis": "sideways"}, env)


def test_valid_call_passes_through(env):
    args = validate_call("pick", {"xyz": [0.1, 0.1, 0.2], "approach_height": 0.15}, env)
    assert args["approach_height"] == 0.15


def test_ask_user_needs_a_question(env):
    with pytest.raises(PlanRejected, match="empty question"):
        validate_call("ask_user", {"question": "  "}, env)


def test_ask_user_is_exposed_to_the_model(env):
    # Tier 2 depends on this tool existing. If it is ever dropped, the
    # clarification result silently becomes untestable.
    assert "ask_user" in TOOL_NAMES


# --------------------------------------------------------- schema agreement

def test_schemas_match_skill_signatures():
    """The failure this catches is a schema promising an argument no skill accepts."""
    class _FakeController:
        def move_to_pose(self, *a, **k): return True
        def move_joints(self, *a, **k): return True
        def gripper_toggle(self, *a, **k): pass
        def get_state(self): return {}

    from fyp.architecture_a.skills import Skills
    cfg = dict(CFG)
    cfg["architecture_a"] = {"skills": {
        "approach_height": 0.12, "retreat_height": 0.15, "grasp_depth_offset": 0.0,
        "speed": 0.1, "acc": 0.3, "settle_s": 0.0, "down_rotvec": [3.142, 0.0, 0.0]}}
    assert validate_schemas_against_skills(Skills(_FakeController(), cfg)) == []


# ---------------------------------------------------------------- rotations

def test_rotvec_round_trip_is_exact_near_pi():
    """
    Regression test. The old R_to_rotvec divided by sin(theta), which collapses
    as theta approaches pi. That is not an edge case here: the default tool
    orientation is the gripper pointing down, whose rotation vector has theta
    almost exactly pi.
    """
    from fyp.shared.helpers.rotations import R_to_rotvec, rotvec_to_R
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rv = axis * (np.pi - 10.0 ** rng.uniform(-14, -3))
        R = rotvec_to_R(rv)
        worst = max(worst, float(np.abs(rotvec_to_R(R_to_rotvec(R)) - R).max()))
    assert worst < 1e-13, f"round trip near pi degraded to {worst:.2e}"


def test_euler_round_trip():
    from fyp.shared.helpers.rotations import euler_to_rotvec, rotvec_to_euler
    rng = np.random.default_rng(1)
    for _ in range(2000):
        e = np.array([rng.uniform(-np.pi, np.pi),
                      rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05),
                      rng.uniform(-np.pi, np.pi)])
        assert np.abs(rotvec_to_euler(euler_to_rotvec(e)) - e).max() < 1e-12


def test_euler_to_R_is_a_rotation():
    from fyp.shared.helpers.rotations import euler_to_R
    rng = np.random.default_rng(2)
    for _ in range(500):
        R = euler_to_R(rng.uniform(-np.pi, np.pi, 3))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-13)
        assert np.isclose(np.linalg.det(R), 1.0)


# --------------------------------------------------------------- evaluation

def test_layouts_are_identical_across_architectures():
    """Both architectures must see the same 20 scenes in the same order, or any
    difference in success rate could be the layouts rather than the policy."""
    from fyp.evaluation.suite import Tier, build_layouts
    a = build_layouts(Tier.PICK_PLACE, 20)
    b = build_layouts(Tier.PICK_PLACE, 20)
    assert [x.positions for x in a] == [x.positions for x in b]
    assert [x.utterance for x in a] == [x.utterance for x in b]


def test_layouts_respect_minimum_separation():
    import math
    from fyp.evaluation.suite import Tier, build_layouts
    for tier in Tier:
        for layout in build_layouts(tier, 20):
            pts = list(layout.positions.values())
            for i, p in enumerate(pts):
                for q in pts[i + 1:]:
                    assert math.dist(p, q) >= 0.12 - 1e-9


def test_every_tier_has_a_written_success_criterion():
    """A criterion that lives only in someone's head is how a marginal trial
    becomes a success on a good day and a failure on a bad one."""
    from fyp.evaluation.suite import TASKS, Tier
    for tier in Tier:
        assert TASKS[tier].success_criterion.strip()
        assert TASKS[tier].heldout_utterances


def test_harness_does_not_import_either_architecture():
    """The harness must not be able to see inside A or B, or it grows an
    affordance for one of them and the comparison stops being like-for-like.

    Checks imports via the AST rather than raw text, because the package
    docstring legitimately names both architectures while explaining this rule.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "fyp" / "evaluation"
    offenders = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if "architecture_a" in name or "architecture_b" in name:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, offenders


# ----------------------------------------------------------------- grounding

class _FakeBlock:
    type = "tool_use"
    name = "look_for"

    def __init__(self, queries):
        self.input = {"queries": queries}


class _FakeResponse:
    def __init__(self, queries):
        self.content = [_FakeBlock(queries)]
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class _FakeClient:
    """Records the call and returns whatever queries the test wants."""

    def __init__(self, queries):
        self._queries = queries
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._queries)


def test_grounding_normalises_and_dedupes():
    from fyp.architecture_a.planner import extract_queries
    client = _FakeClient(["  Red Cube ", "RED CUBE", "metal tray.", "", "  "])
    assert extract_queries("x", client, "m") == ["red cube", "metal tray"]


def test_grounding_respects_max_queries():
    from fyp.architecture_a.planner import extract_queries
    client = _FakeClient([f"object {i}" for i in range(20)])
    assert len(extract_queries("x", client, "m", max_queries=3)) == 3


def test_grounding_forces_the_tool_call():
    """Without tool_choice the model can reply with prose and the parse returns nothing."""
    from fyp.architecture_a.planner import extract_queries
    client = _FakeClient(["cube"])
    extract_queries("put the cube away", client, "m")
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "look_for"}


def test_grounding_raises_rather_than_falling_back():
    """
    A silent fallback to a default vocabulary would restore the very advantage
    this step removes, and would do it invisibly on the hardest trials.
    """
    from fyp.architecture_a.planner import GroundingFailed, extract_queries
    with pytest.raises(GroundingFailed):
        extract_queries("hello", _FakeClient([]), "m")


def test_grounding_sees_only_the_instruction():
    """Architecture A must start from the same string Architecture B gets."""
    from fyp.architecture_a.planner import extract_queries
    client = _FakeClient(["cube"])
    extract_queries("place that red cube into that metal tray", client, "m")
    sent = client.calls[0]["messages"]
    assert sent == [{"role": "user",
                     "content": "place that red cube into that metal tray"}]


def test_grounding_is_its_own_trace_stage():
    """
    A vocabulary error must not be recorded as a detection error, or the
    per-stage breakdown that carries H6 is wrong.
    """
    from fyp.architecture_a.trace import Stage
    assert Stage.GROUNDING.value == "grounding"
    assert Stage.GROUNDING != Stage.DETECTION
