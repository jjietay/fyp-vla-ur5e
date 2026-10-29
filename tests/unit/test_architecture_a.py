"""Unit tests for the parts of Architecture A that need no robot and no API key.

Deliberately covers the safety envelope and plan validation hardest. Those two
are the only things standing between generated output and a physical arm, and
they are the only parts of the pipeline whose failure is expensive rather than
merely annoying.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fyp.architecture_a.planner import PlanRejected, Planner, validate_call
from fyp.architecture_a.tools import TOOL_NAMES, validate_schemas_against_skills
from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope
from fyp.shared.helpers.config import get_config

# The REAL config, not a hand-written one. Testing the envelope against invented
# bounds proves the envelope works on bounds the arm will never use. Every number
# below is derived from it, so editing config.yaml moves the tests with it rather
# than leaving them asserting on numbers that no longer ship.
CFG = get_config()
W = CFG["workspace"]
INSIDE = [(W["x"][0] + W["x"][1]) / 2,
          (W["y"][0] + W["y"][1]) / 2,
          (W["z"][0] + W["z"][1]) / 2]
OUTSIDE = [
    [W["x"][1] + 0.1, INSIDE[1], INSIDE[2]],   # x too far
    [INSIDE[0], W["y"][0] - 0.1, INSIDE[2]],   # y too far
    [INSIDE[0], INSIDE[1], W["z"][0] - 0.1],   # below the table
    [INSIDE[0], INSIDE[1], W["z"][1] + 0.1],   # above the ceiling
]


@pytest.fixture
def env():
    return WorkspaceEnvelope(CFG)


# ------------------------------------------------------------------- safety

def test_point_inside_is_accepted(env):
    assert env.contains(INSIDE)


@pytest.mark.parametrize("bad", OUTSIDE)
def test_point_outside_is_rejected(env, bad):
    assert not env.contains(bad)
    with pytest.raises(SafetyViolation):
        env.check_point(bad)


def test_violation_names_the_axis_and_the_overshoot(env):
    """The message has to say which axis and by how much, or it is not actionable."""
    below = W["z"][0] - 0.01
    with pytest.raises(SafetyViolation,
                       match=rf"z={below:.3f} is below min {W['z'][0]:.3f}"):
        env.check_point([INSIDE[0], INSIDE[1], below])


def test_nan_is_rejected(env):
    with pytest.raises(SafetyViolation):
        env.check_point([INSIDE[0], np.nan, INSIDE[2]])


def test_pose_must_have_six_elements(env):
    with pytest.raises(SafetyViolation, match="6 elements"):
        env.check_pose(INSIDE)


def test_orientation_is_not_bounded(env):
    # rotation vectors have no meaningful box; only translation is checked
    assert env.check_pose([*INSIDE, 99.0, -99.0, 12.0])


def test_speed_is_clamped_not_rejected(env):
    speed, acc = env.clamp_motion(10.0, 10.0)
    assert speed == W["max_speed"] and acc == W["max_acc"]


def test_zero_speed_is_an_error(env):
    with pytest.raises(SafetyViolation):
        env.clamp_motion(0.0, 0.1)


# ---------------------------------------------------------- plan validation

def test_unknown_tool_is_rejected(env):
    with pytest.raises(PlanRejected, match="unknown tool"):
        validate_call("teleport", {"xyz": [0, 0, 0.1]}, env)


def test_out_of_workspace_argument_is_rejected(env):
    with pytest.raises(PlanRejected, match="below min"):
        validate_call("pick", {"xyz": [INSIDE[0], INSIDE[1], W["z"][0] - 0.5]}, env)


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
        # The handle is INSIDE the workspace, so the enum is the only thing wrong.
        # It was [0.1, 0.1, 0.1] until 23 Sep 2026, when z min rose above 0.1 and
        # the call was rejected for its position instead.
        validate_call("open_drawer", {"handle_xyz": INSIDE, "pull_axis": "sideways"}, env)


def test_valid_call_passes_through(env):
    # INSIDE, not a fixed point. [0.1, 0.1, 0.2] was rejected for its position once
    # the workspace was measured on 1 Oct 2026, the same trap as the enum test above.
    args = validate_call("pick", {"xyz": INSIDE, "approach_height": 0.15}, env)
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
    assert validate_schemas_against_skills(Skills(_FakeController(), CFG)) == []


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


def test_layouts_lie_inside_the_envelope():
    """A layout the arm may not reach scores the envelope, not the architecture.
    Moving the arm or the envelope again must fail here before a trial is run."""
    import math
    from fyp.evaluation.suite import Tier, build_layouts
    from fyp.shared.helpers.config import get_config
    ws = get_config()["workspace"]
    lo, hi = ws["angle_deg"]
    for tier in Tier:
        for layout in build_layouts(tier, 20):
            for x, y in layout.positions.values():
                assert ws["min_radius"] <= math.hypot(x, y) <= ws["max_radius"]
                assert lo <= math.degrees(math.atan2(y, x)) <= hi


def test_every_tier_has_a_written_success_criterion():
    """A criterion that lives only in someone's head is how a marginal trial
    becomes a success on a good day and a failure on a bad one."""
    from fyp.evaluation.suite import TASKS, Tier
    for tier in Tier:
        assert TASKS[tier].success_criterion.strip()
        assert TASKS[tier].heldout_utterances


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


def test_grounding_asks_for_the_tool_call():
    """
    Without the tool call the parse returns nothing. Sonnet 5.5 refuses a forced
    tool_choice, so the call is asked for with a strict tool and a prompt rule
    instead, and a reply in prose still ends in GroundingFailed, never a fallback.
    """
    from fyp.architecture_a.planner import extract_queries
    client = _FakeClient(["cube"])
    extract_queries("put the cube away", client, "m")
    call = client.calls[0]
    assert call["tool_choice"] == {"type": "auto"}
    assert call["tools"][0]["strict"] is True
    assert "calling look_for" in call["system"]


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


def test_out_of_reach_objects_reach_the_planner_by_name_only(env):
    """
    13 Oct 2026: a box found by OWLv2 but out of reach was filtered before the
    planner, which then told the user it saw no box. It must be named, with a
    reason, and must carry no position the planner could act on.
    """
    planner = Planner(CFG, env)
    planner._turn = lambda: None
    planner.start("put the orange in the box",
                  [{"name": "orange", "confidence": 0.7, "xyz": INSIDE}],
                  [{"name": "white box", "confidence": 0.44,
                    "reason": "position radius=0.849 is above max 0.700"}])
    text = planner.messages[0]["content"]
    assert "Seen but out of reach" in text
    assert "white box" in text and "radius=0.849" in text
    after = text.split("Seen but out of reach")[1]
    assert "xyz" not in after


# ------------------------------------------------- the planner's turn loop

class ScriptedClient:
    """It takes a list of replies (each a list of content blocks) and gives you
    a client whose `messages.create` plays them in order, counting tokens."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = self

    def create(self, **kwargs):
        return SimpleNamespace(
            content=self.replies.pop(0),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def _use(name, args, tid):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=tid)


def _say(text):
    return SimpleNamespace(type="text", text=text)


def _planner(env, replies):
    return Planner(CFG, env, client=ScriptedClient(replies))


def _results(planner):
    """It takes a planner and gives you its tool_results, by tool_use_id."""
    return {r["tool_use_id"]: r for m in planner.messages
            if m["role"] == "user" and isinstance(m["content"], list)
            for r in m["content"]}


def test_a_rejected_call_goes_back_to_the_model_and_the_corrected_call_runs(env):
    bad = _use("pick", {"xyz": OUTSIDE[2]}, "t1")
    good = _use("pick", {"xyz": INSIDE}, "t2")
    planner = _planner(env, [[bad], [good]])

    reply = planner.start("pick it", [])

    assert [s.tool_use_id for s in reply.steps] == ["t2"]
    fed_back = _results(planner)["t1"]
    assert fed_back["is_error"] is True
    assert "below min" in fed_back["content"]
    assert planner.usage == {"input_tokens": 20, "output_tokens": 10, "calls": 2,
                             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def test_one_bad_call_in_a_reply_rejects_the_whole_reply_with_one_result_each(env):
    ok1 = _use("pick", {"xyz": INSIDE}, "a")
    bad = _use("place", {"xyz": OUTSIDE[0]}, "b")
    ok2 = _use("pick", {"xyz": INSIDE}, "c")
    planner = _planner(env, [[ok1, bad, ok2], [_use("pick", {"xyz": INSIDE}, "d")]])

    reply = planner.start("do it", [])

    assert [s.tool_use_id for s in reply.steps] == ["d"]
    results = _results(planner)
    assert set(results) == {"a", "b", "c"}
    assert results["b"]["content"].startswith("REJECTED:")
    for tid in ("a", "c"):
        assert results[tid]["content"].startswith("NOT EXECUTED")
    assert all(r["is_error"] for r in results.values())


def test_repeated_rejection_ends_as_plan_rejected_inside_the_turn_limit(env):
    n = int(CFG["architecture_a"]["planner"]["max_turns"])
    planner = _planner(env, [[_use("pick", {"xyz": OUTSIDE[2]}, f"t{i}")] for i in range(n + 3)])

    with pytest.raises(PlanRejected, match="below min"):
        planner.start("pick it", [])

    assert planner.usage["calls"] == n


def test_running_out_of_turns_without_a_rejection_is_gave_up(env):
    n = int(CFG["architecture_a"]["planner"]["max_turns"])
    planner = _planner(env, [[_use("pick", {"xyz": INSIDE}, f"t{i}")] for i in range(n)])

    reply = planner.start("pick it", [])
    for i in range(n - 1):
        assert reply.steps and not reply.gave_up
        reply = planner.report_results([(f"t{i}", "OK: done")])

    assert reply.steps
    reply = planner.report_results([(f"t{n - 1}", "OK: done")])
    assert reply.gave_up and not reply.is_done
    assert planner.usage["calls"] == n


def test_report_results_answers_steps_that_did_not_run_and_adds_the_scene(env):
    planner = _planner(env, [[_use("pick", {"xyz": INSIDE}, "a"),
                              _use("place", {"xyz": INSIDE}, "b")],
                             [_say("done")]])
    planner.start("move it", [])

    cup = {"name": "cup", "confidence": 0.5, "xyz": [0.4, -0.3, 0.2]}
    reply = planner.report_results([("a", "FAILED: no grasp")], scene=([cup], ["box"]))

    assert reply.is_done
    results = _results(planner)
    assert results["b"]["content"].startswith("SKIPPED")
    assert "Scene now" in results["b"]["content"]
    assert '"cup"' in results["b"]["content"]
    assert "out of reach: ['box']" in results["b"]["content"]
    assert "Scene now" not in results["a"]["content"]


def test_the_scene_after_a_step_carries_positions_so_a_failed_pick_shows(env):
    """
    13 Oct 2026: after a pick that lifted nothing, the scene said only "orange",
    so the model could not tell "held" from "still on the table" and reported
    success. With the position, an orange still at the table height, where it
    was, is visible to the model.
    """
    planner = _planner(env, [[_use("pick", {"xyz": INSIDE}, "a")], [_say("done")]])
    planner.start("pick it", [])
    orange = {"name": "orange", "confidence": 0.7, "xyz": [0.6478, -0.1597, 0.1954]}

    planner.report_results([("a", "OK: picked")], scene=([orange], []))

    content = _results(planner)["a"]["content"]
    assert "0.6478" in content and "-0.1597" in content and "0.1954" in content


# ----------------------------------------------------------------- API cost

def test_the_grounding_call_is_counted_in_the_run_usage():
    """Grounding is a model call of its own. Leaving it out of the tally would
    understate A's cost per command on every trial."""
    from fyp.architecture_a.planner import extract_queries, new_usage
    client = _FakeClient(["orange"])
    usage = new_usage()
    extract_queries("pick up the orange", client, "m", usage=usage)
    assert usage["calls"] == 1
    assert (usage["input_tokens"], usage["output_tokens"]) == (10, 5)


def test_api_cost_prices_each_token_kind_at_its_own_rate():
    from fyp.architecture_a.planner import api_cost_usd
    usage = {"calls": 3, "input_tokens": 1_000_000, "output_tokens": 100_000,
             "cache_creation_input_tokens": 200_000, "cache_read_input_tokens": 500_000}
    prices = {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.2}
    assert abs(api_cost_usd(usage, prices) - (2.0 + 1.0 + 0.5 + 0.1)) < 1e-9


def test_the_configured_price_is_the_planner_models_list_price():
    """A price that drifts from the model makes every cost in the report wrong."""
    from fyp.shared.helpers.config import get_config
    p = get_config()["architecture_a"]["planner"]
    assert p["model"] == "claude-sonnet-5-5", "the model changed: update price_usd_per_mtok"
    assert p["price_usd_per_mtok"] == {"input": 2.0, "output": 10.0,
                                       "cache_write": 2.5, "cache_read": 0.2}
