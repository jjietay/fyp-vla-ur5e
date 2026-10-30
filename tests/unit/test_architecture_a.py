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


def test_every_tier_has_a_written_success_criterion():
    """A criterion that lives only in someone's head is how a marginal trial
    becomes a success on a good day and a failure on a bad one."""
    from fyp.evaluation.suite import TASKS, Tier
    for tier in Tier:
        assert TASKS[tier].success_criterion.strip()
        assert TASKS[tier].heldout_utterances


# ------------------------------------------------------------- perceive calls

def test_perceive_queries_are_normalised_and_deduped(env):
    args = validate_call("perceive", {"queries": ["  Red Cube ", "RED CUBE", "metal tray.", ""]}, env)
    assert args == {"queries": ["red cube", "metal tray"],
                    "max_per_query": {"red cube": 1, "metal tray": 1}}


def test_max_per_query_is_per_phrase_and_the_rest_keep_one(env):
    """Two bottles and a cup: a shared 2 would let a false second cup in."""
    args = validate_call("perceive", {"queries": ["bottle", "cup"],
                                      "max_per_query": {"Bottle": 2}}, env)
    assert args["max_per_query"] == {"bottle": 2, "cup": 1}


def test_max_per_query_for_a_phrase_not_looked_for_is_rejected(env):
    with pytest.raises(PlanRejected, match="not one of the queries"):
        validate_call("perceive", {"queries": ["bottle"], "max_per_query": {"cup": 2}}, env)


@pytest.mark.parametrize("k", [0, 5, True, 1.5, "2"])
def test_max_per_query_outside_one_to_the_cap_is_rejected(env, k):
    # True is an int in Python, so it must be refused on its own.
    with pytest.raises(PlanRejected, match="max_per_query"):
        validate_call("perceive", {"queries": ["bottle"], "max_per_query": {"bottle": k}}, env,
                      max_per_query=4)


def test_one_number_for_max_per_query_is_rejected(env):
    with pytest.raises(PlanRejected, match="map a phrase to a count"):
        validate_call("perceive", {"queries": ["bottle"], "max_per_query": 2}, env)


def test_perceive_over_the_query_cap_is_rejected_not_cut(env):
    """A silent cut drops the words ranked last, and those can be the destination."""
    with pytest.raises(PlanRejected, match="at most 3"):
        validate_call("perceive", {"queries": [f"object {i}" for i in range(4)]}, env,
                      max_queries=3)


def test_perceive_needs_a_noun_phrase(env):
    with pytest.raises(PlanRejected, match="at least one"):
        validate_call("perceive", {"queries": ["  ", ""]}, env)


def test_grounding_is_its_own_trace_stage():
    """
    A vocabulary error must not be recorded as a detection error, or the
    per-stage breakdown that carries H6 is wrong.
    """
    from fyp.architecture_a.trace import Stage
    assert Stage.GROUNDING.value == "grounding"
    assert Stage.GROUNDING != Stage.DETECTION


# -------------------------------------------------------------- the state

TOL = float(CFG["architecture_a"]["verify"]["tolerance_mm"]) / 1000.0


def _row(name, xyz, conf=0.5):
    return {"name": name, "confidence": conf, "xyz": list(xyz)}


def _state():
    from fyp.architecture_a.state import WorldState
    return WorldState(TOL)


def test_two_objects_with_one_name_get_two_ids():
    s = _state()
    s.update([_row("bottle", [0.4, -0.3, 0.2]), _row("bottle", [0.5, -0.2, 0.2])],
             [], ["bottle"], 1)
    assert sorted(o["id"] for o in s.render()["objects"]) == ["bottle_1", "bottle_2"]


def test_an_id_follows_position_when_the_planner_changes_its_words():
    """
    Tier 2: after "orange juice" the planner looks for "orange juice bottle".
    The bottle that answers is the same bottle_2, so the planner can act on the
    ID it already knows. A match by query would have given it a new ID.
    """
    s = _state()
    s.update([_row("bottle", [0.4, -0.3, 0.2]), _row("bottle", [0.5, -0.2, 0.2])],
             [], ["bottle"], 1)
    s.update([_row("orange juice bottle", [0.505, -0.2, 0.2])], [], ["orange juice bottle"], 3)

    shown = s.render()
    assert [(o["id"], o["name"]) for o in shown["objects"]] == [("bottle_2", "orange juice bottle")]
    assert shown["not_in_view"] == [{"id": "bottle_1", "name": "bottle", "last_seen_turn": 1,
                                     "why": "not looked for"}]


def test_nothing_out_of_view_carries_a_position():
    """A stale position is one the planner could act on; in Tier 3 a person moves things."""
    s = _state()
    s.update([_row("orange", [0.4, -0.3, 0.2])], [], ["orange"], 1)
    s.update([], [], ["orange"], 2)

    shown = s.render()
    assert shown["objects"] == []
    assert shown["not_in_view"][0]["why"] == "not seen"
    assert "xyz" not in shown["not_in_view"][0]
    assert s.target([0.4, -0.3, 0.2]) is None


def test_the_held_object_stays_in_the_gripper_by_id_with_no_position():
    s = _state()
    s.update([_row("orange", [0.4, -0.3, 0.2])], [], ["orange"], 1)
    s.grab("orange_1")
    tcp = [0.4, -0.3, 0.35]
    s.update([_row("orange", [0.4, -0.3, 0.37])], [], ["orange"], 2, gripper=tcp)

    shown = s.render()
    assert shown["in_gripper"] == "orange_1"
    assert shown["objects"] == [] and shown["not_in_view"] == []


def test_a_look_with_other_words_does_not_offer_the_held_object():
    """Holding orange_1, a look for "fruit" boxes it at the gripper. It is still
    the held object, not fruit_1 for the planner to pick."""
    s = _state()
    s.update([_row("orange", [0.4, -0.3, 0.2])], [], ["orange"], 1)
    s.grab("orange_1")
    tcp = [0.4, -0.3, 0.35]
    s.update([_row("fruit", [0.4, -0.3, 0.37])], [], ["fruit"], 2, gripper=tcp)

    assert s.render()["objects"] == []
    assert s.target([0.4, -0.3, 0.37]) is None


def test_a_released_object_keeps_its_id_where_it_landed():
    s = _state()
    s.update([_row("orange", [0.4, -0.3, 0.2])], [], ["orange"], 1)
    s.grab("orange_1")
    s.release([0.3, -0.45, 0.2], 2)
    s.update([_row("orange", [0.302, -0.45, 0.2])], [], ["orange"], 2)

    shown = s.render()
    assert shown["in_gripper"] is None
    assert [o["id"] for o in shown["objects"]] == ["orange_1"]


def test_out_of_reach_objects_are_shown_by_name_only():
    """
    13 Oct 2026: a box found by OWLv2 but out of reach was filtered before the
    planner, which then told the user it saw no box. It must be named, with a
    reason, and must carry no position the planner could act on.
    """
    s = _state()
    s.update([], [{"name": "white box", "confidence": 0.44,
                   "reason": "position radius=0.849 is above max 0.700"}], ["white box"], 1)
    assert s.render()["out_of_reach"] == [{"name": "white box", "confidence": 0.44,
                                           "reason": "position radius=0.849 is above max 0.700"}]


def test_a_pick_must_aim_at_an_object_in_view(env):
    s = _state()
    s.update([_row("orange", INSIDE)], [], ["orange"], 1)
    assert validate_call("pick", {"xyz": INSIDE}, env, state=s)
    away = [INSIDE[0], INSIDE[1] + 2 * TOL, INSIDE[2]]
    with pytest.raises(PlanRejected, match="not within"):
        validate_call("pick", {"xyz": away}, env, state=s)


def test_a_place_may_aim_at_open_table(env):
    s = _state()
    assert validate_call("place", {"xyz": INSIDE}, env, state=s)


# ------------------------------------------------- the planner's turn loop

class ScriptedClient:
    """It takes a list of replies (each a list of content blocks) and gives you
    a client whose `messages.create` plays them in order, counting tokens."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=self.replies.pop(0),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def _use(name, args, tid):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=tid)


def _say(text):
    return SimpleNamespace(type="text", text=text)


def _look(tid="look"):
    return _use("perceive", {"queries": ["orange"]}, tid)


def _planner(env, replies):
    return Planner(CFG, env, client=ScriptedClient(replies))


def _results(planner):
    """It takes a planner and gives you its tool_results, by tool_use_id."""
    return {r["tool_use_id"]: r for m in planner.messages
            if m["role"] == "user" and isinstance(m["content"], list)
            for r in m["content"]}


def test_the_model_starts_from_the_instruction_alone(env):
    """Architecture A must start from the same string Architecture B gets."""
    planner = _planner(env, [[_look()]])
    planner.start("place that red cube into that metal tray")
    assert planner.messages[0] == {"role": "user",
                                   "content": "Instruction: place that red cube into that metal tray"}


def test_parallel_tool_use_is_switched_off_at_the_api(env):
    planner = _planner(env, [[_look()]])
    planner.start("pick it")
    assert planner.client.calls[0]["tool_choice"] == {"type": "auto",
                                                      "disable_parallel_tool_use": True}


def test_the_first_call_must_be_perceive(env):
    planner = _planner(env, [[_use("pick", {"xyz": INSIDE}, "t1")], [_look("t2")]])

    reply = planner.start("pick it")

    assert reply.step.tool == "perceive"
    fed_back = _results(planner)["t1"]
    assert fed_back["is_error"] and "first call must be perceive" in fed_back["content"]


def test_a_reply_with_two_calls_is_refused_whole(env):
    """The model must see the pick's result before it chooses the place."""
    planner = _planner(env, [[_look()],
                             [_use("pick", {"xyz": INSIDE}, "a"), _use("place", {"xyz": INSIDE}, "b")],
                             [_use("pick", {"xyz": INSIDE}, "c")]])
    planner.start("move it")

    reply = planner.report("look", "OK: looked")

    assert reply.step.tool_use_id == "c"
    results = _results(planner)
    for tid in ("a", "b"):
        assert results[tid]["is_error"] and "one tool call per turn" in results[tid]["content"]


def test_the_intent_line_comes_back_with_the_call(env):
    planner = _planner(env, [[_say("Look for the orange first."), _look()]])
    reply = planner.start("pick it")
    assert reply.text == "Look for the orange first." and reply.step.tool == "perceive"


def test_a_rejected_call_goes_back_to_the_model_and_the_corrected_call_runs(env):
    planner = _planner(env, [[_look()], [_use("pick", {"xyz": OUTSIDE[2]}, "t1")],
                             [_use("pick", {"xyz": INSIDE}, "t2")]])
    planner.start("pick it")

    reply = planner.report("look", "OK: looked")

    assert reply.step.tool_use_id == "t2"
    fed_back = _results(planner)["t1"]
    assert fed_back["is_error"] is True
    assert "below min" in fed_back["content"]
    assert planner.usage | {"input_tokens": 30, "output_tokens": 15, "calls": 3} == planner.usage


def test_repeated_rejection_ends_as_plan_rejected_inside_the_turn_limit(env):
    n = int(CFG["architecture_a"]["planner"]["max_turns"])
    planner = _planner(env, [[_use("pick", {"xyz": INSIDE}, f"t{i}")] for i in range(n + 3)])

    with pytest.raises(PlanRejected, match="first call must be perceive"):
        planner.start("pick it")

    assert planner.usage["calls"] == n


def test_running_out_of_turns_without_a_rejection_is_gave_up(env):
    n = int(CFG["architecture_a"]["planner"]["max_turns"])
    planner = _planner(env, [[_look(f"t{i}")] for i in range(n)])

    reply = planner.start("pick it")
    for i in range(n - 1):
        assert reply.step and not reply.gave_up
        reply = planner.report(f"t{i}", "OK: looked")

    assert reply.step
    reply = planner.report(f"t{n - 1}", "OK: looked")
    assert reply.gave_up and not reply.is_done
    assert planner.usage["calls"] == n


def test_the_state_goes_inside_the_result(env):
    """
    13 Oct 2026: after a pick that lifted nothing, the scene said only "orange",
    so the model could not tell "held" from "still on the table" and reported
    success. The state carries the position and what is in the gripper.
    """
    planner = _planner(env, [[_look()], [_say("done")]])
    planner.start("pick it")
    s = _state()
    s.update([_row("orange", [0.6478, -0.1597, 0.1954])], [], ["orange"], 1)

    reply = planner.report("look", "OK: looked", s.render())

    assert reply.is_done
    content = _results(planner)["look"]["content"]
    assert "State now" in content and '"in_gripper": null' in content
    assert "0.6478" in content and '"orange_1"' in content


# ----------------------------------------------------------------- API cost

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
