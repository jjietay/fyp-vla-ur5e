"""planner.py

It takes an instruction and gives you, one turn at a time, the next thing
Architecture A should do: look, run one skill, ask the user, or stop.

The model never sees pixels and never emits a pose. It chooses what the
detector looks for, reads a state of object IDs with base-frame coordinates, and
selects one tool per turn with arguments drawn from that state. That restriction
is deliberate: a wrong plan is always readable as a wrong *decision*, not as an
inscrutable numeric output, and that readability is what Architecture A is
being measured on.

One call per turn (ADR 0015). The model sees the result of each skill, and the
fresh state after it, before it chooses the next. A reply that chains pick and
place would run the place before anyone knew whether the pick worked.

Every tool call is validated before it is allowed anywhere near the controller.
An LLM will occasionally invent a coordinate, invent a skill, or place an object
somewhere off the bench, and none of those should be discovered by watching the
arm move.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from fyp.architecture_a.state import WorldState
from fyp.architecture_a.tools import TOOL_NAMES, TOOLS
from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope
from fyp.shared.helpers.config import get_config

SYSTEM_PROMPT = """You control a UR5e robot arm with a parallel gripper on a table.

Carry out the user's instruction by calling the tools available to you, one \
call per turn. Before each call, write one short line saying what you intend \
and why.

Rules:
- Your first call must be perceive. You see nothing until you look.
- Every result comes with "State now": the object in the gripper, the objects \
in view with an id and a position in the robot's base frame in metres, the \
objects no longer in view, and the objects seen but out of reach.
- Only act on positions taken from "objects" in the latest state. Never invent \
coordinates. Objects under "not_in_view" and "out_of_reach" have no position \
and must not be used. Call perceive again if what you need is not in view.
- Refer to objects by their id when you write or ask. Two objects with the same \
name are different objects. The skills take positions: copy the xyz of the \
object you mean from the latest state.
- perceive keeps only the best match for each name unless you give that name \
a higher count in max_per_query. Raise it only for a name that several objects \
may share, for example {{"bottle": 2}}.
- If the instruction names an object you cannot find, say so instead of \
substituting a similar one. Try perceive with other words first if the words \
you chose may be the reason.
- If an object the task needs is out of reach, do not move anything: tell the \
user it is out of reach and should be moved closer to the arm.
- The gripper holds one object at a time. Always place what you are holding \
before picking something else.
- If the instruction does not determine which object to act on, call ask_user. \
Do not guess and do not pick the first plausible candidate. After the answer, \
call perceive with the user's words if they name something you have not looked for.
- A result that starts with FAILED means the step did not work. Decide from the \
reason and the state whether to try again or to stop.
- Work in the order the task requires. If something must be open before you can \
put an object inside it, open it first.
- When the task is complete, stop calling tools and reply with one short sentence \
describing what you did.

Workspace limits: {workspace}
"""


# Sonnet 5.5 (adopted 1 Oct 2026, when Sonnet 4.5 was deprecated) thinks by default
# and rejects thinking "disabled". "between_tools" is the lowest setting it accepts,
# and it keeps the planner as it was designed: no up-front thinking, so latency and
# token use stay comparable across runs. It is refused above effort "high", so the
# two settings travel together. Sent through extra_body so that an SDK older than
# these fields passes them on rather than rejecting them.
MODEL_OPTIONS = {"thinking": {"type": "between_tools"}, "output_config": {"effort": "high"}}

# The asked-for API setting: at most one tool_use per reply. `_read` refuses a
# reply with more anyway, so a model or SDK that ignores this cannot chain steps.
TOOL_CHOICE = {"type": "auto", "disable_parallel_tool_use": True}

# The argument of each skill that must land on an object in view. A pick aimed at
# a position no look has just confirmed is a pick at empty table, or at where a
# person moved something from. `place` and pour's target stay free inside the
# envelope, since putting something down on open table is a valid instruction.
TARGET_ARGS = {"pick": "xyz", "pour": "source_xyz", "open_drawer": "handle_xyz"}


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
              "cache_read_input_tokens")


def new_usage() -> dict:
    """It takes nothing and gives you an empty tally of API calls and tokens."""
    return {"calls": 0, **{k: 0 for k in USAGE_KEYS}}


def add_usage(usage: dict, response) -> None:
    """
    It takes a running tally and one API response and gives you the tally with
    that call added.

    Thinking tokens arrive inside `output_tokens` and are billed as output, so
    nothing extra is needed for them. Cache writes and reads are billed at their
    own rates and are kept apart for that reason.
    """
    usage["calls"] += 1
    for k in USAGE_KEYS:
        usage[k] += int(getattr(response.usage, k, 0) or 0)


def api_cost_usd(usage: dict, prices: dict) -> float:
    """
    It takes a tally of tokens and the price per million tokens and gives you
    the cost in US dollars.

    `prices` holds `input`, `output`, `cache_write` and `cache_read`, all per
    million tokens, from `architecture_a.planner.price_usd_per_mtok`.
    """
    per = {"input_tokens": "input", "output_tokens": "output",
           "cache_creation_input_tokens": "cache_write",
           "cache_read_input_tokens": "cache_read"}
    return sum(usage.get(k, 0) * float(prices.get(v, 0.0)) for k, v in per.items()) / 1e6


class PlanRejected(Exception):
    """Raised when a proposed tool call fails validation and must not be dispatched."""


@dataclass
class Step:
    """One validated tool call, ready to dispatch."""
    tool: str
    args: dict
    tool_use_id: str = ""


@dataclass
class PlannerReply:
    """
    What the planner produced this turn: one step to run (a look or a skill), a
    question to ask, a final message meaning it considers the task done, or
    `gave_up` meaning the turn limit ran out first. `text` is the model's line of
    intent, or its final message.

    Every reply here is already validated. A rejected call never becomes a
    reply: the planner sends it back to the model, or raises PlanRejected when
    the turn limit leaves no room for another try.
    """
    step: Step | None = None
    question: str | None = None
    options: list[str] = field(default_factory=list)
    question_id: str = ""
    text: str = ""
    gave_up: bool = False

    @property
    def is_done(self) -> bool:
        return self.step is None and self.question is None and not self.gave_up


def validate_call(tool: str, args: dict, envelope: WorkspaceEnvelope, *,
                  max_queries: int = 8, max_per_query: int = 4,
                  state: WorldState | None = None) -> dict:
    """
    It takes a proposed tool call and gives you the arguments back, or raises
    PlanRejected explaining exactly what was wrong.

    `perceive` comes back with its queries normalised (lower case, single
    spaces, no repeats) and `max_per_query` as a count for every query, 1 where
    the model gave none. With a `state`, the target of a pick, a pour or a drawer
    pull must be an object in view.

    This runs before dispatch, every time, with no fast path. The checks are
    cheap and the thing they protect is expensive.
    """
    if tool not in TOOL_NAMES:
        raise PlanRejected(f"unknown tool {tool!r}; available: {sorted(TOOL_NAMES)}")

    if tool == "ask_user":
        if not str(args.get("question", "")).strip():
            raise PlanRejected("ask_user called with an empty question")
        return args

    if tool == "perceive":
        unknown = set(args) - {"queries", "max_per_query"}
        if unknown:
            raise PlanRejected(f"perceive got unexpected arguments {sorted(unknown)}")
        queries = _queries(args, max_queries)
        return {"queries": queries,
                "max_per_query": _counts(args.get("max_per_query", {}), queries, max_per_query)}

    schema = next(t for t in TOOLS if t["name"] == tool)["input_schema"]
    allowed = set(schema.get("properties", {}))
    unknown = set(args) - allowed
    if unknown:
        raise PlanRejected(f"{tool} got unexpected arguments {sorted(unknown)}")
    for req in schema.get("required", []):
        if req not in args:
            raise PlanRejected(f"{tool} is missing required argument {req!r}")

    for key, value in args.items():
        spec = schema["properties"][key]
        if spec.get("type") == "array" and spec.get("minItems") == 3:
            try:
                p = np.asarray(value, dtype=float).reshape(3)
            except Exception as e:
                raise PlanRejected(f"{tool}.{key} is not a 3-element position: {value!r}") from e
            try:
                envelope.check_point(p, f"{tool}.{key}")
            except SafetyViolation as e:
                raise PlanRejected(str(e)) from e
        elif spec.get("type") == "number" and not isinstance(value, (int, float)):
            raise PlanRejected(f"{tool}.{key} must be a number, got {type(value).__name__}")
        elif spec.get("type") == "string" and "enum" in spec and value not in spec["enum"]:
            raise PlanRejected(f"{tool}.{key} must be one of {spec['enum']}, got {value!r}")

    key = TARGET_ARGS.get(tool)
    if state is not None and key is not None and state.target(args[key]) is None:
        raise PlanRejected(
            f"{tool}.{key} {list(args[key])} is not within "
            f"{state.tolerance * 1000:.0f} mm of any object in view. Use a position from "
            "'objects' in the latest state, or call perceive if the object is not in view")

    return args


def _queries(args: dict, max_queries: int) -> list[str]:
    """It takes perceive's arguments and gives you the cleaned query list, or
    raises PlanRejected."""
    raw = args.get("queries")
    if not isinstance(raw, list):
        raise PlanRejected(f"perceive.queries must be a list of noun phrases, got {raw!r}")
    queries: list[str] = []
    for item in raw:
        text = " ".join(str(item).lower().split()).strip(" .,;:'\"")
        if text and text not in queries:
            queries.append(text)
    if not queries:
        raise PlanRejected("perceive needs at least one noun phrase")
    # Refused, not cut: a silent cut drops the words the model ranked last, and
    # those can be the destination.
    if len(queries) > max_queries:
        raise PlanRejected(f"perceive got {len(queries)} queries; at most {max_queries}. "
                           "More slow OWLv2 and add false detections")
    return queries


def _counts(raw, queries: list[str], cap: int) -> dict[str, int]:
    """
    It takes perceive's `max_per_query` and the cleaned queries and gives you a
    count for every query, or raises PlanRejected.

    Per phrase, not one number for the call: "two bottles and a cup" wants two
    bottle boxes and one cup box, and a shared 2 would let a false second cup in.
    """
    if not isinstance(raw, dict):
        raise PlanRejected("perceive.max_per_query must map a phrase to a count, "
                           f'for example {{"bottle": 2}}; got {raw!r}')
    counts = dict.fromkeys(queries, 1)
    for phrase, k in raw.items():
        name = " ".join(str(phrase).lower().split()).strip(" .,;:'\"")
        if name not in counts:
            raise PlanRejected(f"perceive.max_per_query names {phrase!r}, which is not "
                               f"one of the queries {queries}")
        # bool is an int in Python, and True would quietly mean 1.
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= cap:
            raise PlanRejected(f"perceive.max_per_query[{phrase!r}] must be a whole number "
                               f"from 1 to {cap}, got {k!r}")
        counts[name] = k
    return counts


class Planner:
    """
    It takes the run's state and gives you the next validated step, one turn at
    a time.

    Stateful across a run, because clarification and recovery require it: the
    question, the user's answer, and every result so far stay in the
    conversation. That conversation is the run's only memory. It is bounded by
    `max_turns`, so it stays small (ADR 0015) and is never summarised.
    """

    def __init__(self, cfg: dict | None = None, envelope: WorkspaceEnvelope | None = None,
                 client=None, state: WorldState | None = None):
        cfg = cfg or get_config()
        a = cfg["architecture_a"]
        p = a["planner"]
        self.model = p["model"]
        self.max_tokens = int(p["max_tokens"])
        self.max_turns = int(p["max_turns"])
        self.max_queries = int(a["grounding"]["max_queries"])
        self.max_per_query = int(a["detector"]["max_per_query"])
        self.env = envelope or WorkspaceEnvelope(cfg)
        # None skips the in-view check, for tests of the loop alone. The pipeline
        # always passes the run's state.
        self.state = state
        self.messages: list[dict] = []
        self.usage = new_usage()
        # None leaves the cost out rather than reporting 0, which would read as free.
        self.prices = p.get("price_usd_per_mtok")
        self._client = client
        # Model calls made for the current instruction. Retries after a rejection
        # count too, so the limit bounds the whole exchange, not only good replies.
        self.turns = 0
        self._perceived = False

    @property
    def client(self):
        """It takes nothing and gives you an Anthropic client, created once and reused."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError("anthropic is not installed. `uv add anthropic`") from e
            self._client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        return self._client

    def start(self, instruction: str) -> PlannerReply:
        """
        It takes the instruction and gives you the first reply.

        The instruction is all the model gets, exactly as Architecture B gets it.
        It has seen nothing yet, so its first call is a look.
        """
        self.messages = [{"role": "user", "content": f"Instruction: {instruction}"}]
        self.turns = 0
        self._perceived = False
        return self._turn()

    def report(self, tool_use_id: str, text: str, state: dict | None = None) -> PlannerReply:
        """
        It takes the result of the step just run, and the state after it, and
        gives you the next reply.

        Feeding real outcomes back is what lets the model recover. A skill that
        reports "controller refused pose z=-0.02 below min 0.005" gives it enough
        to retry sensibly; a bare "failed" does not.

        The state goes INTO the result, not after it as a second one: a second
        result with the same id is refused outright (found on the first real run,
        1 Oct 2026).
        """
        if state is not None:
            text += f"\nState now:\n{json.dumps(state, indent=2)}"
        self.messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text}],
        })
        return self._turn()

    def answer_question(self, question_id: str, answer: str) -> PlannerReply:
        """It takes the user's answer and gives you the next reply."""
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": question_id, "content": answer}]})
        return self._turn()

    def _turn(self) -> PlannerReply:
        """
        It takes the running conversation and gives you one validated reply.

        A reply with a rejected call goes back to the model as an error and the
        model is asked again, inside the same turn limit. When no turn is left it
        raises PlanRejected with the last reason, so a plan that was never
        accepted stays distinguishable from a run that simply ran out of turns
        (`gave_up`).

        Rejections go back to the model rather than aborting the run, because a
        model that is told "z=-0.02 is below the table" usually fixes it, and a
        recovered run is a more interesting result than a crashed one (ADR 0010).
        """
        last_rejection = ""
        while True:
            if self.turns >= self.max_turns:
                if last_rejection:
                    raise PlanRejected(last_rejection)
                return PlannerReply(gave_up=True)
            self.turns += 1

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT.format(workspace=self.env.describe()),
                tools=TOOLS,
                tool_choice=TOOL_CHOICE,
                messages=self.messages,
                extra_body=MODEL_OPTIONS,
            )
            add_usage(self.usage, response)
            # Appended whole and never edited: Sonnet 5.5 refuses thinking blocks
            # from a history that was changed after they were written.
            self.messages.append({"role": "assistant", "content": response.content})

            reply, rejected = self._read(response.content)
            if not rejected:
                if reply.step is not None and reply.step.tool == "perceive":
                    self._perceived = True
                return reply
            last_rejection = next(iter(rejected.values()))
            self._send_rejection(response.content, rejected)

    def _read(self, content) -> tuple[PlannerReply, dict[str, str]]:
        """
        It takes the model's content blocks and gives you the reply and a
        {tool_use_id: reason} map of the calls that failed validation.
        """
        reply = PlannerReply()
        calls = [b for b in content if b.type == "tool_use"]
        reply.text = "".join(b.text for b in content if b.type == "text")
        if len(calls) > 1:
            reason = (f"one tool call per turn, and this reply has {len(calls)}. "
                      "Send only the next call; you will see its result first")
            return reply, {b.id: reason for b in calls}
        if not calls:
            return reply, {}

        block = calls[0]
        if not self._perceived and block.name != "perceive":
            return reply, {block.id: "the first call must be perceive: nothing has been "
                                     "seen yet, so there is nothing to act on"}
        try:
            args = validate_call(block.name, dict(block.input), self.env,
                                 max_queries=self.max_queries,
                                 max_per_query=self.max_per_query, state=self.state)
        except PlanRejected as e:
            return reply, {block.id: str(e)}
        if block.name == "ask_user":
            reply.question = args["question"]
            reply.options = list(args.get("options", []))
            reply.question_id = block.id
        else:
            reply.step = Step(tool=block.name, args=args, tool_use_id=block.id)
        return reply, {}

    def _send_rejection(self, content, rejected: dict[str, str]) -> None:
        """
        It takes the rejected reply and gives you the error results appended to
        the conversation, one per tool_use, because the API wants exactly one.
        """
        results = [{"type": "tool_result", "tool_use_id": b.id,
                    "content": f"REJECTED: {rejected[b.id]}. Correct this and try again.",
                    "is_error": True}
                   for b in content if b.type == "tool_use"]
        self.messages.append({"role": "user", "content": results})
