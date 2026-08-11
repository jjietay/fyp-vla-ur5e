"""planner.py

It takes an instruction and a list of objects the camera found, and gives you an
ordered sequence of skill calls to execute.

The model never sees pixels and never emits a pose. It sees a table of object
names with base-frame coordinates, and it selects tools by name with arguments
drawn from that table. That restriction is deliberate: it means a wrong plan is
always readable as a wrong *decision*, not as an inscrutable numeric output, and
that readability is what Architecture A is being measured on.

Every tool call is validated before it is allowed anywhere near the controller.
An LLM will occasionally invent a coordinate, invent a skill, or place an object
somewhere off the bench, and none of those should be discovered by watching the
arm move.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from fyp.architecture_a.tools import MOTION_TOOLS, TOOL_NAMES, TOOLS
from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope
from fyp.shared.helpers.config import get_config

SYSTEM_PROMPT = """You control a UR5e robot arm with a parallel gripper on a table.

You are given the objects a camera has detected, with their positions in the \
robot's base frame in metres. Carry out the user's instruction by calling the \
tools available to you.

Rules:
- Only act on positions taken from the detected objects list. Never invent \
coordinates, never estimate a position for something that was not detected.
- If the instruction names an object that is not in the list, say so instead of \
substituting a similar one.
- The gripper holds one object at a time. Always place what you are holding \
before picking something else.
- If the instruction does not determine which object to act on, call ask_user. \
Do not guess and do not pick the first plausible candidate.
- Work in the order the task requires. If something must be open before you can \
put an object inside it, open it first.
- When the task is complete, stop calling tools and reply with one short sentence \
describing what you did.

Workspace limits: {workspace}
"""


GROUNDING_PROMPT = """You turn a spoken robot command into a short list of things to \
look for with an open-vocabulary object detector.

Rules:
- Return the physical objects the command refers to, as short noun phrases.
- KEEP any distinguishing word the user actually said. "the red cube" becomes \
"red cube", not "cube", because there may be cubes of other colours on the table.
- Do NOT invent distinguishing words the user did not say. "the cube" stays "cube".
- Include the destination as well as the thing being moved. "put the snack in the \
drawer" needs both "snack" and "drawer handle", because the robot has to grasp the \
handle, not the drawer.
- If the command names no specific object, return the plausible object CATEGORIES \
the task implies. "pour me a drink" has no named object, so return things a drink \
could come from and go into: "bottle", "glass", "cup".
- Lower case, no articles, no verbs, no quantities.
- At most {max_queries} entries.

You are choosing what the robot looks at. Too narrow and it will not find the \
object; too broad and the detector returns clutter."""

GROUNDING_TOOL = {
    "name": "look_for",
    "description": "Report the objects the detector should search the scene for.",
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short noun phrases, lower case.",
            },
        },
        "required": ["queries"],
    },
}


class GroundingFailed(Exception):
    """Raised when an instruction could not be turned into detector queries."""


def extract_queries(instruction: str, client, model: str,
                    max_queries: int = 8, max_tokens: int = 512) -> list[str]:
    """
    It takes an instruction and gives you the object names to hand the detector.

    This exists so Architecture A receives exactly what Architecture B receives:
    the instruction string, and nothing else. Before it, the object vocabulary was
    supplied by a human on the command line, which was information B never got and
    which quietly weakened the generalisation result.

    It is the same model as the planner, doing the first part of planning. No new
    component is introduced, and none should be: swapping in a cheaper model here
    would make the comparison harder to describe honestly.

    Failure raises rather than falling back to a default vocabulary. A silent
    fallback would restore the very advantage this removes, and would do it
    invisibly, on exactly the trials where the model struggled.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=GROUNDING_PROMPT.format(max_queries=max_queries),
        tools=[GROUNDING_TOOL],
        tool_choice={"type": "tool", "name": "look_for"},
        messages=[{"role": "user", "content": instruction}],
    )

    raw: list = []
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "look_for":
            raw = list(block.input.get("queries", []))
            break

    queries: list[str] = []
    for item in raw:
        text = " ".join(str(item).lower().split()).strip(" .,;:'\"")
        if text and text not in queries:
            queries.append(text)

    if not queries:
        raise GroundingFailed(
            f"no detector queries could be derived from {instruction!r}. "
            "The instruction may not refer to any physical object.")
    return queries[:max_queries]


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
    What the planner produced this turn: steps to run, a question to ask, or a
    final message meaning it considers the task done.
    """
    steps: list[Step] = field(default_factory=list)
    question: str | None = None
    options: list[str] = field(default_factory=list)
    question_id: str = ""
    final_text: str = ""

    @property
    def is_done(self) -> bool:
        return not self.steps and self.question is None


def describe_objects(located, T_base_cam) -> list[dict]:
    """
    It takes camera-frame detections and the hand-eye transform, and gives you the
    object table the model is allowed to plan over.

    Objects without valid depth are dropped rather than passed through with a
    null position. A model handed a null coordinate will sometimes use it anyway.
    """
    from fyp.architecture_a.calibration.hand_eye import camera_to_base

    rows = []
    for obj in located:
        if not obj.valid:
            continue
        p_base = np.asarray(camera_to_base(obj.p_cam, T_base_cam), dtype=float).reshape(3)
        rows.append({
            "name": obj.query,
            "confidence": round(float(obj.score), 3),
            "xyz": [round(float(v), 4) for v in p_base],
        })
    return rows


def validate_call(tool: str, args: dict, envelope: WorkspaceEnvelope,
                  object_names: set[str] | None = None) -> dict:
    """
    It takes a proposed tool call and gives you the arguments back, or raises
    PlanRejected explaining exactly what was wrong.

    This runs before dispatch, every time, with no fast path. The checks are
    cheap and the thing they protect is expensive.
    """
    if tool not in TOOL_NAMES:
        raise PlanRejected(f"unknown tool {tool!r}; available: {sorted(TOOL_NAMES)}")

    if tool == "ask_user":
        if not str(args.get("question", "")).strip():
            raise PlanRejected("ask_user called with an empty question")
        return args

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

    return args


class Planner:
    """
    It takes the detected objects and gives you the next batch of skill calls.

    Stateful across a run, because clarification requires it: the question, the
    user's answer, and the results of skills already executed all have to stay in
    context or the model re-plans from scratch and repeats itself.
    """

    def __init__(self, cfg: dict | None = None, envelope: WorkspaceEnvelope | None = None,
                 client=None):
        cfg = cfg or get_config()
        p = cfg["architecture_a"]["planner"]
        self.model = p["model"]
        self.max_tokens = int(p["max_tokens"])
        self.max_turns = int(p["max_turns"])
        self.env = envelope or WorkspaceEnvelope(cfg)
        self.messages: list[dict] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        self._client = client

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

    def start(self, instruction: str, objects: list[dict]) -> PlannerReply:
        """It takes the instruction and the object table and gives you the first reply."""
        self.messages = [{
            "role": "user",
            "content": (
                f"Detected objects:\n{json.dumps(objects, indent=2)}\n\n"
                f"Instruction: {instruction}"
            ),
        }]
        return self._turn()

    def report_results(self, results: list[tuple[str, str]]) -> PlannerReply:
        """
        It takes (tool_use_id, outcome text) pairs and gives you the planner's next reply.

        Feeding real outcomes back is what lets the model recover. A skill that
        reports "controller refused pose z=-0.02 below min 0.005" gives it enough
        to retry sensibly; a bare "failed" does not.
        """
        self.messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}
                        for tid, text in results],
        })
        return self._turn()

    def answer_question(self, question_id: str, answer: str) -> PlannerReply:
        """It takes the user's answer and gives you the re-planned next step."""
        self.messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": question_id, "content": answer}],
        })
        return self._turn()

    def _turn(self) -> PlannerReply:
        """It takes the running conversation and gives you one validated reply."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT.format(workspace=self.env.describe()),
            tools=TOOLS,
            messages=self.messages,
        )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += getattr(response.usage, "input_tokens", 0)
        self.usage["output_tokens"] += getattr(response.usage, "output_tokens", 0)
        self.messages.append({"role": "assistant", "content": response.content})

        reply = PlannerReply()
        for block in response.content:
            if block.type == "text":
                reply.final_text += block.text
            elif block.type == "tool_use":
                args = validate_call(block.name, dict(block.input), self.env)
                if block.name == "ask_user":
                    reply.question = args["question"]
                    reply.options = list(args.get("options", []))
                    reply.question_id = block.id
                elif block.name in MOTION_TOOLS:
                    reply.steps.append(Step(tool=block.name, args=args, tool_use_id=block.id))
        return reply

    def reject(self, tool_use_id: str, reason: str) -> PlannerReply:
        """
        It takes a rejection reason and gives you the planner's corrected reply.

        Rejections go back to the model rather than aborting the run, because a
        model that is told "z=-0.02 is below the table" usually fixes it, and a
        recovered run is a more interesting result than a crashed one. The turn
        limit stops this looping forever.
        """
        self.messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id,
                         "content": f"REJECTED: {reason}. Correct this and try again.",
                         "is_error": True}],
        })
        return self._turn()
