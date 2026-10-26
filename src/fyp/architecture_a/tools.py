"""tools.py

It takes nothing and gives you the JSON schemas that tell the LLM what the robot
can do.

These schemas must match the signatures in `skills.py` exactly. When they drift,
the model emits an argument the skill does not accept and the run fails at
dispatch with a TypeError, which reads like a bug in the planner and is not.
`validate_schemas_against_skills()` at the bottom checks the match, and the
pipeline runs it at startup so drift fails loudly at second zero rather than
mid-trial.

`ask_user` is here alongside the motion skills, and that placement is the whole
Tier 2 experiment. Asking a clarifying question is, to this architecture, just
another tool call. Architecture B has no equivalent: SmolVLA emits TCP pose
deltas and has no channel through which a question could travel. The asymmetry is a
result, not something to paper over.

`perceive` is a tool for the same reason (ADR 0012). The planner chooses what to
look for and when, so a clarification answer can become new detector words.
"""
from __future__ import annotations

import inspect

_XYZ = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
    "description": "Position [x, y, z] in metres, robot base frame.",
}

TOOLS: list[dict] = [
    {
        "name": "perceive",
        # ADR 0004's rules live here now, because choosing these words IS the
        # grounding step (ADR 0012). OWLv2 matches each phrase as one embedding,
        # so a sentence finds nothing.
        "description": (
            "Look at the table for the objects you name, and get back the state: what is "
            "in the gripper, each object in view with an id and a base-frame position, "
            "what is no longer in view, and what is seen but out of reach. Your first "
            "call must be this one. Call it again whenever you need different objects, "
            "for example after the user answers a question. Name objects as short noun "
            "phrases, lower case, no articles or verbs. Keep every distinguishing word "
            "the user said ('red cube', not 'cube') and add none they did not say. "
            "Include the destination as well as the object to move, and name a drawer "
            "by its handle ('drawer handle'). If the user names no specific object, name "
            "the categories the task implies ('a drink': 'bottle', 'glass', 'cup')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Short noun phrases for the object detector.",
                },
                "max_per_query": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "How many objects each phrase may find. Default 1, the best match "
                        "only, which is right when there is one of each thing. Raise it when "
                        "several objects may share a name and you need to see them all, for "
                        "example two bottles. Each extra one is also a chance of a false "
                        "detection, so keep it small."
                    ),
                },
            },
            "required": ["queries"],
        },
    },
    {
        "name": "pick",
        "description": (
            "Close the gripper on the object at a position and lift it clear. "
            "Approaches from directly above. Use only when the gripper is empty."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": _XYZ,
                "approach_height": {
                    "type": "number",
                    "description": "Metres above the object to approach from. Raise it for tall objects.",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "place",
        "description": (
            "Release the currently held object at a position. "
            "Use only after a successful pick."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": _XYZ,
                "approach_height": {"type": "number", "description": "Metres above the drop point to approach from."},
                "release_clearance": {
                    "type": "number",
                    "description": "Metres above the target to open the gripper. Raise it when placing into a container.",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "pour",
        "description": (
            "Pick up the container at source_xyz, tilt it over target_xyz to pour, "
            "then return it to where it came from. Handles the whole sequence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_xyz": {**_XYZ, "description": "Position of the container to pour FROM."},
                "target_xyz": {**_XYZ, "description": "Position of the receptacle to pour INTO."},
                "tilt_rad": {"type": "number", "description": "Peak tilt angle in radians. About 1.9 empties a bottle."},
                "hold_s": {"type": "number", "description": "Seconds to hold at peak tilt. Longer pours more."},
            },
            "required": ["source_xyz", "target_xyz"],
        },
    },
    {
        "name": "open_drawer",
        "description": (
            "Grasp a drawer handle and pull it open. Use this whenever a drawer must be "
            "open and you cannot see that it already is."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "handle_xyz": {**_XYZ, "description": "Position of the drawer handle."},
                "pull_distance": {"type": "number", "description": "Metres to pull. Default 0.20."},
                "pull_axis": {
                    "type": "string",
                    "enum": ["+x", "-x", "+y", "-y"],
                    "description": "Base-frame direction the drawer travels when opening.",
                },
            },
            "required": ["handle_xyz"],
        },
    },
    {
        "name": "home",
        "description": "Return the arm to its home joint configuration. Use to finish a task cleanly.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_user",
        "description": (
            "Ask the user a clarifying question and wait for their answer. "
            "Use this when the instruction does not determine which object to act on, "
            "for example when the user asks for 'a drink' and both juice and water are "
            "present. Do NOT guess in that situation. Do not use this for questions you "
            "could answer yourself from the state, or by calling perceive with other words."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question, phrased for a person to answer out loud."},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The choices you are deciding between, taken from the objects in view.",
                },
            },
            "required": ["question"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOLS}
# perceive and ask_user are handled by the pipeline itself, not by Skills.
NON_MOTION_TOOLS = {"perceive", "ask_user"}
MOTION_TOOLS = TOOL_NAMES - NON_MOTION_TOOLS


def validate_schemas_against_skills(skills_obj) -> list[str]:
    """
    It takes a Skills instance and gives you a list of mismatches between these
    schemas and the actual method signatures. Empty list means they agree.

    Checks the two directions that actually break things: a schema advertising a
    parameter the method will not accept, and a schema marking something required
    that the method has no room for. It does not require the method to expose
    every argument the schema omits, because defaults are allowed to stay hidden
    from the model.
    """
    problems: list[str] = []
    for tool in TOOLS:
        name = tool["name"]
        if name in NON_MOTION_TOOLS:
            continue
        method = getattr(skills_obj, name, None)
        if method is None:
            problems.append(f"{name}: no matching method on Skills")
            continue
        params = set(inspect.signature(method).parameters)
        advertised = set(tool["input_schema"].get("properties", {}))
        unknown = advertised - params
        if unknown:
            problems.append(f"{name}: schema advertises {sorted(unknown)}, not in the signature")
        for req in tool["input_schema"].get("required", []):
            if req not in params:
                problems.append(f"{name}: schema requires {req!r}, not in the signature")
    return problems
