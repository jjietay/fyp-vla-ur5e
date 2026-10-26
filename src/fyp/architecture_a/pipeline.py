"""pipeline.py

It takes an instruction and gives you the arm carrying it out, with a
per-stage record of everything that happened on the way.

    instruction -> plan -> one call -> result + state -> plan -> ... -> done
                            |
                            +-- perceive(queries): capture, detect, depth, transform
                            +-- a skill, then a look that checks it
                            +-- ask_user

One tool call per turn (ADR 0012). The planner chooses what to look for, sees
what each step did and the table after it, and only then chooses the next step.

`run` takes a string and never learns how it arrived. Speech is a front end that
hands over a transcript, so nothing in here changes when it lands, and the
evaluation harness drives the same entry point from a fixed utterance list.

Architecture A gets the same string Architecture B receives, and nothing else.
The detector vocabulary is the planner's own choice of words in `perceive`, so
neither architecture starts with a list of what is on the table.

The loop matters. A single forward pass would be an open-loop pipeline that
cannot notice a failed grasp or a drawer that closed behind it, and Tier 3 exists
precisely to expose that. Every skill is followed by a look in code, not on the
model's request, so it cannot be skipped and costs no planning turn. Its result
and the state it gives go back to the planner.

Every stage is wrapped in a trace context. When something fails, the trace names
the stage, which is the evidence for the claim that Architecture A's failures are
attributable and Architecture B's are not.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from fyp.architecture_a.calibration.store import CalibrationStore
from fyp.architecture_a.perception.scene import Perception, Seen
from fyp.architecture_a.planner import PlanRejected, Planner
from fyp.architecture_a.skills import Skills
from fyp.architecture_a.state import WorldState
from fyp.architecture_a.tools import validate_schemas_against_skills
from fyp.architecture_a.trace import Event, Stage, Trace
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.helpers.config import get_config


@dataclass
class Outcome:
    """The result of one run, in the shape the evaluation harness wants."""
    ok: bool
    summary: str
    run_id: str
    failed_stage: str | None = None
    # Every noun phrase the planner looked for during the run, first use first.
    queries: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    # Every call that ran, perceive included, in order.
    executed: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    # One row per executed skill when an operator judged it: {step, tool, a_ok,
    # operator_ok}. Never shown to the planner.
    judged: list[dict] = field(default_factory=list)
    # Seconds spent waiting for the operator's typed verdicts. Not A's time: the
    # harness takes it off the trial duration.
    operator_s: float = 0.0


class ArchitectureA:
    """
    It takes a camera, a controller and a way to ask the user, and gives you
    something that executes spoken instructions.

    `judge(step) -> bool` is the operator's verdict on each executed skill, typed
    at the bench. Optional: None asks nothing. Its answer is RECORDED, in the
    trace and on the Outcome, and never reaches the planner, because an operator
    who tells A "that pick failed" is a perfect sensor B does not have.

    `ask(question, options) -> str` is how a clarifying question gets answered.
    It is required and has no default: user input is speech only (13 Oct 2026),
    so callers pass `SpeechToText.answer`, and a typed fallback cannot creep back
    in by someone forgetting the argument.

    Constructed once and reused. The detector loads an OWLv2 checkpoint on first
    use, and reloading it per instruction would dominate the latency numbers that
    go into the comparison.
    """

    def __init__(self, camera, controller, cfg: dict | None = None, *, ask,
                 calibration=None, planner_client=None, judge=None):
        self.cfg = cfg or get_config()
        a = self.cfg["architecture_a"]
        self.camera = camera
        self.controller = controller
        self.ask = ask
        self.judge = judge
        # None builds the Anthropic client on first use. Tests pass a scripted one.
        self.planner_client = planner_client
        self.depth_frames = int(a["perception"]["depth_frames"])

        v = a["verify"]
        self.verify = bool(v["enabled"])
        self.verify_tol = float(v["tolerance_mm"]) / 1000.0
        self.verify_settle_s = float(v["settle_s"])
        # The planner's latest perceive arguments. The look after each skill uses
        # them, so it sees what the planner last asked to see.
        self._queries: list[str] = []
        self._per_query = 1

        self.envelope = WorkspaceEnvelope(self.cfg)
        self.skills = Skills(controller, self.cfg, self.envelope)

        problems = validate_schemas_against_skills(self.skills)
        if problems:
            raise RuntimeError("tool schemas do not match Skills:\n  " + "\n  ".join(problems))

        # `calibration` is a 4x4 T_base_cam or a CalibrationStore. None reads the
        # file the calibration scripts write, which is the real-run behaviour.
        if calibration is None:
            calibration = CalibrationStore(self.cfg)
        if isinstance(calibration, CalibrationStore):
            calibration = calibration.current()[0]
        self.T_base_cam = calibration
        self.perception = Perception(self.cfg, self.T_base_cam, self.envelope)

    # ------------------------------------------------------------ perception

    def perceive(self, queries: list[str], per_query: int, trace: Trace,
                 state: WorldState, turn: int) -> str:
        """
        It takes the planner's queries and how many boxes each may keep, and
        gives you the result text for it, having brought `state` up to date with
        what the camera sees.

        The planner's choice of words is recorded as GROUNDING, so a wrong noun
        phrase stays attributable (H9) and is never blamed on the detector.
        Capture, detect, depth and transform are their own trace stages. A look
        that finds nothing usable goes back to the planner as FAILED, so it can
        try other words, and the stage that found nothing is still recorded.
        """
        trace.record(Event(Stage.GROUNDING, True, f"planner chose {queries}",
                           data={"queries": queries, "max_per_query": per_query}))
        self._queries, self._per_query = queries, per_query
        with trace.stage(Stage.CAPTURE) as ev:
            # Depth averaged over a few frames while the arm is at rest; see camera.capture.
            frame = self.camera.capture(depth_frames=self.depth_frames)
            ev.data["resolution"] = [frame.intrinsics.width, frame.intrinsics.height]

        try:
            # Detection, depth and transform are opened inside `see`, one trace
            # stage each, so a failure is still blamed on the stage that caused it.
            seen = self.perception.see(frame, queries, trace, per_query=per_query)
        except RuntimeError as e:
            state.nothing_seen(turn, queries)
            trace.look(queries, {q: [] for q in queries}, [])
            return f"FAILED: {e}"

        state.update(seen.in_reach, seen.out_of_reach, queries, turn, gripper=self._tcp())
        found = {q: sorted((o["confidence"] for o in seen.in_reach + seen.out_of_reach
                            if o["name"] == q), reverse=True) for q in queries}
        trace.look(queries, found, [o["name"] for o in seen.out_of_reach])
        missing = [q for q, scores in found.items() if not scores]
        text = f"OK: looked for {queries}"
        if per_query > 1:
            text += f", up to {per_query} each; found " + ", ".join(
                f"{q} x{len(s)}" for q, s in found.items() if s)
        if missing:
            text += f"; nothing found for {missing}"
        return text

    def _look(self) -> tuple[Seen, np.ndarray]:
        """
        It takes nothing and gives you what the camera sees now, and the TCP.

        No trace and not strict. This is the look after a skill: an object the
        gripper hides is expected, and must not be blamed on the detection stage.
        """
        time.sleep(self.verify_settle_s)
        frame = self.camera.capture(depth_frames=self.depth_frames)
        seen = self.perception.see(frame, self._queries, None, strict=False,
                                   per_query=self._per_query)
        return seen, self._tcp()

    def _tcp(self) -> np.ndarray:
        return np.asarray(self.controller.get_state()["tcp_pose"][:3], dtype=float)

    # -------------------------------------------------------------- dispatch

    def act(self, step, trace: Trace, state: WorldState, turn: int,
            outcome: Outcome | None = None) -> str:
        """
        It takes one validated skill call and gives you its result text, having
        run it, looked at the table after it, and updated `state`.

        The look runs after every skill, failed or not, so the planner always
        plans on the table as it is now. After a pick or a place it is also the
        check (see `_check`).
        """
        with trace.stage(Stage.EXECUTION, f"{step.tool}({step.args})") as ev:
            # Named before the motion, while the object is still where the planner saw it.
            target = state.target(step.args["xyz"]) if step.tool == "pick" else None
            result = getattr(self.skills, step.tool)(**step.args)
            try:
                seen, tcp = self._look()
            except Exception as e:
                seen, tcp = None, None
                ev.data["look_error"] = str(e)

            if seen is None:
                if result.ok and step.tool == "pick" and target is not None:
                    state.grab(target.id)
                elif result.ok and step.tool == "place":
                    state.release(None, turn)
                state.nothing_seen(turn, self._queries)
                result.message += "; the look after the step failed, so nothing is in view"
            else:
                if result.ok and step.tool in ("pick", "place"):
                    self._check(step, result, target, seen, tcp, state, turn, ev)
                elif result.ok:
                    result.message += f"; unverified: no check exists for {step.tool}"
                state.update(seen.in_reach, seen.out_of_reach, self._queries, turn, gripper=tcp)

            ev.ok = result.ok
            ev.message = f"{step.tool}: {result.message}"
            ev.data["waypoints"] = len(result.waypoints)
            ev.data["state"] = state.render()
        # After the stage has closed, so the wait is not A's execution time.
        if self.judge is not None:
            self._ask_operator(step, result, trace, outcome)
        return ("OK: " if result.ok else "FAILED: ") + result.message

    def _ask_operator(self, step, result, trace: Trace, outcome: Outcome | None) -> None:
        """
        It takes an executed step and gives you the operator's verdict on it,
        recorded beside A's own.

        The question names the step and nothing A concluded about it. Showing
        "verified: orange at the gripper" first would push the operator towards
        yes, and the two verdicts would stop being independent.
        """
        text = f"{step.tool} {step.args}"
        t0 = time.perf_counter()
        operator_ok = bool(self.judge(text))
        wait = time.perf_counter() - t0
        trace.verdict(text, result.ok, operator_ok, wait)
        if outcome is not None:
            outcome.judged.append({"step": text, "tool": step.tool,
                                   "a_ok": result.ok, "operator_ok": operator_ok})
            outcome.operator_s += wait

    def _check(self, step, result, target, seen: Seen, tcp: np.ndarray,
               state: WorldState, turn: int, ev) -> None:
        """
        It takes a pick or place the skill reported as done and gives you its
        result corrected by what the camera sees: a pick fails when the object is
        not at the gripper, a place fails when it still is. It also moves the
        object into or out of the gripper in `state`.

        The skill can only read the finger width, and on 13 Oct 2026 the width
        said "held" while the orange stayed on the table. The camera is the
        independent witness. A pick checks the object the planner aimed at, by
        its ID in the state, so with two objects of the same name it checks the
        right one; a place checks the object in the gripper.

        Not seeing the object is "unverified", not a failure: the gripper can hide
        it from the fixed camera, and failing every hidden pick would hide the
        picks that worked. With checking off in config, the width is believed.

        Uses the RAW located points. A held object can be above the envelope,
        which `in_reach` would drop.
        """
        picking = step.tool == "pick"
        held = state.objects.get(state.held) if state.held else None
        obj = target if picking else held
        if obj is None:
            ev.data["verify"] = f"unverified: no known object for this {step.tool}"
            result.message += f"; unverified: no known object for this {step.tool}"
            return
        if not self.verify:
            _believe_width(state, obj, picking, turn)
            return

        points = [np.asarray(o.p_base, dtype=float) for o in seen.located
                  if o.valid and o.query == obj.name]
        if not points:
            ev.data["verify"] = f"unverified: {obj.id} not seen after the {step.tool}"
            result.message += f"; unverified: {obj.id} not seen after the {step.tool}"
            _believe_width(state, obj, picking, turn)
            return

        nearest = min(points, key=lambda p: np.linalg.norm(p - tcp))
        d = float(np.linalg.norm(nearest - tcp))
        at_gripper = d <= self.verify_tol
        ev.data["verify"] = {"object": obj.id, "xyz": np.round(nearest, 4).tolist(),
                             "tcp": np.round(tcp, 4).tolist(), "distance_mm": round(d * 1000, 1),
                             "at_gripper": at_gripper}
        where = (f"seen at {np.round(nearest, 3).tolist()}, {d * 1000:.0f} mm from the TCP "
                 f"(limit {self.verify_tol * 1000:.0f} mm)")
        if picking and at_gripper:
            state.grab(obj.id)
            result.message += f"; verified: {obj.id} is at the gripper, {d * 1000:.0f} mm from the TCP"
        elif picking:
            result.ok = False
            result.message = (f"{obj.id} is not at the gripper after the pick: {where}. "
                              f"The width said held: {result.message}")
        elif not at_gripper:
            # Where it landed is the point of its name nearest the drop, not the
            # TCP: with two oranges the one nearest the TCP may be the other one.
            drop = np.asarray(step.args["xyz"], dtype=float)
            landed = min(points, key=lambda p: np.linalg.norm(p - drop))
            state.release(landed, turn)
            result.message += f"; verified: {obj.id} released, {where}"
        else:
            result.ok = False
            result.message = f"{obj.id} is still at the gripper after the place: {where}"

    # ------------------------------------------------------------------- run

    def run(self, instruction: str) -> Outcome:
        """
        It takes an instruction and gives you the Outcome, having driven the arm.

        The turn limit is a real safety property, not tidiness. A model that keeps
        proposing a slightly different failing grasp would otherwise retry until
        the API bill or the arm gives out.
        """
        trace = Trace(instruction, cfg=self.cfg)
        state = WorldState(self.verify_tol)
        planner = Planner(self.cfg, self.envelope, client=self.planner_client, state=state)
        outcome = Outcome(ok=False, summary="", run_id=trace.run_id)
        self._queries, self._per_query = [], 1

        try:
            with trace.stage(Stage.PLANNING) as ev:
                reply = planner.start(instruction)
                _note(ev, reply)

            # No bound here: every pass makes a planner call, and the planner
            # owns the turn limit. It ends the loop with `gave_up` or PlanRejected.
            while True:
                if reply.question is not None:
                    with trace.stage(Stage.CLARIFY, reply.question) as ev:
                        answer = self.ask(reply.question, reply.options)
                        ev.data["answer"] = answer
                    outcome.asked.append(reply.question)
                    with trace.stage(Stage.PLANNING) as ev:
                        reply = planner.answer_question(reply.question_id, answer)
                        _note(ev, reply)
                    continue

                if reply.is_done:
                    outcome.ok = True
                    outcome.summary = reply.text.strip() or "planner reported done"
                    break

                if reply.gave_up:
                    outcome.summary = f"gave up after {planner.max_turns} planning turns"
                    break

                step = reply.step
                if step.tool == "perceive":
                    queries = step.args["queries"]
                    outcome.queries += [q for q in queries if q not in outcome.queries]
                    text = self.perceive(queries, step.args["max_per_query"], trace,
                                         state, planner.turns)
                else:
                    text = self.act(step, trace, state, planner.turns, outcome)
                outcome.executed.append(step.tool)

                with trace.stage(Stage.PLANNING) as ev:
                    reply = planner.report(step.tool_use_id, text, state.render())
                    _note(ev, reply)

        except PlanRejected as e:
            outcome.summary = f"plan rejected: {e}"
        except Exception as e:
            outcome.summary = f"{type(e).__name__}: {e}"

        outcome.failed_stage = trace.failed_stage.value if trace.failed_stage else None
        outcome.usage = planner.usage
        trace.finish(outcome.ok, outcome.summary)
        return outcome


def _believe_width(state: WorldState, obj, picking: bool, turn: int) -> None:
    """It takes a pick or place the camera could not judge and gives you the
    state the finger width reported: held after a pick, empty after a place."""
    if picking:
        state.grab(obj.id)
    else:
        state.release(None, turn)


def _note(ev, reply) -> None:
    """It takes a PLANNING event and the reply it produced and gives you the
    model's intent and its call written into the event, so a trace reads as
    thought, call, result."""
    ev.data["intent"] = reply.text.strip()
    if reply.step is not None:
        ev.data["call"] = {"tool": reply.step.tool, "args": reply.step.args}
    elif reply.question is not None:
        ev.data["call"] = {"tool": "ask_user", "question": reply.question}
