"""pipeline.py

It takes an instruction and gives you the arm carrying it out, with a
per-stage record of everything that happened on the way.

    instruction -> queries -> detect -> filter -> depth -> camera to base
                                                                  |
                              plan -> validate -> execute <-------+
                                ^                      |
                                +-- results fed back --+

`run` takes a string and never learns how it arrived. Speech is a front end that
hands over a transcript, so nothing in here changes when it lands, and the
evaluation harness drives the same entry point from a fixed utterance list.

The instruction reaches the detector before the camera does. Architecture A
derives its own object vocabulary from the same string Architecture B receives,
rather than being handed a list of what is on the table, so neither architecture
starts with information the other lacks.

The loop matters. A single forward pass would be an open-loop pipeline that
cannot notice a failed grasp or a drawer that closed behind it, and Tier 3 exists
precisely to expose that. Skill outcomes go back to the planner, and by default
the scene is re-perceived before each planning turn, so a plan is always made
against what is on the table now rather than what was there when the run started.

Every stage is wrapped in a trace context. When something fails, the trace names
the stage, which is the evidence for the claim that Architecture A's failures are
attributable and Architecture B's are not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fyp.architecture_a.calibration.store import CalibrationStore
from fyp.architecture_a.perception.scene import Perception
from fyp.architecture_a.planner import GroundingFailed, PlanRejected, Planner, extract_queries
from fyp.architecture_a.skills import Skills
from fyp.architecture_a.tools import validate_schemas_against_skills
from fyp.architecture_a.trace import Stage, Trace
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.helpers.config import get_config


@dataclass
class Outcome:
    """The result of one run, in the shape the evaluation harness wants."""
    ok: bool
    summary: str
    run_id: str
    failed_stage: str | None = None
    queries: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ArchitectureA:
    """
    It takes a camera, a controller and a way to ask the user, and gives you
    something that executes spoken instructions.

    `ask(question, options) -> str` is how a clarifying question gets answered.
    It is required and has no default: user input is speech only (13 Oct 2026),
    so callers pass `SpeechToText.answer`, and a typed fallback cannot creep back
    in by someone forgetting the argument.

    Constructed once and reused. The detector loads an OWLv2 checkpoint on first
    use, and reloading it per instruction would dominate the latency numbers that
    go into the comparison.
    """

    def __init__(self, camera, controller, queries: list[str] | None = None,
                 cfg: dict | None = None, *, ask, reperceive: bool = True,
                 calibration=None, planner_client=None):
        self.cfg = cfg or get_config()
        a = self.cfg["architecture_a"]
        self.camera = camera
        self.controller = controller
        self.ask = ask
        self.reperceive = reperceive
        # None builds the Anthropic client on first use. Tests pass a scripted one.
        self.planner_client = planner_client
        # Filled by perceive(): objects seen but outside the reachable workspace,
        # as {name, confidence, reason}, never with a position. See perceive().
        self.out_of_reach: list[dict] = []

        # `queries` pins the detector vocabulary and BYPASSES grounding. It is a
        # debugging aid only. Leave it None for any recorded trial: supplying it
        # hands Architecture A a list of what is on the table, which is
        # information Architecture B never receives, and the generalisation
        # result stops meaning anything.
        self.fixed_queries = list(queries) if queries else None
        self._query_cache: dict[str, list[str]] = {}

        g = a["grounding"]
        self.grounding_model = g["model"] or a["planner"]["model"]
        self.max_queries = int(g["max_queries"])
        self.grounding_max_tokens = int(g["max_tokens"])

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

    def ground(self, instruction: str, planner, trace: Trace) -> list[str]:
        """
        It takes the instruction and gives you the detector vocabulary derived
        from it.

        Its own stage because its failures are its own. If the model returns
        "cube" where the scene needed "red cube" to be found reliably, that is a
        vocabulary error, and recording it as a detection error would corrupt the
        per-stage breakdown that is the whole H6 argument.

        This is also a failure mode Architecture B structurally cannot have,
        since B never names anything. That asymmetry belongs in the results.

        Cached per instruction so re-perceiving each turn does not re-extract.
        """
        if self.fixed_queries is not None:
            with trace.stage(Stage.GROUNDING, "bypassed, queries fixed at construction") as ev:
                ev.data["queries"] = self.fixed_queries
                ev.data["bypassed"] = True
            return self.fixed_queries

        if instruction in self._query_cache:
            return self._query_cache[instruction]

        with trace.stage(Stage.GROUNDING, instruction) as ev:
            queries = extract_queries(instruction, planner.client, self.grounding_model,
                                      self.max_queries, self.grounding_max_tokens)
            ev.data["queries"] = queries
            ev.message = f"{instruction!r} -> {queries}"

        self._query_cache[instruction] = queries
        return queries

    def perceive(self, queries: list[str], trace: Trace) -> list[dict]:
        """
        It takes the detector vocabulary and gives you the objects currently on
        the table, in base frame, ready to hand to the planner.

        Capture, detect, depth and transform are separate trace stages on purpose.
        Capture is opened here; the other three are opened by `Perception.see`.
        "The run failed" is not a result; "the run failed at depth on 3 of 20
        trials" is.
        """
        with trace.stage(Stage.CAPTURE) as ev:
            # Depth averaged over a few frames while the arm is at rest; see camera.capture.
            frame = self.camera.capture(
                depth_frames=int(self.cfg["architecture_a"]["perception"]["depth_frames"]))
            ev.data["resolution"] = [frame.intrinsics.width, frame.intrinsics.height]

        # Detection, depth and transform are opened inside `see`, one trace stage
        # each, so a failure is still blamed on the stage that caused it.
        seen = self.perception.see(frame, queries, trace)
        self.out_of_reach = seen.out_of_reach
        return seen.in_reach

    # -------------------------------------------------------------- dispatch

    def execute(self, steps, trace: Trace) -> list[tuple[str, str]]:
        """
        It takes validated steps and gives you (tool_use_id, outcome) pairs to feed
        back to the planner.

        Stops at the first failure rather than pushing on. Later steps assume
        earlier ones worked, and a place after a failed pick drops nothing onto
        the table from a height.
        """
        results = []
        for step in steps:
            with trace.stage(Stage.EXECUTION, f"{step.tool}({step.args})") as ev:
                method = getattr(self.skills, step.tool)
                result = method(**step.args)
                ev.ok = result.ok
                ev.message = f"{step.tool}: {result.message}"
                ev.data["waypoints"] = len(result.waypoints)
                results.append((step.tool_use_id,
                                ("OK: " if result.ok else "FAILED: ") + result.message))
            if not result.ok:
                break
        return results

    # ------------------------------------------------------------------- run

    def run(self, instruction: str) -> Outcome:
        """
        It takes an instruction and gives you the Outcome, having driven the arm.

        The turn limit is a real safety property, not tidiness. A model that keeps
        proposing a slightly different failing grasp would otherwise retry until
        the API bill or the arm gives out.
        """
        trace = Trace(instruction, cfg=self.cfg)
        planner = Planner(self.cfg, self.envelope, client=self.planner_client)
        outcome = Outcome(ok=False, summary="", run_id=trace.run_id)

        try:
            queries = self.ground(instruction, planner, trace)
            outcome.queries = queries
            objects = self.perceive(queries, trace)

            with trace.stage(Stage.PLANNING) as ev:
                reply = planner.start(instruction, objects, self.out_of_reach)
                ev.data["objects_offered"] = [o["name"] for o in objects]

            # No bound here: every pass makes a planner call, and the planner
            # owns the turn limit. It ends the loop with `gave_up` or PlanRejected.
            while True:
                if reply.question is not None:
                    with trace.stage(Stage.CLARIFY, reply.question) as ev:
                        answer = self.ask(reply.question, reply.options)
                        ev.data["answer"] = answer
                    outcome.asked.append(reply.question)
                    with trace.stage(Stage.PLANNING):
                        reply = planner.answer_question(reply.question_id, answer)
                    continue

                if reply.is_done:
                    outcome.ok = True
                    outcome.summary = reply.final_text.strip() or "planner reported done"
                    break

                if reply.gave_up:
                    outcome.summary = f"gave up after {planner.max_turns} planning turns"
                    break

                results = self.execute(reply.steps, trace)
                outcome.executed += [s.tool for s in reply.steps[:len(results)]]

                scene, scene_error = None, None
                if self.reperceive and results:
                    try:
                        objects = self.perceive(queries, trace)
                        scene = ([o["name"] for o in objects],
                                 [o["name"] for o in self.out_of_reach])
                    except Exception as e:
                        scene_error = str(e)

                with trace.stage(Stage.PLANNING):
                    reply = planner.report_results(results, scene, scene_error)

        except GroundingFailed as e:
            outcome.summary = f"grounding failed: {e}"
        except PlanRejected as e:
            outcome.summary = f"plan rejected: {e}"
        except Exception as e:
            outcome.summary = f"{type(e).__name__}: {e}"

        outcome.failed_stage = trace.failed_stage.value if trace.failed_stage else None
        outcome.usage = planner.usage
        trace.finish(outcome.ok, outcome.summary)
        return outcome
