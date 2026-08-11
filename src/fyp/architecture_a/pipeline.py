"""pipeline.py

It takes a typed instruction and gives you the arm carrying it out, with a
per-stage record of everything that happened on the way.

    instruction -> queries -> detect -> filter -> depth -> camera to base
                                                                  |
                              plan -> validate -> execute <-------+
                                ^                      |
                                +-- results fed back --+

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

from fyp.architecture_a.calibration.hand_eye import load as load_calibration
from fyp.architecture_a.perception.detector import detect
from fyp.architecture_a.perception.filters import apply_filters, to_json_records
from fyp.architecture_a.perception.localiser import locate
from fyp.architecture_a.planner import (GroundingFailed, PlanRejected, Planner,
                                        describe_objects, extract_queries)
from fyp.architecture_a.skills import Skills
from fyp.architecture_a.tools import validate_schemas_against_skills
from fyp.architecture_a.trace import Stage, Trace
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.helpers.config import get_config, resolve


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


def ask_on_terminal(question: str, options: list[str]) -> str:
    """
    It takes a clarifying question and gives you the user's typed answer.

    The default way the robot asks. Swap it for a text-to-speech prompt plus
    speech capture and the clarification loop becomes spoken, which is what the
    live demonstration wants; nothing else in the pipeline changes.
    """
    if options:
        print(f"\n  ROBOT ASKS: {question}")
        print(f"  options: {', '.join(options)}")
    else:
        print(f"\n  ROBOT ASKS: {question}")
    return input("  your answer > ").strip()


class ArchitectureA:
    """
    It takes a camera, a controller and a calibration, and gives you something
    that executes spoken or typed instructions.

    Constructed once and reused. The detector loads an OWLv2 checkpoint on first
    use, and reloading it per instruction would dominate the latency numbers that
    go into the comparison.
    """

    def __init__(self, camera, controller, queries: list[str] | None = None,
                 cfg: dict | None = None, ask=ask_on_terminal,
                 reperceive: bool = True):
        self.cfg = cfg or get_config()
        a = self.cfg["architecture_a"]
        self.camera = camera
        self.controller = controller
        self.ask = ask
        self.reperceive = reperceive

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

        self.det_model = a["detector"]["model"]
        self.det_threshold = float(a["detector"]["threshold"])
        self.det_nms_iou = float(a["detector"]["nms_iou"])
        self.max_depth = float(self.cfg["camera"]["max_depth"])
        self.depth_radius = int(self.cfg["camera"]["depth_patch_radius"])

        self.envelope = WorkspaceEnvelope(self.cfg)
        self.skills = Skills(controller, self.cfg, self.envelope)

        problems = validate_schemas_against_skills(self.skills)
        if problems:
            raise RuntimeError("tool schemas do not match Skills:\n  " + "\n  ".join(problems))

        cal_path = resolve(self.cfg["paths"]["calibration_dir"]) / "T_base_cam.json"
        if not cal_path.is_file():
            raise FileNotFoundError(
                f"no calibration at {cal_path}. Run the hand-eye procedure first; "
                "without T_base_cam the pipeline cannot convert a detection into a "
                "reachable target."
            )
        self.T_base_cam = load_calibration(cal_path)

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
        "The run failed" is not a result; "the run failed at depth on 3 of 20
        trials" is.
        """
        with trace.stage(Stage.CAPTURE) as ev:
            frame = self.camera.capture()
            ev.data["resolution"] = [frame.intrinsics.width, frame.intrinsics.height]

        with trace.stage(Stage.DETECTION) as ev:
            from PIL import Image
            raw = detect(Image.fromarray(frame.rgb), queries, self.det_model)
            kept, counts = apply_filters(raw, threshold=self.det_threshold,
                                         nms_iou=self.det_nms_iou, keep_top1=True)
            ev.data["counts"] = {"raw": counts[0], "over_threshold": counts[1],
                                 "after_nms": counts[2], "final": counts[3]}
            ev.data["found"] = [d[0] for d in kept]
            if not kept:
                raise RuntimeError(
                    f"nothing detected above {self.det_threshold} for queries {queries}")

        with trace.stage(Stage.DEPTH) as ev:
            located = locate(to_json_records(kept), frame.depth, frame.intrinsics,
                             max_depth=self.max_depth, radius=self.depth_radius)
            valid = [o for o in located if o.valid]
            ev.data["valid"] = len(valid)
            ev.data["dropped"] = [o.query for o in located if not o.valid]
            if not valid:
                raise RuntimeError("every detection had invalid depth; check the depth stream")

        with trace.stage(Stage.TRANSFORM) as ev:
            objects = describe_objects(valid, self.T_base_cam)
            in_bounds = [o for o in objects if self.envelope.contains(o["xyz"])]
            ev.data["objects"] = objects
            ev.data["out_of_workspace"] = [o["name"] for o in objects if o not in in_bounds]
            if not in_bounds:
                raise RuntimeError(
                    "every detected object is outside the workspace envelope; "
                    "the calibration or the bounds are wrong")

        return in_bounds

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
        planner = Planner(self.cfg, self.envelope)
        outcome = Outcome(ok=False, summary="", run_id=trace.run_id)

        try:
            queries = self.ground(instruction, planner, trace)
            outcome.queries = queries
            objects = self.perceive(queries, trace)

            with trace.stage(Stage.PLANNING) as ev:
                reply = planner.start(instruction, objects)
                ev.data["objects_offered"] = [o["name"] for o in objects]

            for _ in range(planner.max_turns):
                if reply.question is not None:
                    with trace.stage(Stage.CLARIFY, reply.question) as ev:
                        answer = self.ask(reply.question, reply.options)
                        ev.data["answer"] = answer
                    outcome.asked.append(reply.question)
                    reply = planner.answer_question(reply.question_id, answer)
                    continue

                if reply.is_done:
                    outcome.ok = True
                    outcome.summary = reply.final_text.strip() or "planner reported done"
                    break

                results = self.execute(reply.steps, trace)
                outcome.executed += [s.tool for s in reply.steps[:len(results)]]

                if self.reperceive:
                    try:
                        objects = self.perceive(queries, trace)
                        results.append((results[-1][0], results[-1][1] +
                                        f"\nScene now: {[o['name'] for o in objects]}"))
                    except Exception as e:
                        # Losing the scene mid-task is informative but not fatal;
                        # the planner can still finish from what it is holding.
                        results.append((results[-1][0], results[-1][1] +
                                        f"\nRe-perception failed: {e}"))

                with trace.stage(Stage.PLANNING):
                    reply = planner.report_results(results)
            else:
                outcome.summary = f"gave up after {planner.max_turns} planning turns"

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
