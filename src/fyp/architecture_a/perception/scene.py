"""scene.py

It takes a camera frame and the detector vocabulary and gives you what is on the
table: where each object is in base frame, which ones the arm can reach, and
which ones it cannot, with the reason.

    frame, queries -> detect -> filter -> localise -> lift to floor -> reach split

This is the one place that chains those steps and reads their config keys. The
pipeline, `place_and_look.py` and `calibrate_ball.py` all call `Perception.see`,
so a number measured by one script is the number the pipeline would act on.

## Two results, on purpose

`Seen.located` is the RAW base-frame point, straight from the localiser. The
calibration scripts read it, because calibration corrects the transform: a point
that the floor lift has already clamped, or that the reach split has already
removed, would hide the very error being measured.

`Seen.in_reach` and `Seen.out_of_reach` are what Architecture A uses at runtime,
after the floor lift and the split.

## Trace stages

`see` takes the run's `Trace` and opens DETECTION, DEPTH and TRANSFORM itself.
The caller cannot wrap `see` in one stage, because a failure in detection and a
failure in the transform are different results in the report (H6). CAPTURE stays
with the caller, which owns the camera. Without a trace the stages are no-ops.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from fyp.architecture_a.perception.detector import detect
from fyp.architecture_a.perception.filters import apply_filters, to_json_records
from fyp.architecture_a.perception.localiser import LocatedObject, localise
from fyp.architecture_a.trace import Event, Stage, Trace
from fyp.shared.hardware.safety import SafetyViolation, WorkspaceEnvelope


@dataclass
class Seen:
    """
    What one look at the table found.

    `in_reach` rows are {name, confidence, xyz}, base frame, floor-lifted, ready
    for the planner. `out_of_reach` rows are {name, confidence, reason} and carry
    no position. `located` is every localised object with its RAW base-frame
    `p_base`, valid or not, before the lift and the split.
    """
    in_reach: list[dict] = field(default_factory=list)
    out_of_reach: list[dict] = field(default_factory=list)
    located: list[LocatedObject] = field(default_factory=list)


class Perception:
    """
    It takes the config and the camera-to-base transform and gives you an object
    that turns a camera frame into a `Seen`.

    Constructed once and reused. The detector loads OWLv2 on first use (see
    `detector._load`), so this must not be rebuilt per instruction.
    """

    def __init__(self, cfg: dict, T_base_cam, envelope: WorkspaceEnvelope | None = None):
        a = cfg["architecture_a"]
        self.cfg = cfg
        self.T_base_cam = np.asarray(T_base_cam, dtype=float)
        self.envelope = envelope or WorkspaceEnvelope(cfg)
        self.det_model = a["detector"]["model"]
        self.det_threshold = float(a["detector"]["threshold"])
        self.det_nms_iou = float(a["detector"]["nms_iou"])
        self.method = a["perception"]["localiser"]
        self.max_below_floor = float(a["perception"]["max_below_floor_mm"]) / 1000.0

    def see(self, frame, queries: list[str], trace: Trace | None = None, *,
            strict: bool = True) -> Seen:
        """
        It takes a frame, the detector vocabulary and optionally the run's trace,
        and gives you a `Seen`.

        `strict` is for the pipeline: it raises RuntimeError inside the stage that
        found nothing usable (no detection, no valid depth, nothing in reach), so
        the trace blames that stage. The calibration scripts pass `strict=False`
        and read empty lists instead, because for them "not seen" is a skipped
        sample and an out-of-reach point is still a measurement.
        """
        kept = self.detect(frame, queries, trace, strict=strict)
        if not kept:
            return Seen()

        with _stage(trace, Stage.DEPTH) as ev:
            # The method (mask_ray or box_depth) comes from config; see localiser.py.
            located = localise(to_json_records(kept), frame, self.T_base_cam, self.cfg)
            valid = [o for o in located if o.valid]
            ev.data["method"] = self.method
            if self.method == "mask_ray":
                ev.data["heights"] = {o.query: [o.height_m, o.height_from, o.mask_px]
                                      for o in located}
            ev.data["valid"] = len(valid)
            ev.data["dropped"] = [o.query for o in located if not o.valid]
            ev.data["box_depth"] = [o.query for o in located
                                    if o.valid and o.depth_from == "box"]
            if not valid and strict:
                raise RuntimeError("every detection had invalid depth; check the depth stream")
        if not valid:
            return Seen(located=located)

        with _stage(trace, Stage.TRANSFORM) as ev:
            # Objects without valid depth are dropped rather than passed on with a
            # null position: a model handed a null coordinate will sometimes use it.
            objects = [{"name": o.query,
                        "confidence": round(float(o.score), 3),
                        "xyz": [round(float(v), 4) for v in o.p_base]}
                       for o in valid]
            lifted = self._lift_to_floor(objects)
            if lifted:
                ev.data["lifted_to_floor"] = lifted
            in_reach = [o for o in objects if self._reachable(o["xyz"])]
            # Out-of-reach objects are kept by NAME, with the reason and without a
            # position, so the planner can say "the box is too far" instead of "I
            # do not see a box". 13 Oct 2026: twice a box found by OWLv2 was
            # filtered here, and the planner told the user it saw no box at all.
            out_of_reach = [{"name": o["name"], "confidence": o["confidence"],
                             "reason": self._why_out(o["xyz"])}
                            for o in objects if o not in in_reach]
            ev.data["objects"] = objects
            # Traces written before 17 Oct 2026 name these two keys
            # out_of_workspace and out_of_workspace_reason. Read both.
            ev.data["out_of_reach"] = [o["name"] for o in out_of_reach]
            if out_of_reach:
                ev.data["out_of_reach_reason"] = {o["name"]: o["reason"]
                                                  for o in out_of_reach}
            if not in_reach and strict:
                raise RuntimeError(
                    "every detected object is out of reach: " + "; ".join(
                        f"{o['name']} ({o['reason']})" for o in out_of_reach)
                    + ". Move them closer to the arm, or check the calibration")

        return Seen(in_reach=in_reach, out_of_reach=out_of_reach, located=located)

    def detect(self, frame, queries: list[str], trace: Trace | None = None, *,
               strict: bool = True) -> list[tuple]:
        """
        It takes a frame and the detector vocabulary and gives you the kept
        detections as (query, score, box_xyxy), best first, with no depth yet.

        `see` runs this as its DETECTION stage. `calibrate_ball.py` calls it alone,
        because the ball fit builds its own camera-frame point and must not go
        through the transform it is trying to fit.
        """
        with _stage(trace, Stage.DETECTION) as ev:
            from PIL import Image
            raw = detect(Image.fromarray(frame.rgb), queries, self.det_model)
            kept, counts = apply_filters(raw, threshold=self.det_threshold,
                                         nms_iou=self.det_nms_iou, keep_top1=True)
            ev.data["counts"] = {"raw": counts[0], "over_threshold": counts[1],
                                 "after_nms": counts[2], "final": counts[3]}
            ev.data["found"] = [d[0] for d in kept]
            if not kept and strict:
                raise RuntimeError(
                    f"nothing detected above {self.det_threshold} for queries {queries}")
        return kept

    def _lift_to_floor(self, objects: list[dict]) -> list[dict]:
        """
        It takes object rows and gives you the list of rows it lifted, having
        moved each one's z up to the floor in place.

        Nothing on the table has its centre below the table. A point a little
        under the floor is the centre push overshooting, not a real position:
        for an open box the depth lands on its inner floor and the push goes
        through it (13 Oct 2026: a reachable box read z 0.156 against a 0.177
        floor and was reported out of reach). Anything further below than
        `max_below_floor_mm` is a bad reading and stays below, so it is refused.
        """
        floor = self.envelope.z[0]
        lifted = []
        for o in objects:
            if floor - self.max_below_floor <= o["xyz"][2] < floor:
                lifted.append({"name": o["name"], "from_z": o["xyz"][2]})
                o["xyz"] = [o["xyz"][0], o["xyz"][1], round(floor, 4)]
        return lifted

    def _reachable(self, xyz) -> bool:
        """
        It takes a seen object's base-frame point and gives you whether the arm
        can work with it: inside the horizontal sector and not under the floor.

        The z ceiling is left out on purpose. It bounds poses the arm is SENT to,
        and a seen object is not one: an orange held in the gripper after a pick
        reads above it (21 Oct 2026, z 0.383 against 0.380), and refusing it made
        the next instruction fail before the planner was asked. Every pose a skill
        commands still goes through the full envelope, ceiling included.
        """
        x, y, z = (float(v) for v in xyz)
        return self.envelope.contains([x, y, min(z, self.envelope.z[1])])

    def _why_out(self, xyz) -> str:
        """It takes a base-frame point and gives you the envelope's reason for refusing it."""
        try:
            self.envelope.check_point(xyz, "position")
        except SafetyViolation as e:
            return str(e)
        return "outside the workspace"


def _stage(trace: Trace | None, stage: Stage):
    """
    It takes an optional trace and a stage and gives you a context yielding the
    event to fill, recorded when there is a trace and thrown away when there is not.
    """
    if trace is not None:
        return trace.stage(stage)
    return nullcontext(Event(stage=stage, ok=True, message=""))
