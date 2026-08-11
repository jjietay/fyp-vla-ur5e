"""trace.py

It takes everything that happens during one run and gives you a record of which
stage did it, how long it took, and where it failed.

This is not housekeeping. Per-stage diagnosability is the entire argument for
Architecture A, and it is a stated hypothesis: A's failures are attributable to a
specific stage, B's are not attributable at all. This file produces the evidence
for that claim, so it has to be right the first time. Retrofitting it means
re-running every trial.

One JSONL file per run. Line-delimited rather than one big JSON object because a
crash mid-run still leaves a readable file, and because the evaluation harness
wants to concatenate hundreds of runs without parsing each one whole.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fyp.shared.helpers.config import get_config, resolve


class Stage(str, Enum):
    """
    The stages a run can fail at. This list IS the failure taxonomy in the report,
    so adding a member changes the results table and should be a deliberate act.

    Ordered as the pipeline executes.
    """
    CAPTURE = "capture"        # camera gave no frame, or a corrupt one
    GROUNDING = "grounding"    # instruction -> detector vocabulary. A-only failure mode:
                               # B never names anything, so it cannot fail here
    DETECTION = "detection"    # detector found nothing, or nothing above threshold
    DEPTH = "depth"            # detection had no valid depth to sample
    TRANSFORM = "transform"    # camera to base frame conversion, or out of workspace
    PLANNING = "planning"      # LLM produced no plan, or one that failed validation
    CLARIFY = "clarify"        # the run stopped to ask the user a question
    EXECUTION = "execution"    # a skill was dispatched and the arm did not comply


@dataclass
class Event:
    """One thing that happened, at one stage."""
    stage: Stage
    ok: bool
    message: str
    duration_s: float | None = None
    data: dict = field(default_factory=dict)


class Trace:
    """
    It takes a run identifier and gives you something that records events and
    writes them to disk as they happen.

    Usage is deliberately blunt, because a logging API you have to think about is
    a logging API that gets skipped under time pressure:

        with trace.stage(Stage.DETECTION) as ev:
            dets = detect(...)
            ev.data["n_detections"] = len(dets)
    """

    def __init__(self, instruction: str, run_id: str | None = None, cfg: dict | None = None):
        cfg = cfg or get_config()
        self.run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        self.instruction = instruction
        self.events: list[Event] = []
        self.started = time.perf_counter()

        out_dir = resolve(cfg["architecture_a"]["trace"]["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{self.run_id}.jsonl"
        self._write({"type": "run_start",
                     "run_id": self.run_id,
                     "instruction": instruction,
                     "utc": datetime.now(timezone.utc).isoformat()})

    def _write(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def record(self, event: Event) -> None:
        """It takes an event and gives you it appended to both memory and disk."""
        self.events.append(event)
        self._write({"type": "event", "stage": event.stage.value, "ok": event.ok,
                     "message": event.message, "duration_s": event.duration_s,
                     **({"data": event.data} if event.data else {})})

    @contextmanager
    def stage(self, stage: Stage, message: str = ""):
        """
        It takes a stage and gives you a context that times the block and records
        whether it raised.

        An exception is recorded with its stage attached and then re-raised, so
        the pipeline still controls what to do about it. Swallowing here would
        turn a failed run into a silently successful one, which is the single
        worst thing a logging layer can do to an experiment.
        """
        ev = Event(stage=stage, ok=True, message=message)
        t0 = time.perf_counter()
        try:
            yield ev
        except Exception as e:
            ev.ok = False
            ev.message = f"{type(e).__name__}: {e}"
            ev.data["traceback"] = traceback.format_exc(limit=6)
            ev.duration_s = round(time.perf_counter() - t0, 4)
            self.record(ev)
            raise
        ev.duration_s = round(time.perf_counter() - t0, 4)
        self.record(ev)

    def finish(self, ok: bool, summary: str = "") -> None:
        """It takes the run outcome and gives you a closed, complete trace file."""
        self._write({"type": "run_end", "ok": ok, "summary": summary,
                     "failed_stage": self.failed_stage.value if self.failed_stage else None,
                     "total_s": round(time.perf_counter() - self.started, 3)})

    @property
    def failed_stage(self) -> Stage | None:
        """It takes nothing and gives you the first stage that failed, or None."""
        return next((e.stage for e in self.events if not e.ok), None)

    def summary(self) -> str:
        """It takes nothing and gives you the run as one readable line per stage."""
        lines = [f"run {self.run_id}  instruction={self.instruction!r}"]
        for e in self.events:
            mark = "ok  " if e.ok else "FAIL"
            secs = f"{e.duration_s:6.3f}s" if e.duration_s is not None else "       "
            lines.append(f"  [{mark}] {e.stage.value:<9} {secs}  {e.message}")
        return "\n".join(lines)
