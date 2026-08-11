"""harness.py

It takes an architecture and a tier and gives you a scored trial block, plus the
per-trial record the report is built from.

Architecture-agnostic by construction. It talks to whatever it is given through
one method, `run(instruction) -> outcome`, and never imports Architecture A or B.
That is not tidiness: if the harness could see inside one of them it would grow
an affordance for it, and the comparison would stop being like-for-like.

The human is in the loop on purpose. Success is judged by a person against a
written criterion, because "the cube is in the tray" is not something the robot
can be trusted to self-report, and a system grading its own homework is the
fastest way to an indefensible results table.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fyp.evaluation.suite import TASKS, Layout, Tier, build_layouts
from fyp.shared.helpers.config import get_config, resolve


@dataclass
class TrialRecord:
    """
    One trial. These columns are the results table, so adding one later means
    re-running everything to fill it in. Think before trimming.
    """
    architecture: str
    tier: int
    tier_name: str
    trial: int
    layout: int
    utterance: str
    heldout_phrasing: bool
    success: bool
    failed_stage: str | None
    duration_s: float
    asked_question: bool
    question_text: str = ""
    chosen_object: str = ""      # Tier 2: which drink it went for. The actual result.
    interventions: int = 0
    notes: str = ""
    transcript: str = ""
    run_id: str = ""
    utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Harness:
    """
    It takes an agent and gives you a completed trial block on disk.

    `agent` needs one method, `run(instruction)`, returning an object with `ok`,
    `failed_stage`, `asked`, and `run_id`. Architecture A's Outcome satisfies
    this already; Architecture B's wrapper must be written to match rather than
    the harness being bent to fit it.
    """

    def __init__(self, agent, architecture: str, cfg: dict | None = None,
                 out_dir: str | Path | None = None, n_trials: int = 20):
        self.cfg = cfg or get_config()
        self.agent = agent
        self.architecture = architecture
        self.n_trials = n_trials
        base = Path(out_dir) if out_dir else resolve("data/evaluation")
        base.mkdir(parents=True, exist_ok=True)
        self.out_dir = base
        self.records: list[TrialRecord] = []

    def run_tier(self, tier: Tier, layouts: list[Layout] | None = None,
                 interactive: bool = True) -> list[TrialRecord]:
        """
        It takes a tier and gives you its trial records, prompting you between
        trials to reset the cell and score the outcome.

        Writes each record to disk immediately. A crash at trial 18 must not cost
        the 17 trials already run, because re-running them is lab time.
        """
        task = TASKS[tier]
        layouts = layouts or build_layouts(tier, self.n_trials)
        path = self.out_dir / f"{self.architecture}_tier{int(tier)}.csv"
        records: list[TrialRecord] = []

        print(f"\n{'=' * 72}")
        print(f"  {self.architecture}  |  Tier {int(tier)}: {task.name}")
        print(f"  success: {task.success_criterion}")
        if task.perturbation:
            print(f"  PERTURBATION: {task.perturbation}")
        print(f"{'=' * 72}")

        for layout in layouts[:self.n_trials]:
            print(f"\n--- trial {layout.index + 1}/{self.n_trials} ---")
            print(layout.describe())
            if interactive and input("  set up, then Enter (or 's' to skip) > ").strip().lower() == "s":
                continue

            t0 = time.perf_counter()
            outcome = self.agent.run(layout.utterance)
            duration = time.perf_counter() - t0

            rec = TrialRecord(
                architecture=self.architecture,
                tier=int(tier), tier_name=task.name,
                trial=layout.index, layout=layout.index,
                utterance=layout.utterance, heldout_phrasing=layout.heldout,
                success=False,
                failed_stage=getattr(outcome, "failed_stage", None),
                duration_s=round(duration, 2),
                asked_question=bool(getattr(outcome, "asked", [])),
                question_text=" | ".join(getattr(outcome, "asked", [])),
                run_id=getattr(outcome, "run_id", ""),
            )

            if interactive:
                print(f"  system reported: {'ok' if getattr(outcome, 'ok', False) else 'failed'}"
                      f" ({getattr(outcome, 'summary', '')})")
                rec.success = input("  SUCCESS by the criterion above? [y/N] > ").strip().lower() == "y"
                if tier is Tier.AMBIGUITY:
                    rec.chosen_object = input("  which object did it act on? > ").strip()
                iv = input("  interventions (Enter for 0) > ").strip()
                rec.interventions = int(iv) if iv.isdigit() else 0
                rec.notes = input("  notes (optional) > ").strip()
            else:
                rec.success = bool(getattr(outcome, "ok", False))

            records.append(rec)
            self.records.append(rec)
            _append_csv(path, rec)

        print(f"\n  wrote {len(records)} records -> {path}")
        return records


def _append_csv(path: Path, rec: TrialRecord) -> None:
    """It takes a record and gives you it appended, writing a header if the file is new."""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rec)))
        if new:
            w.writeheader()
        w.writerow(asdict(rec))


def load_records(out_dir: str | Path | None = None) -> list[dict]:
    """It takes the results directory and gives you every trial ever recorded."""
    base = Path(out_dir) if out_dir else resolve("data/evaluation")
    rows: list[dict] = []
    for p in sorted(base.glob("*.csv")):
        with p.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "    -"


def summarise(rows: list[dict]) -> str:
    """
    It takes trial records and gives you the tables that go into the report.

    Four tables, because four different claims are being made: overall success,
    where Architecture A fails (which Architecture B cannot answer at all),
    robustness to unseen phrasing, and what Architecture B does when the
    instruction is ambiguous.
    """
    if not rows:
        return "no trials recorded yet"

    def truthy(r, k):
        return str(r.get(k, "")).strip().lower() in ("true", "1", "yes")

    archs = sorted({r["architecture"] for r in rows})
    tiers = sorted({int(r["tier"]) for r in rows})
    out: list[str] = []

    out.append("SUCCESS RATE BY TIER")
    out.append(f"  {'tier':<28}" + "".join(f"{a:>14}" for a in archs))
    for t in tiers:
        name = next(r["tier_name"] for r in rows if int(r["tier"]) == t)
        line = f"  {t} {name:<26}"
        for a in archs:
            sel = [r for r in rows if int(r["tier"]) == t and r["architecture"] == a]
            line += f"{_pct(sum(truthy(r, 'success') for r in sel), len(sel)):>14}"
        out.append(line)

    out.append("\nFAILURE STAGE, ARCHITECTURE A ONLY")
    out.append("  (Architecture B has no stages; that absence is the finding)")
    stages: dict[str, int] = {}
    for r in rows:
        if r["architecture"].lower().startswith("a") and not truthy(r, "success"):
            stages[r.get("failed_stage") or "unattributed"] = \
                stages.get(r.get("failed_stage") or "unattributed", 0) + 1
    total = sum(stages.values())
    for stage, n in sorted(stages.items(), key=lambda kv: -kv[1]):
        out.append(f"  {stage:<28}{n:>4}  {_pct(n, total)}")
    if not stages:
        out.append("  no failures recorded")

    out.append("\nROBUSTNESS TO HELD-OUT PHRASING")
    out.append(f"  {'':<28}{'trained':>14}{'held out':>14}{'delta':>10}")
    for a in archs:
        sel = [r for r in rows if r["architecture"] == a]
        tr = [r for r in sel if not truthy(r, "heldout_phrasing")]
        ho = [r for r in sel if truthy(r, "heldout_phrasing")]
        rt = 100.0 * sum(truthy(r, "success") for r in tr) / len(tr) if tr else 0.0
        rh = 100.0 * sum(truthy(r, "success") for r in ho) / len(ho) if ho else 0.0
        out.append(f"  {a:<28}{rt:13.1f}%{rh:13.1f}%{rh - rt:+9.1f}")

    out.append("\nTIER 2: WHAT WAS CHOSEN WHEN THE INSTRUCTION WAS AMBIGUOUS")
    for a in archs:
        sel = [r for r in rows if int(r["tier"]) == int(Tier.AMBIGUITY) and r["architecture"] == a]
        if not sel:
            continue
        asked = sum(truthy(r, "asked_question") for r in sel)
        out.append(f"  {a}: asked the user in {asked}/{len(sel)} trials")
        choices: dict[str, int] = {}
        for r in sel:
            c = (r.get("chosen_object") or "").strip().lower()
            if c:
                choices[c] = choices.get(c, 0) + 1
        for obj, n in sorted(choices.items(), key=lambda kv: -kv[1]):
            out.append(f"      chose {obj:<24}{n:>4}/{len(sel)}  {_pct(n, len(sel))}")

    return "\n".join(out)


def export_summary(out_dir: str | Path | None = None) -> Path:
    """It takes the results directory and gives you a written summary file beside the CSVs."""
    base = Path(out_dir) if out_dir else resolve("data/evaluation")
    rows = load_records(base)
    text = summarise(rows)
    path = base / "summary.txt"
    path.write_text(text + "\n")
    (base / "summary.json").write_text(json.dumps(rows, indent=2))
    return path
