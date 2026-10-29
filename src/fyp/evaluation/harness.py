"""harness.py

It takes an architecture and a tier and gives you a scored trial block, plus the
per-trial record the report is built from.

Architecture-agnostic by construction. It talks to whatever it is given through
one method, `run(instruction) -> outcome`, and never imports Architecture A or B.
That is not tidiness: if the harness could see inside one of them it would grow
an affordance for it, and the comparison would stop being like-for-like.

Every trial is SPOKEN (13 Oct 2026). The suite's utterance is shown as a script,
the operator says it, and the agent receives the transcript, never the script.
`utterance` in the record is what was meant and `transcript` is what was heard,
so a trial lost to a mishearing can be told apart from one the architecture lost.
The harness takes the recogniser as `ears` rather than importing speech itself,
which keeps it agnostic about how the transcript was produced.

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
    # Time. latency_s runs from the end of speech recognition to the moment the
    # agent stops, by finishing or by failing. operator_s (typed step verdicts)
    # and clarify_s (the user answering a question aloud) are people's time
    # inside that span; duration_s above is latency_s minus both.
    latency_s: float = 0.0
    operator_s: float = 0.0
    clarify_s: float = 0.0
    # Money. API calls and tokens for the trial, and their list-price cost. Zero
    # for an agent that calls no API, which is a result, not a gap.
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    api_cost_usd: float = 0.0
    # Architecture B's control loop: action chunks produced and how long each
    # took to infer. Empty for an agent that does not report them.
    chunks: int = 0
    chunk_ms_mean: float = 0.0
    chunk_ms_max: float = 0.0
    # The agent's own last word, recorded after the verdict and never shown
    # before it. Says why a run stopped, e.g. "gave up after 6 planning turns".
    agent_said: str = ""


class Harness:
    """
    It takes an agent and gives you a completed trial block on disk.

    `agent` needs one method, `run(instruction)`, returning an object with `ok`,
    `failed_stage`, `asked`, and `run_id`. Architecture A's Outcome satisfies
    this already; Architecture B's wrapper must be written to match rather than
    the harness being bent to fit it.

    Optional fields, read when present and recorded as zero otherwise:
    `operator_s` and `clarify_s` (seconds of people's time inside `run`),
    `usage` (a dict with `calls`, the four token counts and `cost_usd`), and
    `timing` (a dict with `chunk_ms`, the inference time of each action chunk).
    `run` must return when the attempt ends, finished or failed, because the
    clock stops there.
    """

    def __init__(self, agent, architecture: str, ears, cfg: dict | None = None,
                 out_dir: str | Path | None = None, n_trials: int = 20, operator=input):
        self.cfg = cfg or get_config()
        self.agent = agent
        # Anything with hear(prompt) -> Transcript; in practice SpeechToText.
        self.ears = ears
        # How the operator's typed answers are read: input() at the bench, a
        # script in tests.
        self.operator = operator
        self.architecture = architecture
        self.n_trials = n_trials
        base = Path(out_dir) if out_dir else resolve("data/evaluation")
        base.mkdir(parents=True, exist_ok=True)
        self.out_dir = base
        self.records: list[TrialRecord] = []

    def run_tier(self, tier: Tier, layouts: list[Layout] | None = None) -> list[TrialRecord]:
        """
        It takes a tier and gives you its trial records, prompting you between
        trials to reset the cell and score the outcome.

        Success is ALWAYS the operator's verdict. There is no automatic scoring:
        an agent's `ok` means only that it stopped, and 13 Oct 2026 showed A
        saying "picked up" with the orange still on the table.

        Writes each record to disk immediately. A crash at trial 18 must not cost
        the 17 trials already run, because re-running them is lab time.
        """
        task = TASKS[tier]
        layouts = layouts or build_layouts(tier, self.n_trials)
        path = self.out_dir / f"{self.architecture}_tier{int(tier)}.csv"
        steps_path = self.out_dir / f"{self.architecture}_tier{int(tier)}_steps.csv"
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
            if self.operator("  set up, then Enter (or 's' to skip) > ").strip().lower() == "s":
                continue

            # layout.describe() above already printed the line to say.
            try:
                heard = self.ears.hear("Enter, then say it")
            except (EOFError, KeyboardInterrupt):
                print("\n  stopped before this trial ran; nothing recorded for it")
                break

            # The clock starts after the recogniser, so duration_s is the
            # architecture's time alone. ASR latency is in the speech log.
            t0 = time.perf_counter()
            outcome = self.agent.run(heard.text)
            latency = time.perf_counter() - t0
            # An agent that stops for the operator's per-step verdicts, or for the
            # user's spoken answer to a question, reports the wait. That is
            # people's time, not the architecture's.
            operator_s = float(getattr(outcome, "operator_s", 0.0))
            clarify_s = float(getattr(outcome, "clarify_s", 0.0))
            duration = latency - operator_s - clarify_s
            usage = getattr(outcome, "usage", None) or {}
            chunk_ms = list((getattr(outcome, "timing", None) or {}).get("chunk_ms", []))

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
                transcript=heard.text,
                run_id=getattr(outcome, "run_id", ""),
                latency_s=round(latency, 2), operator_s=round(operator_s, 2),
                clarify_s=round(clarify_s, 2),
                api_calls=int(usage.get("calls", 0)),
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0)),
                cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
                api_cost_usd=round(float(usage.get("cost_usd", 0.0)), 6),
                chunks=len(chunk_ms),
                chunk_ms_mean=round(sum(chunk_ms) / len(chunk_ms), 1) if chunk_ms else 0.0,
                chunk_ms_max=round(max(chunk_ms), 1) if chunk_ms else 0.0,
            )

            # Nothing the agent concluded is printed before this question, so the
            # verdict is independent of the system it judges.
            rec.success = self.operator("  SUCCESS by the criterion above? [y/N] > ").strip().lower() == "y"
            rec.agent_said = str(getattr(outcome, "summary", ""))
            print(f"  the agent said: {rec.agent_said}")
            if tier is Tier.AMBIGUITY:
                rec.chosen_object = self.operator("  which object did it act on? > ").strip()
            iv = self.operator("  interventions (Enter for 0) > ").strip()
            rec.interventions = int(iv) if iv.isdigit() else 0
            rec.notes = self.operator("  notes (optional) > ").strip()

            records.append(rec)
            self.records.append(rec)
            _append_csv(path, rec)
            # Per-step verdicts, when the agent has steps and the operator judged
            # them. Read through getattr like every other outcome field, so an
            # agent without steps (B) writes nothing and needs no special case.
            for step in getattr(outcome, "judged", []):
                _append_row(steps_path, {"trial": rec.trial, "run_id": rec.run_id,
                                         "step": step["step"], "tool": step["tool"],
                                         "a_ok": step["a_ok"],
                                         "operator_ok": step["operator_ok"]})

        print(f"\n  wrote {len(records)} records -> {path}")
        return records


def judge_step(step: str, operator=input) -> bool:
    """
    It takes the description of one executed step and gives you the operator's
    y/n on whether it worked, asking again until the answer is one of the two.

    Passed to an agent as its `judge` by the scripts. The prompt shows the step
    only, never what the agent concluded about it.
    """
    while True:
        a = operator(f"  {step}\n    did it work? [y/n] > ").strip().lower()
        if a in ("y", "n"):
            return a == "y"


def _append_csv(path: Path, rec: TrialRecord) -> None:
    """It takes a record and gives you it appended, writing a header if the file is new."""
    _append_row(path, asdict(rec))


def _append_row(path: Path, row: dict) -> None:
    """It takes a row and gives you it appended, writing a header if the file is new."""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def load_records(out_dir: str | Path | None = None) -> list[dict]:
    """It takes the results directory and gives you every trial ever recorded."""
    base = Path(out_dir) if out_dir else resolve("data/evaluation")
    rows: list[dict] = []
    # The *_steps.csv files beside them are per-step verdicts, not trials.
    for p in sorted(q for q in base.glob("*.csv") if not q.stem.endswith("_steps")):
        with p.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "    -"


def summarise(rows: list[dict]) -> str:
    """
    It takes trial records and gives you the tables that go into the report.

    Five tables, because five different claims are being made: overall success,
    where Architecture A fails (which Architecture B cannot answer at all),
    robustness to unseen phrasing, what each costs in time and money per trial,
    and what Architecture B does when the instruction is ambiguous.
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

    out.append("\nTIME AND COST PER TRIAL (means)")
    out.append(f"  {'':<14}{'tier':>5}{'architecture s':>16}{'end to end s':>14}"
               f"{'API US$':>10}{'chunk ms':>10}")

    def mean(sel, k):
        vals = [float(r[k]) for r in sel if str(r.get(k, "")).strip() not in ("", "None")]
        return sum(vals) / len(vals) if vals else 0.0

    for a in archs:
        for t in tiers:
            sel = [r for r in rows if r["architecture"] == a and int(r["tier"]) == t]
            if sel:
                out.append(f"  {a:<14}{t:>5}{mean(sel, 'duration_s'):16.1f}"
                           f"{mean(sel, 'latency_s'):14.1f}{mean(sel, 'api_cost_usd'):10.4f}"
                           f"{mean(sel, 'chunk_ms_mean'):10.1f}")

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
