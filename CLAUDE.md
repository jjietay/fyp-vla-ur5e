# CLAUDE.md

NTU EEE FYP A1005-261. User speaks a command, a UR5e does it. The deliverable is a
**comparison of two architectures** on the same cell, not a working robot.

Only what you need to not break things. Each fact has one home:

- Scope, schedule, **status** and open actions: `docs/fyp_plan.md` (authoritative; it wins)
- Why a design is the way it is: `docs/adr/`
- Layout and what the project is: `README.md`
- Commands for each session at the bench: `docs/commands.md`

## Rules

- **`shared/` admission:** a module goes there only if **both** architectures use it at
  runtime. Test: delete `architecture_b/` and `architecture_a/` should still run. Breaking this
  destroys the fairness argument the whole FYP rests on.
- **`evaluation/suite.py` is frozen.** It was written before either pipeline was tuned. If a
  change is unavoidable, append to its `CHANGELOG` with a date and a reason, and say so in the
  report.
- **The harness never imports `architecture_a` or `architecture_b`.** It talks to an agent
  through `run(instruction) -> outcome`. A test enforces this.
- **Never pass `queries=` to `ArchitectureA` for a recorded trial.** It skips grounding and hands
  A a list of what is on the table that B never receives. Debugging aid only.
- **`simulation/`** is a gitignored archive and not part of this project. Never read it, raise
  it, or spend a tool call on it.

## Commands

```bash
uv sync
uv run pytest tests/unit                             # no device needed, always green
uv run pytest tests                                  # adds tests/hardware, which skip without a robot
uv run python scripts/run_evaluation.py --tier 0 --layouts-only   # no robot needed

# Architecture A needs the lerobot venv (OWLv2 needs PyTorch). Speech is the ONLY input.
PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/run_architecture_a.py
PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/run_evaluation.py --architecture A --tier 0
```

**`ANTHROPIC_API_KEY` must be set**, or Architecture A cannot run a single instruction.
`uv run` for everything else, Python 3.12+. `tests/hardware/` MOVES THE ARM.

## Landmines

Each of these has already cost time.

- **A stale `.git/index.lock` blocks every git operation.** Zero bytes and more than a few
  minutes old means stale: delete it. Through a sandboxed file bridge, use
  `GIT_OPTIONAL_LOCKS=0 git status`, or the lock is left behind.
- **`pytest tests` must be GREEN.** Hardware tests carry the `integration` marker and skip when
  nothing answers on `robot.host:30004`; force them with `--integration`. 5 errors instead of 5
  skips means `tests/conftest.py` is missing. A new `tests/hardware/` file without `pytestmark`
  turns the suite red on a laptop.
- **Everything stays in base frame.** No world frame, and no table coordinates without
  converting first. See the `hand_eye.py` header.
- **Read poses from RTDE, never from the pendant.** It shows TCP z in another frame.
- **Never hardcode `parents[n]` to find the repo root.** `shared/helpers/config.py` walks up to
  `pyproject.toml`.
- **`data/` is gitignored and has no backup.** Never `rm` under `data/` without saying so first.
- **faster-whisper on the RTX 50-series** needs `compute_type="float16"`; int8 crashes (ADR 0003).
- **`detect()` loads OWLv2 once per process** (`_load` in `detector.py`). Never move loading back
  into `detect()`: it cost 10.7 s per detection, all counted as A's latency.
- **OWLv2 takes short noun phrases, never sentences** (ADR 0004). This is why
  `planner.py::extract_queries` exists.
- **`open_drawer(pull_axis=...)` is still a placeholder** and WILL misbehave on hardware.
- **The RG2 gripper is driven by URScript via `sendCustomScriptFunction`.** Pendant in Remote
  Control mode. Each call ends any open servo stream and blocks until the fingers stop (about
  1.5 s full stroke). Never use the grip-detected input; it is unreliable.
- **Every motion goes through `WorkspaceEnvelope` first**, including approach and retreat
  waypoints, and every joint move through `check_joint_move`, because UR joints turn +/-360
  degrees and the wrong equivalent angle spins the wrist a full turn. Build rotations with
  rotation matrices, never by adding angles.
- **No fiducial markers, anywhere.** `T_base_cam` is fitted each session with a ball (ADR 0007).

## Conventions

**Docstrings open with what the thing does in one plain line**, in the form
*"it takes X and gives you Y"*, then detail underneath. Never lead with mechanism.

**Comments explain why, not what.** Match the existing code: traps and reasoning, not
line-by-line behaviour.

**ADRs** (`docs/adr/NNNN-slug.md`): one decision each, a few sentences, title states the
decision. A revised decision is updated in place with a dated note.

**Keep docs thin.** Before writing a doc, check whether the fact already has a home above.
A fact written in two places drifts.

## Agent skills

### Issue tracker

Issues and specs are local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles, written as `Status:` lines in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Working with JJ

- Be concise and direct. He notices padding and dislikes it.
- **Ask before restructuring or any multi-file change.** He has his own view of the layout.
- Don't act on the backlog unasked. Report findings, let him choose.
- He is doing this solo to learn it, so explain reasoning rather than only handing over code.
- "Operation not permitted" on the mounted folder is a sandbox delete restriction, not a real
  error.

## Verify before claiming done

There is no CI. After any move or rename, run this and expect `none`:

```bash
python3 - <<'PY'
import ast, pathlib
root = pathlib.Path('.'); mods = set()
for p in (root/'src').rglob('*.py'):
    if '__pycache__' in p.parts: continue
    m = '.'.join(p.relative_to(root/'src').with_suffix('').parts)
    mods |= {m, m.removesuffix('.__init__')}
bad = [(str(p), n.lineno, m) for p in list((root/'src').rglob('*.py')) + list((root/'scripts').rglob('*.py')) + list((root/'tests').rglob('*.py'))
       if '__pycache__' not in p.parts
       for n in ast.walk(ast.parse(p.read_text()))
       for m in ([n.module] if isinstance(n, ast.ImportFrom) and n.module else
                 [a.name for a in n.names] if isinstance(n, ast.Import) else [])
       if m.split('.')[0] == 'fyp' and m not in mods]
print("BROKEN:", bad or "none")
PY
```
