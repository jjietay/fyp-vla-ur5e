# CLAUDE.md

NTU EEE FYP A1005-261. User speaks a command, a UR5e does it. The deliverable is a
**comparison of two architectures** on the same cell, not a working robot.

**`docs/fyp_plan.md` is authoritative** for scope, deliverables and schedule. This file
is only what you need to not break things. If the two disagree, the plan wins.

- Target: all work **complete** by 10 Nov 2026. Submissions stay on official NTU dates in 2027.
- Current tasks: `docs/plan_vault/Reference/Open Actions.md`
- Vault: `docs/plan_vault/` (Obsidian, 39 notes) mirrors the plan for tracking.

## Repo map

```
src/fyp/
├── shared/           CONTROLLED VARIABLE. Both architectures, at runtime.
│   ├── helpers/      rotations, transforms, config, rate_limiter
│   └── hardware/     ur5e_controller, camera, safety, intrinsics
├── architecture_a/   modular. Nothing trained.
│   ├── perception/   detector, filters, localiser, pixel_to_3d
│   ├── calibration/  hand_eye  <- A only; B never forms a 3D point
│   ├── skills.py     pick/place/pour/open_drawer, the only things A can do
│   ├── tools.py      schemas the LLM sees; must match skills.py signatures
│   ├── planner.py    LLM + plan validation + ask_user + query grounding
│   ├── pipeline.py   the orchestrator
│   └── trace.py      per-stage failure log <- the H6 evidence
├── architecture_b/   end-to-end SmolVLA. Nothing hand-authored.
│   └── demos/        recorder, hdf5_store, lerobot_export
└── evaluation/       the harness. Imports NEITHER architecture, on purpose.
    ├── suite.py      tiers, utterances, layouts, success criteria. FROZEN.
    └── harness.py    trial runner + results tables

scripts/              thin CLIs: check_detector, export_lerobot, replay_episode,
                      run_architecture_a, run_evaluation
config/               config.yaml + calibration/ (tracked on purpose)
docs/                 fyp_plan.md, plan_vault/, guidelines/ (NTU specs)
```

**`shared/` admission rule:** a module goes there only if **both** architectures use it at
runtime, not merely because it is generically useful. Pure maths used by one side stays with
that side. Test: delete `architecture_b/` and `architecture_a/` should still run.

Violating this quietly destroys the fairness argument the whole FYP rests on.

## Commands

```bash
uv sync
uv run pytest tests                                  # green without a robot

uv run python scripts/run_architecture_a.py --instruction "put the red cube in the tray"
uv run python scripts/run_evaluation.py --tier 0 --layouts-only   # no robot needed
uv run python scripts/run_evaluation.py --architecture A --tier 0
```

**`ANTHROPIC_API_KEY` must be set** or Architecture A cannot run a single instruction:
the planner and the grounding step both need it.

`uv run` for everything. Python 3.12+. There is a separate lerobot venv at `~/lerobot` for
torch work. URSim (Universal Robots' own controller simulator) provides the
`127.0.0.1:30004` endpoint the controller tests need.

## Landmines

Each of these has already cost time. Verified as of 11 Aug 2026.

- **`pytest tests` should be GREEN.** Controller tests are marked `integration` and auto-skip
  when nothing answers on `127.0.0.1:30004`. Force them with `--integration`. If you see 5
  errors instead of 5 skips, `tests/conftest.py` is missing.
- **Everything stays in base frame.** `get_state` reports the TCP in base frame and `moveL`
  expects targets in it. Do not invent a world frame or mix in coordinates measured against
  the table without converting first. See the `hand_eye.py` header.
- **Never hardcode `parents[n]` to find the repo root.** `shared/helpers/config.py` walks up
  to `pyproject.toml` instead, because a hardcoded depth silently broke during the
  architecture split and every path in `config.yaml` resolves through it.
- **`data/` is gitignored and has no backup.** Only `ep_001.h5` survives, recovered from
  history after a reformat wiped it. Deleting anything there is permanent. Never `rm` under
  `data/` without saying so first.
- **faster-whisper on the RTX 50-series** must use `compute_type="float16"`; the default int8
  crashes with `CUBLAS_STATUS_NOT_SUPPORTED` on sm_120.
- **`detect()` reloads the model on every call.** Fine for `check_detector.py`, fatal in a
  loop. `ArchitectureA` is built once and reused for exactly this reason.
- **Four config values in `architecture_a` are placeholders and WILL misbehave on hardware:**
  `workspace.{x,y,z}` bounds, `skills.down_rotvec` (measure by hand-guiding the arm and
  reading `get_state()["tcp_pose"][3:]`), `robot.home_q` (absent entirely, `home()` fails
  until added), and `open_drawer(pull_axis=...)`. Measure all four before the first run.
- **Everything in `skills.py` goes through `WorkspaceEnvelope` first**, including intermediate
  approach and retreat waypoints. Never add a motion path that skips it.
- **OWLv2 takes short noun phrases, never sentences.** It encodes each query into ONE CLIP
  embedding matched against image patches, so a sentence has no corresponding region and the
  match is meaningless. This is why `planner.py::extract_queries` exists at all. The 77-token
  CLIP limit is real but is not the binding constraint: a command fits and still fails.
  Reasoning in `architecture_a/perception/detector.py` and `docs/plan_vault/Decisions/Detector Query Format.md`.
- **Never pass `queries=` to `ArchitectureA` for a recorded trial.** It pins the detector
  vocabulary and skips grounding, which hands A a list of what is on the table that B never
  receives. Debugging aid only. `run_evaluation.py` deliberately passes nothing.

## What exists right now (11 Aug 2026)

**Written, never run against hardware:** Architecture A end to end (grounding, perception,
skills, planner, clarification, tracing), `shared/hardware/{camera,safety}.py`, and the
evaluation harness. 31 tests pass without a robot.

**Not started:** Architecture B (`architecture_b/` has only the demo-capture path from
earlier), and the speech front end (`shared/speech.py`, W2b).

**Blocked on lab access:** calibration, all demonstration capture, every trial, every number.
One decision must be made the moment access arrives: the action space. The calibration marker
question closed on 12 Aug 2026 with **no marker** (see below). See
`docs/plan_vault/Reference/Open Actions.md`.

**No fiducial markers, anywhere.** `T_base_cam` is written by hand into
`config/calibration/T_base_cam.json` from the measured camera mount geometry, not fitted.
`solve_rigid_transform` in `hand_eye.py` stays in the repo as the documented fallback (matte
ball on the gripper, still tag-free) if the measured transform proves too coarse to grasp
with. Expect a systematic 1-3 cm bias: 1 degree of mount orientation error is ~17 mm at a 1 m
standoff. The reach test in W3 is the only check on stage 4, because a hand-measured transform
produces no residual of its own. Reasoning in
`docs/plan_vault/Decisions/Calibration Marker.md`.

## Evaluation harness

`src/fyp/evaluation/suite.py` is the experimental design and it is **frozen**. It was written
before either pipeline was tuned. Changing a tier, an utterance or a success criterion after
tuning begins means the benchmark has been fitted to whichever architecture was built first.
If a change is genuinely unavoidable, append to `CHANGELOG` at the bottom of that file with a
date and a reason, and say so in the report.

The harness must never import `architecture_a` or `architecture_b`. It talks to an agent
through one method, `run(instruction) -> outcome`. A test enforces this.

## simulation/

A local archive folder that git ignores completely. **It is not part of this project.** JJ is
finished with it. Never read it, never raise it, never debug it, never spend a tool call on
it. Treat anything inside as if it does not exist.

## Conventions

**Docstrings open with what the thing does in one plain line**, in the form
*"it takes X and gives you Y"*, then detail underneath. Never lead with mechanism.

```python
def solve_rigid_transform(p_cam, p_base):
    """
    It takes matched 3D points seen in the camera and the same points measured
    in the robot base frame, and gives you the 4x4 transform T_base_cam.
    """
```

**Comments explain why, not what.** The existing code documents traps and reasoning, not
line-by-line behaviour. Match that.

**Vault notes** (`docs/plan_vault/`) follow a strict house style: no em dashes or en dashes,
bullets never end in a full stop, one concept per note, sparse wikilinks. Verify with:

```bash
find docs/plan_vault -name '*.md' -print0 | xargs -0 python3 <skill>/scripts/check_notes.py --vault docs/plan_vault
```

## Working with JJ

- Be concise and direct. He notices padding and dislikes it.
- **Ask before restructuring or any multi-file change.** He has reformatted the repo himself
  several times and has his own view of the layout.
- Don't act on the backlog unasked. Report findings, let him choose.
- He is doing this solo to learn it, so explain reasoning rather than only handing over code.
- When a shell command fails on the mounted folder with "Operation not permitted", it is a
  sandbox delete restriction, not a real error.

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
