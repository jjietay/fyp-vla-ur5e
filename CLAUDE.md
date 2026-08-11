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
│   └── hardware/     ur5e_controller  <- the only path to the arm
├── architecture_a/   modular. Nothing trained.
│   ├── perception/   detector, filters, localiser, pixel_to_3d
│   └── calibration/  hand_eye  <- A only; B never forms a 3D point
└── architecture_b/   end-to-end SmolVLA. Nothing hand-authored.
    └── demos/        recorder, hdf5_store, lerobot_export

scripts/              thin CLIs only: check_detector, export_lerobot, replay_episode
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
uv run pytest tests
uv run python scripts/check_detector.py --image <png> --model google/owlv2-base-patch16-ensemble --threshold 0.3
```

`uv run` for everything. Python 3.12+. There is a separate lerobot venv at `~/lerobot` for
torch work. URSim (Universal Robots' own controller simulator) provides the
`127.0.0.1:30004` endpoint the controller tests need.

## Landmines

Each of these has already cost time. Verified as of 11 Aug 2026.

- **`pytest tests` is red by default.** All 5 `test_ur5e_controller` errors are URSim not
  running on `127.0.0.1:30004`. Not a regression. There are no other tests.
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
- **`detect()` reloads the model on every call.** Fine for `check_detector.py`, fatal for the
  W4 orchestrator, which must load once and reuse.

## Known bugs, not yet fixed (these are W1 tasks, do not fix unasked)

- `shared/helpers/config.py::get_config(path)` ignores `path` after the first call because of
  a module-level singleton, so it looks configurable and is not
- `architecture_b/demos/hdf5_store.py::episode_paths` returns `sorted(*.h5) + sorted(*.hdf5)`,
  so mixed extensions give silently wrong episode order and corrupt a dataset without erroring
- `scripts/check_detector.py` defaults to `google/owlvit-base-patch32` at threshold 0.1, but
  `detector.py` documents OWLv2 at 0.3 as the thing that actually works. Defaults are a trap.
  Its default queries are also the old sim objects, not the current task suite.

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
