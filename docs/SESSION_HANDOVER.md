# Session Handover — FYP (UR5e VLA vs Modular)

_Last updated: 2026-07-27. Read this first to pick up where we left off._

Sessions are logged newest-first below. Nothing from an earlier session is
deleted, only struck through or annotated when it stops being true, so the
reasoning behind a decision stays recoverable.

## Project context (the big picture)

FYP is a **two-architecture comparison** on a real UR5e, developed in MuJoCo as a
substrate (sim is NOT a deliverable; the real comparison uses real UR5e demos).

- **Architecture A (the core FYP code):** modular pipeline —
  `frame → detector → depth-to-3D → camera→base transform → LLM planner → pick/place primitives`.
  Its selling point is being **modular and per-stage diagnosable**.
- **Architecture B (stretch goal):** end-to-end VLA (SmolVLA). Pipeline already
  proven with a 1-step fine-tune. Data collection / long fine-tunes are explicitly
  OUT of scope for the 3-week prep (`docs/FYP_Plan_V3_Week_5.pdf`).

Governing plan: `docs/FYP_Plan_V3_Week_5.pdf` (20 Jul–9 Aug 2026).

## Environments (important — two separate venvs)

- **FYP venv** (`~/Documents/NTU/Y4S1/FYP/.venv`, run via `uv run`): has **mujoco**,
  h5py, ur_rtde, numpy, PIL. NOT torch/transformers.
- **lerobot venv** (`~/lerobot`, run via `uv run` there): has **torch + transformers**
  (SmolVLA). Needs `--extra dataset --extra training` for LeRobot data/training.
  Detection runs here.
- So: **render in the FYP venv, detect/train in the lerobot venv.**

## Architecture A — stage status

| # | Stage | Status | Where |
|---|-------|--------|-------|
| 1 | frame (RGB + depth) | ✅ verified | `src/fyp/perception/camera.py::render_rgbd` |
| 2 | open-vocab detector | ✅ working, 1 caveat | `scripts/detect_objects.py` |
| 3 | depth-to-3D | ✅ verified 0.00 mm | `camera.py::pixel_to_camera`, `scripts/depth_to_3d.py` |
| 4 | camera→base (hand-eye) | ⬜ next | maths ready in `src/fyp/transforms.py` |
| 5 | LLM planner | ⬜ | — |
| 6 | pick / place primitives | 🟡 substrate only | `controller.py` + `sim/mujoco_controller.py` share an interface |

## Deliverable status (from the plan)

- **Week 1 — DONE:** recorder 20 Hz ✅, SmolVLA read + summary ✅, scene with
  objects ✅, detector returning boxes ✅. (Email to Prof Cheah deliberately skipped
  by JJ — holiday.)
- **Week 2 — in progress:** depth-to-3D ✅ (Mon 27), hand-eye (Tue 28), OXE paper
  (Wed 29), `euler_to_rotvec` + pick (Thu 30), place (Fri 31), Anthropic SDK
  (Sat 1), LLM planner (Sun 2).
- **Week 3:** close the Architecture A loop, read RTC paper, refine proposal,
  consolidate repo, handover.

---

# Session log

## Session 2 — 2026-07-27: detector NMS + depth-to-3D (stage 3 DONE)

### What changed, and why

1. **Detector de-duplication** — `scripts/detect_objects.py`.
   *Why:* session 1 flagged that at threshold 0.3 two overlapping boxes could land
   on one cube, which would give depth-to-3D a phantom 3D point.
   *What:* added class-agnostic greedy NMS (`nms`), optional per-query top-1
   (`top1_per_query`), flags `--nms-iou` / `--per-query-nms` / `--top1-per-query`,
   and `--json` output with `center_uv` precomputed so stage 3 consumes it directly.
   The overlay now also draws the centre crosshair — the exact pixel depth samples.
   *Non-obvious:* **class-agnostic is the default on purpose.** The real failure is
   OWLv2 firing *different* labels on one cube; per-query NMS and top-1 both let
   that through because different labels never compete. Tested:
   ```
   class-agnostic NMS -> [red 0.62, green 0.55]              ✅ 2
   per-query NMS      -> [red 0.62, green 0.55, yellow 0.41] ❌ 3
   top1 alone         -> [red 0.62, green 0.55, yellow 0.41] ❌ 3
   ```
   Filter order in `main` is threshold → NMS → optional top-1.

2. **`workspace` camera recentred** — `assets/mujoco/ur5e/scene_gripper.xml`.
   *Why:* a bug found while checking the frame covered the scene. At the session-1
   pose `pos "0.45 0 0.85" fovy 50` the view stopped at y = −0.396, while the bin
   spans y = −0.49..−0.31. **The bin was never in frame** — which is why session 1
   only ever saw four cubes and no bin. The planner cannot place into a bin it
   cannot see.
   *What:* recentred on the union of blocks + bin and widened slightly →
   **`pos "0.505 -0.145 0.85" fovy 55`**, leaving ~8 cm margin for the randomised
   placement planned later. Derivation is in the XML comment, not just here.

3. **New package `src/fyp/perception/`.**
   *Why:* stages 1–4 are the reusable FYP code (they survive the move to real
   hardware); scripts should stay thin. This is also what Week 3's "runnable demo
   each" consolidation wants.
   *What:* `camera.py` with `render_rgbd()` (aligned RGB + metric depth from one
   `update_scene`), `intrinsics_from_fovy()`, `intrinsics_for_camera()`,
   `pixel_to_camera()`, `camera_to_pixel()`, `surface_to_centroid()`, `depth_at()`.

4. **Verification scripts** — `scripts/render_depth.py`, `scripts/depth_to_3d.py`.
   *Why:* per-stage diagnosability is the whole argument for Architecture A, so
   each stage gets a check that fails in isolation. `depth_to_3d.py --verify`
   deliberately runs **without the detector**, so a failure means the camera model
   is wrong rather than OWLv2. Ground truth is read from the model
   (`cam_xpos`/`cam_xmat`, `xpos`, `geom_size`), not hardcoded, so the checks
   survive scene edits.

### Results (verified on JJ's machine)

```
render_depth.py --verify
  bare table @ image centre   0.8500 m  (err +0.0 mm)
  all four cube tops          0.8090 m  (err +0.0 mm)
  bin centre                  in frame at (u=316.8, v=377.8)
  convention                  z-depth (0.850 vs ray-length 1.1245 at the corner)

depth_to_3d.py --verify
  round trip (2000 px)        max error 1.14e-13
  back-projection, all 4      err (0.00, 0.00, 0.00) mm
  top-surface -> centroid     err (0.00, 0.00, 0.00) mm
```

### Facts worth keeping

- MuJoCo depth is **z-depth in metres** — perpendicular distance from the camera
  plane, not ray length, not normalised. So `depth` divides straight into the
  pinhole model with no ray-length correction. This was measured, not assumed.
- Background reads the **floor distance (1.600 m)** — a finite, plausible-looking
  number, not `inf`. It must be rejected explicitly; `depth_at(max_depth=...)` does.
- The **arm is in frame** (min depth 0.6466 m ≈ 20 cm above the table), so
  occlusion is real. `depth_at` returns NaN rather than a confidently wrong number.
- A back-projected box centre is a point on the object's **top face**, not its
  centroid. `surface_to_centroid()` does the offset that `pick()` will need.
- Principal point is `(W-1)/2`, not `W/2` — half a pixel, ~0.8 mm at 0.85 m.

### Files created
- `src/fyp/perception/__init__.py`, `src/fyp/perception/camera.py`
- `scripts/render_depth.py`, `scripts/depth_to_3d.py`

### Files edited
- `scripts/detect_objects.py` — NMS, top-1, `--json`, centre crosshair
- `assets/mujoco/ur5e/scene_gripper.xml` — `workspace` camera recentred/widened
- `docs/commands.md` — added stage 1/2/3 commands
- `docs/SESSION_HANDOVER.md` — restructured into a dated log (see below)

### Changes to this document
Session 1's content is **retained in full** below. Two things were removed because
they were completed, not because they were wrong:
- its `NEXT TASK: Depth-to-3D` section — that task is now done and is recorded above;
- the caveat _"Detector: add per-query top-1 or NMS before feeding depth-to-3D"_ —
  implemented this session (item 1).

Everything else was reorganised, not edited: session 1's log was undated and headed
"What we did this session", which stops working once there are two sessions.

---

## Session 1 — 2026-07-26: recorder fix, LeRobot pipeline, scene, detector

1. **Fixed the DemoRecorder 20 Hz cadence bug** (was recording ~4 Hz).
   - `sim_server.py`: recording is now wall-clock rate-gated (`_maybe_record`), not
     tick-counted; crash-guarded so a bad frame can't kill the sim loop.
   - `record_replay_episode.py`: offline generator now stamps **sim-time**
     (`step * control_dt`) via a global step counter (uniform 20 Hz).
   - `demo_recorder.py`: `record()` gained an optional `timestamp` arg.
   - Verified: re-recorded `ep_001.h5` = 20.02 Hz, dt 50 ms.
2. **Auto-naming for episodes:** `sim_client.py` `stop_and_save()` now auto-picks the
   next free `ep_NNN.h5` (never overwrites). Added `next_episode_path()`.
3. **Proved the LeRobot + SmolVLA pipeline** end to end: converted `ep_001.h5`
   (1236 frames, fps=20, 7-dim state/action incl. gripper) → `data/lerobot_ur5e`,
   then ran a 1-step SmolVLA fine-tune successfully.
4. **MuJoCo scene with graspable objects** (Week-1 deliverable): created
   `assets/mujoco/ur5e/props.xml` (4 coloured free-joint cubes + static bin),
   included into `scene_gripper.xml`, extended the home keyframe qpos to nq=42.
   Verified: compiles, gripper can pick a cube into the bin.
5. **Open-vocab detector** (Week-1 deliverable): `scripts/detect_objects.py`.
   Added a top-down `workspace` camera to `scene_gripper.xml`. **OWLv2**
   (`google/owlv2-base-patch16-ensemble`) at `--threshold 0.3` detects all 4 cubes
   with correct labels. (OWL-ViT base was too weak.)
   - _Amended 2026-07-27:_ the camera pose used here did not frame the bin, so
     "all 4 cubes" was the complete picture only by accident. See session 2 item 2.

### Files created this session
- `assets/mujoco/ur5e/props.xml` — cubes + bin.
- `scripts/detect_objects.py` — render + OWLv2 detector.
- `docs/SESSION_HANDOVER.md` — this file.

### Files edited
- `src/fyp/sim/sim_server.py`, `src/fyp/sim/demo_recorder.py`,
  `scripts/record_replay_episode.py`, `src/fyp/sim/sim_client.py`
- `assets/mujoco/ur5e/scene_gripper.xml` (props include, keyframe, workspace camera)

---

# NEXT TASK: Hand-eye (task #5, Tue 28 Jul)

Stage 3 outputs **camera-frame** XYZ. Stage 4 turns that into the **base frame**,
which is what the primitives consume.

1. Build the camera→base transform with `pose_to_T` / `pose_trans` (the camera
   pose is exactly known in sim: `data.cam_xpos` / `cam_xmat`).
2. Transform a detected object into the base frame.
3. Verify against MuJoCo's ground-truth body position — the same style of check as
   `depth_to_3d.py --verify`, which should be reusable.

**Watch out:** the arm base carries `quat="0 0 0 -1"` in `scene_gripper.xml`, so the
robot base frame is rotated 180° from world. Hand-eye is camera→**base**, not
camera→world. Verify that specifically rather than assuming they coincide.

Then #6 pick/place primitives (+ `euler_to_rotvec` in `transforms.py`), #7 LLM
planner (Anthropic tool-calling over primitives), #8 read OXE paper.

## Known caveats / TODO

- **The bin is not detectable as "bin".** Confirmed 2026-07-27 with the corrected
  camera: OWLv2 returns no `bin` label at any score. It *does* see the bin, but
  labels it **"blue cube" at 0.269** (box `[264.0, 323.6, 372.7, 441.9]`, centred on
  the bin). The material is `rgba 0.30 0.30 0.33`, a blue-grey, so this is not
  unreasonable. Two consequences:
  - Lowering the threshold to catch the bin injects a **phantom blue cube** at the
    bin's location. Right now `--top1-per-query` hides this (real blue cube 0.580
    beats 0.269) — but that protection vanishes the moment the scene holds two of
    anything and top-1 has to be dropped.
  - **Undecided design question, needed before the planner (Sun 2 Aug):** treat the
    bin as a *known fixture* with a calibrated constant pose (robust, and nobody
    detects their own workcell furniture), or make it detectable via better queries
    (`"grey tray"`, `"container"`, `"open box"`) or a recolour (keeps the
    "name any target by text" generalisation claim intact for the target too).
- **Spurious detections at the table edge:** `blue cube` fires at u≈589–636, which
  back-projects to world x ≈ 1.0 — the table/floor boundary. Currently suppressed by
  `--top1-per-query`; same fragility as above.
- **`depth_to_3d.py --detections` not yet run** end to end. `--verify` passes; the
  detector-in-the-loop number (how much error stage 2 adds on top of an exact
  stage 3) is still unmeasured.
- **Scene regeneration:** `scene_gripper.xml` is generated by `attach_gripper.py`.
  Re-running it will wipe the props include + keyframe + workspace camera edits.
  If regeneration is needed, make `attach_gripper.py` props-aware first.
- **Sim recorder clock:** `sim_server.py` uses wall-clock gating (right for the real
  robot). For the slow MuJoCo sim, the plan preferred sim-time (`data.time`); the
  `timestamp` param in `record()` makes that a 1-line change if wanted. Low priority
  (sim demos are throwaway).

## Key commands

```bash
cd ~/Documents/NTU/Y4S1/FYP

# --- Stage 1+3: render RGB-D and verify the geometry (FYP venv) ---
uv run python scripts/render_depth.py --camera workspace --verify
uv run python scripts/depth_to_3d.py --verify

# --- Stage 2: render a frame for the detector (FYP venv) ---
uv run python scripts/detect_objects.py --render-only \
  --camera workspace --out data/frames/frame_top.png

# --- Stage 2: detect (lerobot venv — needs torch) ---
cd ~/lerobot
uv run python /home/jj/Documents/NTU/Y4S1/FYP/scripts/detect_objects.py \
  --image /home/jj/Documents/NTU/Y4S1/FYP/data/frames/frame_top.png \
  --queries "red cube" "green cube" "blue cube" "yellow cube" "bin" \
  --model google/owlv2-base-patch16-ensemble --threshold 0.3 \
  --nms-iou 0.5 --top1-per-query \
  --out  /home/jj/Documents/NTU/Y4S1/FYP/data/frames/detections_owlv2.png \
  --json /home/jj/Documents/NTU/Y4S1/FYP/data/frames/detections.json

# --- Stage 3 with the detector in the loop (FYP venv) ---
cd ~/Documents/NTU/Y4S1/FYP
uv run python scripts/depth_to_3d.py --detections data/frames/detections.json

# --- Launch the scene in the viewer (FYP venv) ---
XDG_SESSION_TYPE=x11 uv run python -c "import mujoco,mujoco.viewer as v; m=mujoco.MjModel.from_xml_path('assets/mujoco/ur5e/scene_gripper.xml'); d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0); mujoco.mj_forward(m,d); v.launch(m,d)"
```
