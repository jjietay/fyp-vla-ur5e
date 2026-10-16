# FYP Technical Plan V5 — Deliverables and Route

**Project A1005-261** · Modular vs. End-to-End Vision-Language-Action Models for Embodied Robotic Manipulation

*Title updated by Prof Cheah, 31 Aug 2026, from "Development of a Large Language Model-Based Interface for Robotic Systems".*
**Student:** Tay Juen Jie · **Supervisor:** A/P Cheah Chien Chern · **Lab:** Robotics I (S2-B6a-01)
**Cycle:** FYP 2026 S1 (Aug 26 – May 27) · **Written:** 11 Aug 2026 · **Trimmed:** 16 Oct 2026, completed work and history removed (see git history)

**Research question:** a modular LLM pipeline (A) versus an end-to-end VLA (B), on the same UR5e, through the same controller interface — what does each buy and what does each cost?

**Nov 2026 is a completion deadline, not a submission deadline.** All work, the evaluation, the final report and a demo-ready system are finished by Nov 2026. Everything is submitted on the official NTU dates in 2027.

---

## 1 · NTU-mandated deliverables (D-series)

From `docs/guidelines/fyp_guidelines.pdf` and the schedule table. The "ready by" column is the internal completion deadline.

| ID | Deliverable | Submit on | Spec | Ready by |
|----|-------------|-----------|------|----------|
| **D1** | Project Plan/Strategy | **14 Sep 2026** (W6) | Objectives, background, proposed approach, weekly schedule in chart form | Submitted |
| **D2** | Interim Report + Video | **10 Nov 2026** (W13) | 5–10 A4 pages; video 2–3 min; Turnitin first | 10 Nov 2026 — content is near-final results, not "progress so far" |
| **D3** | Draft Final Report | **25 Mar 2027** | To supervisor; Turnitin | **2 Nov 2026** (W12) |
| **D4** | Final Report | **09 Apr 2027** | 40–60 pages main body, TNR 12, 1.5 spacing, 35 mm left / 30 mm other margins | **9 Nov 2026** (W13) |
| **D5** | Project Demonstration | **12–16 Apr 2027** | Arranged with examiner; live demo | System demo-ready and rehearsed by W12 Nov 2026 |
| **D6** | Oral Presentation | **10–12 May 2027** | 15 min talk + 10 min Q&A | Slides drafted W13 Nov 2026; rehearse Apr–May 2027 |
| **D7** | Full-text to Library + revised final report | **19 May 2027** | DR-NTU submission | Post-oral revision only |

**Mark allocation** (from `assessment_criteria_for_final_report.pdf`):

- Interim Assessment — 20 (Plan/Strategy 7, Interim Report 13)
- Report, demonstration and oral presentation — 40 (writing/references 11, intro-theory-review-discussion 12, demo + oral 17)
- Approach, development, achievement — 40 (initiative + supervisor updates 8, **design/implementation/data/results/complexity 24**, ability and independence 8)

The 24-mark block is the largest single item and is explicitly about *design of experiments, appropriate experimental technique, mastery of data collection, understanding of tools and their limitations, new and interesting results*. §4, §5 and §6 exist to hit that block directly.

---

## 2 · Scope statement (for D1, reuse verbatim)

> This project compares two architectures for **spoken** natural-language control of a UR5e manipulator, evaluated on the same physical cell, the same controller interface, the same speech front-end, and the same task suite. The user speaks a command — *"place that red cube into that metal tray"* — and the system executes it.
>
> A shared automatic speech recognition front-end converts the spoken command to text and hands **the same transcript** to both architectures. It sits outside both, so speech quality cannot bias the comparison.
>
> **Architecture A (modular)** composes an open-vocabulary object detector, RGB-D depth back-projection, a hand-eye transform into the robot base frame, and a large language model that plans over a schema of scripted motion primitives. Nothing in it is trained; behaviour is changed by editing prompts, schemas and primitives.
>
> **Architecture B (end-to-end)** fine-tunes SmolVLA, a compact vision-language-action model, on demonstrations recorded on the same UR5e. It maps camera images, robot state and a natural-language instruction directly to chunks of robot-hand (TCP) pose changes. Behaviour is changed only by collecting more data and retraining.
>
> The two are evaluated on a four-tier task suite spanning static pick-and-place, unambiguous liquid pouring, **ambiguous instructions that require clarifying the user's intent**, and **dynamic tasks whose world state changes mid-execution**. The contribution is a like-for-like characterisation of where each architecture succeeds, where each fails, how each fails, and what each costs in data, compute, money and engineering effort.

**Explicitly out of scope:** simulated results of any kind, mobile manipulation, multi-arm, learned grasp synthesis, training a VLA from scratch, training or fine-tuning any speech model (an off-the-shelf ASR system is used as-is), wake-word detection and always-on listening (push-to-talk, see §6.3).

---

## 3 · Technical deliverables (T-series)

Each has a **definition of done** — a binary condition, checkable, no judgement call.

| ID | Deliverable | Definition of done |
|----|-------------|--------------------|
| **T1** | Physical cell, calibrated | Fixed RGB-D + wrist camera streaming; `T_base_cam` solved and saved to `config/calibration/`; validated residual RMS reported in mm against measured ground truth |
| **T2** | Architecture A, end to end | One command line turns a spoken instruction into executed motion on the real arm, with per-stage logs; succeeds on Tier 0 at ≥70% over 20 trials |
| **T3** | Demonstration dataset | ≥50 real UR5e episodes per trained task, exported to a LeRobot dataset that loads without error, with fps asserted against the configured record rate |
| **T4** | Architecture B, trained and running | Fine-tuned SmolVLA checkpoint driving the arm through the same controller interface as T2, with a safety envelope on predicted actions; succeeds on Tier 0 at ≥50% over 20 trials |
| **T5** | Evaluation harness | **DONE 11 Aug 2026.** `src/fyp/evaluation/`, suite frozen before either pipeline was tuned, per-trial CSV, four results tables, imports neither architecture |
| **T6** | Comparison result | The T5 harness run to completion on both architectures across all attempted tiers, with the results tables and figures that go into D4 |
| **T7** | Repo consolidation | README and docs match the tree; no dead references |

---

## 4 · Task suite and hypotheses

Fix this **before** tuning either pipeline. Once you start tuning, any change to the task set is fitting the benchmark to whichever pipeline you happened to build first.

### Tier 0 — Static pick-and-place *(verification)*
Spoken: `"Place that red cube into that metal tray."` Single unambiguous target, static scene, fixed tray. Deictic phrasing on purpose (§6.3).
**Role:** proves both pipelines are wired correctly. Not a headline result. If a pipeline cannot do Tier 0, nothing above it is meaningful.

### Tier 1 — Constrained-motion manipulation *(pouring, unambiguous)*
`"Pour the orange juice into the glass."` Both bottles present; the instruction names one. Requires grasping a bottle, transporting without spilling, and a controlled tilt over the glass.
**Why it matters:** the primitive is no longer a two-pose pick. Architecture A needs a hand-authored `pour` primitive with a tilt profile; Architecture B has to learn the tilt from demonstration. This is the cleanest test of *"who handles motion that is hard to script?"*

### Tier 2 — Ambiguous instruction *(the clarification tier)*
`"Pour me a drink."` Both orange juice and water are on the table. The instruction is under-determined.

- **Architecture A** exposes an `ask_user(question, options)` tool alongside the motion primitives. The planner emits it when detections are ambiguous relative to the instruction, receives the answer, and re-plans. Adding this is a schema change, not new architecture.
- **Architecture B has no channel to ask.** SmolVLA maps (images, state, instruction string) to chunks of TCP pose deltas. There is no output head that can produce a question. Under an ambiguous instruction it must do something — most likely collapse to whichever choice dominated its training data, or produce a hesitant/averaged trajectory between the two bottles.

**This asymmetry is a headline result, not a fairness problem.** Do not paper over it with an LLM wrapper. Measure it: run the ambiguous instruction N≥20 times against B and record *which bottle it picks and how consistently*. A model that picks orange juice 19/20 times regardless of what the user wanted is a concrete, quantified demonstration that an end-to-end policy substitutes its training prior for the user's intent. That single figure is worth more to the report than any success-rate table.

### Tier 3 — Dynamic, state-dependent task *(the drawer)*
`"Put the snack in the drawer."` The drawer starts closed. The robot opens it. **The human closes it again mid-task.** The robot must notice and re-open before placing.

**Why it matters:** this is the tier where the prediction flips. Architecture A must *detect* the state change and re-plan — which is exactly what the open-loop plan-then-execute pattern is bad at, and why re-detection before each grasp (§7-W4) is necessary work. Architecture B is a closed-loop visuomotor policy at 20 Hz; if the training demonstrations included re-closures, reactive recovery is the thing it is naturally good at.

### Tier 4 — Generalisation *(unseen objects and phrasings)*
Objects and instruction phrasings never seen in the demonstrations or the prompt examples. A's open-vocabulary detector should transfer; B should degrade.

### Why Architecture A needs a language model in front of its detector

OWLv2 matches one embedding per query, so it takes short noun phrases, never sentences. `planner.py::extract_queries` converts the command first, and that step is failure mode H9, which B does not have. Reasoning in `docs/adr/0004-owlv2-takes-noun-phrases.md`.

### Hypotheses to state up front

| # | Hypothesis | Tier that tests it |
|---|-----------|-------------------|
| H1 | Both architectures reach comparable success on static, well-specified tasks | 0 |
| H2 | B produces smoother, more successful contact-rich and constrained motion than hand-authored primitives | 1 |
| H3 | A resolves ambiguous instructions; B cannot, and instead collapses to its training prior | 2 |
| H4 | B recovers from mid-task world-state changes without explicit state estimation; open-loop A does not, and closing A's loop costs latency | 3 |
| H5 | A generalises to unseen objects far better than B for the same engineering effort | 4 |
| H6 | A's failures are attributable to a specific stage; B's are not attributable at all | all |
| H7 | A is robust to spoken phrasing that drifts from the training/prompt wording; B degrades as the transcript drifts from its training instruction strings | all, via §6.3 |
| H8 | A grounds deictic reference ("*that* red cube") against detected scene context; B cannot | 0, 2 |
| H9 | A has a failure mode B cannot have: choosing the wrong noun phrase for the detector, before perception even runs | all |

H6 is the one that most cleanly separates the architectures and it costs nothing extra to test — it falls straight out of A's per-stage logging (`architecture_a/trace.py`). H7 is nearly as cheap: run each tier's trials with a **held-out set of spoken paraphrases never used during recording**, and report success rate against phrasing distance.

---

## 5 · Evaluation protocol

**Fixed before tuning. Any change after tuning begins must be recorded with a reason in the report.**

- **Trials:** 20 per (architecture × tier). Object start poses drawn from a written randomisation table, identical for both architectures — same 20 layouts, same order.
- **Spoken commands are scripted and shared.** Write the 20 utterances down in advance, half using training-set phrasings and half held-out paraphrases (H7). Speak the same utterance to both architectures. Do not improvise at the microphone — an unscripted phrasing that happens to favour one architecture is an uncontrolled variable.
- **Record the transcript with every trial.** If ASR mis-transcribes, that trial's failure belongs to the speech front-end, not to A or B, and you can only tell them apart if you kept the text.
- **Success:** defined per tier in one sentence each, in the harness source, not in your head. E.g. Tier 1 success = "liquid transferred into the glass, glass upright, bottle returned to table, no spill outside the tray."
- **Per-trial record:** architecture, tier, trial index, layout ID, **spoken utterance + ASR transcript**, outcome, failure stage (shared: ASR; A only: detection / depth / transform / planning / execution), wall-clock duration, human interventions.

### Metrics

| Axis | A | B |
|------|---|---|
| Success rate per tier | ✓ | ✓ |
| Time to completion | ✓ | ✓ |
| Failure attributable to a stage | ✓ | ✗ *(this is the finding)* |
| Ambiguity resolved correctly | ✓ | expected ✗ |
| Recovery from mid-task perturbation | measure | measure |
| Robustness to held-out spoken phrasing | ✓ | expected to degrade *(H7)* |
| Deictic reference resolved | ✓ | expected ✗ *(H8)* |
| Effective control rate | plan-then-execute, report end-to-end latency | 20 Hz target, report chunk latency with/without async |
| ASR latency | shared — report separately, not charged to either | shared |
| Data cost | 0 demonstrations | episodes recorded, hours of lab time |
| Compute cost | 0 GPU-hours; log API tokens and SGD cost per episode | GPU-hours and wall-clock per fine-tune |
| Engineering cost | primitives + prompts + schemas, LOC and hours | dataset plumbing + training config, LOC and hours |
| Generalisation to unseen objects | ✓ | ✓ |

**Log the money.** Architecture A costs cents per command forever; Architecture B costs GPU-hours once and nothing per command. That tradeoff is a real axis of comparison, it is trivially measurable, and almost no student report includes it.

---

## 6 · Hardware and comparison rules

### 6.1 Cameras and teleoperation

Decided in `docs/adr/0002-fixed-rgbd-plus-wrist-rgb-square-on.md` (nothing may depend on wrist depth) and `docs/adr/0005-teleoperation-3d-jog-no-freedrive.md` (device still to confirm, A15).

### 6.2 Pouring rig — safety and mess

- Use **water tinted with food colouring** in both bottles for development. Real orange juice attracts pests, stains, and goes off in a shared lab.
- Everything happens **inside a shallow tray**. Spills are expected and a spill onto a UR5e controller or a RealSense is a project-ending event.
- **The robot pours into a glass standing on a table. It does not pour into a person.** "Feed the user" is the motivating scenario in the write-up, not a physical demo action. Any assisted-feeding framing in D4 should be explicitly labelled as motivation, with the safety case for why it was not physically demonstrated — that is a maturity signal to the examiner, not a weakness.
- Bottles should be **light, rigid and graspable by the OnRobot RG2 within its stroke (0 to 110 mm)**. Check the gripper stroke against the bottle diameter before buying anything.

### 6.3 Speech front-end

Setup in `docs/adr/0003-local-faster-whisper-push-to-talk.md`. Rules for the comparison:

- **ASR sits outside both architectures, and both get the same raw transcript.** Cleaning it for B inserts an LLM into B, and it is no longer end-to-end.
- **Paraphrase the instructions at record time**, 5 to 10 per task across the episode set, from the first recording session. Split them into a record-time set and a held-out evaluation set. Retrofitting means re-recording, and without it H7 cannot be tested.
- **Use deictic phrasings in the evaluation** ("*that* red cube"). A resolves them from detected scene context and B has only token statistics, so it tests H8 at no extra cost.
- **Log ASR latency separately** from planning and execution latency in §5.
- Spoken replies (TTS) for Tier 2 clarification are recommended, not required. If time is short, print the question.

### 6.4 Shopping list

Both cameras are in hand (13 Sep).

| Item | Priority | Approx. | Notes |
|------|----------|---------|-------|
| Wrist camera mount/bracket | **must** | S$0–60 | Print or fabricate; must not foul the gripper or the wrist-3 joint |
| USB cable routing / strain relief | **must** | S$20 | A cable snagging on a joint during a demo will end the demo |
| 3D jog device (type to confirm, A15) | **strong** | S$200–500 | See §6.1 |
| Tray, glasses, sealable bottles, food colouring | **must** | S$40 | |
| Small drawer unit for Tier 3 | **must** | S$40–100 | Handle must be graspable by the RG2 |
| **Headset or lapel microphone** | **must** | S$30–80 | §6.3. Laptop mic will not survive a noisy lab |
| Footswitch for push-to-talk | nice | S$30 | Frees both hands during a demo; a keyboard key works fine |

---

## 7 · Remaining work (W-series)

Ordered by dependency, not by week. §8 places them in time. W1, W2 and W2b are done.

### W3 · Calibration

1. First ball fit of `T_base_cam` with `scripts/calibrate_ball.py`, per `docs/adr/0007-no-marker-ball-fit-each-session.md`.
2. **Validate.** `scripts/check_calibration.py` moves the TCP above where the transform says an object is, and the miss is measured with a rule. This reach test is the only number that says whether the transform works. **A calibration you did not validate numerically is a calibration you do not have**, and every downstream failure will be blamed on it.

### W4 · Architecture A on hardware

1. Close the loop: re-detect before each grasp instead of trusting the plan-time position. Required for Tier 3. Document the failure modes you hit; they are results.
2. Run Tier 0 to the T2 threshold.

### W5 · Demonstration capture *(critical path)*

1. Teleop rig per §6.1.
2. **Lock the dataset conventions and write down the reasons** — action space (TCP pose deltas, `docs/adr/0006-tcp-pose-delta-action-space.md`), observation set (2 RGB views + state vector), resolution, control rate, **and the per-episode instruction string drawn from the paraphrase sets (§6.3), not a single fixed template**. **The action space must match what Architecture A's primitives command, or the two architectures are not solving the same problem and the comparison is unfair.** This is the single most important paragraph in the methodology chapter.
3. Record. **Budget: ~50 episodes per task minimum.** Community evidence says workspace *density* matters more than raw count — 50 episodes spread over a 30 cm region failed where 75 over 10 cm succeeded. **Constrain the workspace for the first fine-tune**, then widen it. At ~1.5–2 min per episode including reset, 250 episodes is 8–10 hours of lab time before retakes. Budget 15. Record in two or three dedicated multi-hour blocks, not scattered one-hour sessions.
4. Export to LeRobot; confirm the dataset loads and fps asserts.

### W6 · Architecture B training and inference

1. Fine-tune on Tier 0 first. Log config, seed, wall-clock, GPU-hours. **Keep the checkpoint and the exact dataset version together** — a checkpoint whose dataset you cannot identify is not a result.
2. **Inference loop** — checkpoint loads, observation in, action chunk out, `chunk_size = 50` matching the SmolVLA config.
3. **Safety envelope before the controller** — workspace bounds and velocity clamp on every predicted action. Non-negotiable. A VLA under a novel observation can emit a chunk that slams the arm.
4. **Async inference** so the arm is not idle during the forward pass; the reported gain is roughly 30% off completion time. Read the RTC paper (Black et al. 2025) **before** designing this, not after.
5. Re-train per tier as demonstrations land.

### W7 · Evaluation and write-up

1. Run both architectures through the T5 harness across all attempted tiers. **Freeze results before writing** — do not tune anything after the numbers start going into the report.
2. Figures: success rate by tier and architecture; A's failure-stage breakdown; Tier 2 choice distribution for B; cost table.
3. D3 → D4 → demo rehearsal.

### W8 · Reading *(interleave as breaks)*

- **RTC** (Black et al. 2025) — async inference and action chunking. Needed **before** W6.4.
- **OXE** (Padalkar et al. 2023) — action/observation formats, UR5 inclusion.
- SmolVLA — read and summarised already.

---

## 8 · Schedule to 10 Nov 2026

Week 1 = 10 Aug 2026. Recess week = 28 Sep. **Re-baselined 9 Sep 2026 after the lab move, and submitted in D1.** The Google Doc of D1 is the master copy of this schedule.

| Week | Dates | Focus | Milestone |
|------|-------|-------|-----------|
| **W1–W6** | 10 Aug–20 Sep | Architecture A software, speech, harness, B's robot side, detector, recorder, dataset conventions | Done; D1 submitted |
| **W7** | 21–27 Sep | Cell setup, action space check, hand-eye, `T_base_cam` validated with the reach error in mm | First arm session 23 Sep |
| **Recess** | 28 Sep–4 Oct | Architecture A end to end on hardware, Tier 0 | **T2 — A passes Tier 0** |
| **W8** | 5–11 Oct | First demonstration batch, first SmolVLA fine-tune | **T3 partial** |
| **W9** | 12–18 Oct | B on hardware behind the safety envelope, Tier 0. Tier 1 both architectures | **T4 — B passes Tier 0** |
| **W10** | 19–25 Oct | Tiers 2 and 3 | Tiers 2–3 both architectures |
| **W11** | 26 Oct–1 Nov | Full sweep, Tier 4, **freeze results** | **T5, T6 done** |
| **W12** | 2–8 Nov | Write D4, record the D2 video, rehearse the demo, draft to supervisor | Demo-ready; draft with supervisor |
| **W13** | 9–15 Nov | **Submit D2 interim + video, 10 Nov** | **D2 submitted; D4 complete** |

If arm access slips further, drop from the bottom of the descope ladder in §9 (Tier 4 first, then Tier 3). Do not thin every tier.

**Critical path:** W3 calibration → W5 demo capture → W6 training → W7 evaluation.

### After W13 — holding pattern to submission

| When | Action |
|------|--------|
| Dec 2026 – Mar 2027 | **Monthly:** re-run the full demo end-to-end, fix what has rotted, log it. Act on supervisor feedback on the draft |
| Jan–Feb 2027 | Optional extra trials to tighten the §5 numbers, or recover the highest descoped tier (§9) |
| **25 Mar 2027** | Submit D3 draft final report |
| **09 Apr 2027** | Submit D4 final report |
| **12–16 Apr 2027** | D5 demonstration — arrange the date with the examiner **well in advance**, not in April |
| **10–12 May 2027** | D6 oral presentation, 15 + 10 min |
| **19 May 2027** | D7 full text to DR-NTU + revised final report to supervisor |

---

## 9 · Risk register and descope ladder

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Hands-in-frame contaminates demos | Retrain from scratch, ~2 weeks lost | 3D jog device, no freedrive (A15); inspect the first 5 episodes' frames before recording 200 |
| **Demos recorded with one templated instruction string** | H7 becomes untestable and B looks artificially bad against spoken input; unfixable without re-recording | Paraphrase sets (§6.3) in use from the first recording session |
| Lab noise wrecks ASR | Trials fail for reasons unrelated to either architecture | Headset mic + push-to-talk; transcript logged per trial so speech failures are separable |
| `T_base_cam` too coarse to grasp reliably | A fails everywhere; B unaffected | Ball fit and reach test with `check_calibration.py` each session |
| B fails to converge on 50 episodes | No comparison | Constrain the workspace first (density beats count); widen only after the constrained set works |
| Spill damages equipment | Project-ending | Tray, tinted water, never real juice, arm speed capped during pour |
| **12 GB VRAM too tight for the SmolVLA fine-tune** | W6 stalls on the critical path | Run a short fine-tune at batch size 1 on the first real episodes (A4). If it does not fit: small batches with gradient accumulation, gradient checkpointing, an 8-bit optimiser, a frozen vision backbone |
| GPU time insufficient | Training bottleneck | Confirm GPU access (A4); SmolVLA is 450M params and fine-tunes on a single consumer GPU, but budget the hours |
| Report underruns 40 pages | Resubmission | The per-stage failure logs and cost tables generate content; start writing W11, not W13 |

### Descope ladder — drop from the bottom

1. **Architecture A end to end, spoken input, Tiers 0–2.** No training, no data collection, entirely in your control. **This alone is a passing FYP.**
2. **Architecture B, Tier 0 only.** Establishes both pipelines exist and are comparable.
3. **Tier 1 pouring, both architectures.**
4. **Tier 2 ambiguity experiment.** Cheap for A, and B's failure mode costs only trial time — high result-per-hour.
5. **Tier 3 dynamic drawer.**
6. **Tier 4 generalisation sweep.**
7. Spoken clarification replies (TTS). Text questions are an acceptable fallback.
8. Repo consolidation and documentation polish.

Note the ordering: **Tier 2 is cheaper than Tier 3 and produces a more distinctive result.** If time gets short, cut the drawer before cutting the ambiguity experiment.

**Speech input is not on this ladder.** The official summary promises *"more intuitive human–robot interaction"*, and spoken command is the most direct reading of it.

---

## 10 · Status and open actions

### Status, 13 Oct 2026

This is the only status record in the repo. Update it here, not in `CLAUDE.md` or `README.md`.

- **Runs on the real cell:** speech; `URController` motion (`go_home.py` within 0.012 degrees of `home_q`, `tests/hardware -k "not gripper"` 7 of 7 twice, 23 Sep); the RG2 gripper; the fixed camera and OWLv2 on real frames; Architecture A end to end ("pick up the orange", 1 Oct).
- **Written, not yet run on hardware:** the finger-width grasp check in `skills.py`, `calibrate_ball.py` and the detector model cache (all 13 Oct); `test_orientation.py`; the evaluation harness; Architecture B's robot side. There is no Architecture B agent for the harness yet: `run_evaluation.py --architecture B` prints a message and exits.
- **Not done:** the first ball fit of `T_base_cam` (the current file is the 1 Oct hand correction, `validated: false`); the drawer `pull_axis`; the e-stop recovery test; all demonstration capture. **There are zero episodes.**

### Open actions

Closed actions are in git history.

| ID | Action | Owner | By |
|----|--------|-------|-----|
| **A4** | Confirm GPU availability and hours for fine-tuning. The throwaway fine-tune that checks whether 12 GB is enough waits for the first real episode, because no episode data exists | JJ | before W6 |
| **A7** | Agree the D5 demonstration date with the examiner. Arrange it months ahead, not in April 2027 | JJ | Jan 2027 |
| **A9** | Send Prof Cheah a short written progress update **every two weeks**. This was skipped through July. It is directly worth 8 marks | JJ | fortnightly |
| **A10** | Set `ANTHROPIC_API_KEY`. Architecture A cannot run a single instruction without it | JJ | open |
| **A11** | Back up `data/raw/` outside the repo before recording starts. It is gitignored, and the demonstrations are 15 hours of lab time that git will not protect | JJ | before W5 |
| **A14** | Ask Koji about moving the arm closer to the table and about the new layout. The arm base is about 40 cm from the useful table area. Measure `workspace.x` and `workspace.y` again only after the layout is fixed | JJ | next week |
| **A15** | Confirm the 3D jog device that will drive the demonstrations, see `docs/adr/0005-teleoperation-3d-jog-no-freedrive.md` | JJ | before W5 |

---

## 11 · Mapping deliverables to marks

| Assessment item | Marks | What earns it here |
|-----------------|-------|--------------------|
| Project Plan/Strategy | 7 | D1, submitted |
| Interim Report | 13 | D2 — by Nov the work is largely complete, so this reads as results rather than progress |
| Report writing, structure, references | 11 | Cite SmolVLA, RTC, OXE, OWLv2 properly; §7-W8 is not optional reading |
| Intro, theory, review, discussion, conclusion | 12 | The A-vs-B framing is inherently a comparative review; §4's hypotheses give the discussion its spine, and §9 gives you "limitations of current work" for free |
| Demonstration + oral | 17 | D5 shows both architectures live on the same cell — Tier 2 side by side (A asks, B guesses) is the moment that lands. Finishing in Nov means five months of rehearsal, which is the whole point |
| Initiative + regular supervisor updates | 8 | §10-A9, fortnightly. **This was skipped through July.** It is 8 marks for an email |
| **Design, implementation, data, results, complexity** | **24** | §5's protocol is a designed experiment with stated hypotheses and controlled layouts; T3 is real data collection; §5's cost table shows understanding of tool *limitations*. This is the block to over-serve |
| Ability, independence, extending ideas | 8 | The widened task suite beyond pick-and-place *is* the extension. Say so explicitly in the report |

**One framing note for D4.** The official summary promises *"a user interface based on large language models… enabling more intuitive human–robot interaction."* Speech is the visible half of that promise and the comparison is the intellectual half. Lead the report with the interface — the user talks, the robot acts — then use the A-vs-B comparison to answer *what has to sit behind that interface for it to work.* That reads as a project with a thesis, rather than two pipelines benchmarked against each other.

---

*Sources for the deliverable dates and specifications: `docs/guidelines/fyp_guidelines.pdf`, `fyp_deliverables.png`, `assessment_criteria_for_final_report.pdf`, `SuggestedFormatInterimReport.pdf`, `project_background.png`.*
