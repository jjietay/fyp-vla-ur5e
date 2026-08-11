# FYP Technical Plan V5 — Deliverables and Route

**Project A1005-261** · Development of a Large Language Model-Based Interface for Robotic Systems
**Student:** Tay Juen Jie · **Supervisor:** A/P Cheah Chien Chern · **Lab:** Robotics I (S2-B6a-01)
**Cycle:** FYP 2026 S1 (Aug 26 – May 27) · **Written:** 11 Aug 2026 · **Supersedes and replaces:** all earlier plans, `SESSION_HANDOVER.md`, and the old vault architecture note (deleted 11 Aug)

---

## 0 · Purpose of this document

V4 answered *what code is missing*. This answers *what has to exist for the FYP to be assessed well, and in what order*.

Two things changed after the supervisor meeting on/around 10 Aug 2026:

1. **Application scope widened.** The comparison is no longer only pick-and-place. It now spans a tiered task suite including interactive clarification (pouring a drink the user must disambiguate) and dynamic, state-dependent tasks (drawer that gets closed mid-task). Pick-and-place is retained but demoted to a *verification tier*, not the headline result.
2. **Work-complete target pulled forward to Nov 2026.** All technical work, the full evaluation, the final report and a demo-ready system are to be **finished** by Nov 2026. **Submission dates are unchanged** — the draft final report, final report, demonstration and oral presentation all go in on the official NTU dates in 2027. Nov 2026 is an internal completion deadline, not a submission one. See §2.

The research question is unchanged: **a modular LLM pipeline (A) versus an end-to-end VLA (B), on the same UR5e, through the same controller interface — what does each buy and what does each cost?**

---

## 1 · NTU-mandated deliverables (D-series)

From `docs/fyp_deliverables/fyp_guidelines.pdf` and the schedule table. **All submissions happen on the official dates.** The "ready by" column is the internal completion deadline — the artefact exists, finished, on that date and then sits until submission.

| ID | Deliverable | Submit on | Spec | Ready by |
|----|-------------|-----------|------|----------|
| **D1** | Project Plan/Strategy | **14 Sep 2026** (W6) | Objectives, background, proposed approach, weekly schedule in chart form | 14 Sep 2026 — derived from §8/§9 of this doc |
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

The 24-mark block is the largest single item and is explicitly about *design of experiments, appropriate experimental technique, mastery of data collection, understanding of tools and their limitations, new and interesting results*. §5, §6 and §7 of this document exist to hit that block directly.

---

## 2 · What the Nov 2026 target actually means

Today is 11 Aug 2026. Week 1 of AY2026 S1 begins 10 Aug 2026; the recess week falls the week of 28 Sep. That gives **13 teaching weeks + 1 recess week to 10 Nov 2026**, i.e. **~13 usable weeks** to a finished project.

**Nov 2026 is a completion deadline, not a submission deadline.** Everything still goes in on the official dates. This is a deliberately safe structure and it is worth being explicit about why:

- **Submission risk goes to zero.** By the time the final report is due in Apr 2027, it has existed in finished form for five months. Nothing about the outcome depends on the last fortnight going well.
- **Slippage is absorbed silently.** If lab access lands late or a fine-tune fails, the overrun eats buffer instead of eating the deadline. The plan degrades rather than breaks.
- **The interim assessment gets results, not progress.** D2 on 10 Nov is the one date that is both a submission and a completion point. Submitting an interim report full of finished comparative results — rather than "I have built half a pipeline" — is worth real marks against the 13-mark interim criterion.

Consequences for how the 13 weeks are run:

- There is **no slack for a late lab start.** Every week without arm access comes straight out of Track B, which is on the critical path. §12-A1 is the highest-priority action in this document.
- Data collection must be **batched, not incremental.** Recording 200–300 demonstrations in scattered one-hour sessions will not fit. Plan two or three dedicated multi-hour blocks.
- The descope ladder in §11 is a live tool, not a contingency to think about later.

### What Dec 2026 – May 2027 is for

Do not let this window become scope creep, and do not let it become idle. In priority order:

1. **Absorb whatever slipped.** Most likely candidate is the highest descoped tier.
2. **Deepen the evidence.** More trials on the tiers that produced the most interesting results — a 20-trial result becomes a 50-trial result with tighter error bars for nothing but lab time. Cheapest possible quality gain.
3. **Keep the demo alive.** A system that has not been run since Nov will not work in Apr. Re-run the full demo end-to-end **once a month, Dec through Mar**, and fix the rot when you find it. Hardware demos rot silently.
4. **Report and oral polish.** Supervisor feedback on the Nov draft has months to be acted on. Oral prep from Mar.

Put a monthly recurring reminder on items 3 and 4 now, while you are thinking about it.

---

## 3 · Scope statement (for D1, reuse verbatim)

> This project compares two architectures for **spoken** natural-language control of a UR5e manipulator, evaluated on the same physical cell, the same controller interface, the same speech front-end, and the same task suite. The user speaks a command — *"place that red cube into that metal tray"* — and the system executes it.
>
> A shared automatic speech recognition front-end converts the spoken command to text and hands **the same transcript** to both architectures. It sits outside both, so speech quality cannot bias the comparison.
>
> **Architecture A (modular)** composes an open-vocabulary object detector, RGB-D depth back-projection, a hand-eye transform into the robot base frame, and a large language model that plans over a schema of scripted motion primitives. Nothing in it is trained; behaviour is changed by editing prompts, schemas and primitives.
>
> **Architecture B (end-to-end)** fine-tunes SmolVLA, a compact vision-language-action model, on demonstrations recorded on the same UR5e. It maps camera images, robot state and a natural-language instruction directly to joint action chunks. Behaviour is changed only by collecting more data and retraining.
>
> The two are evaluated on a four-tier task suite spanning static pick-and-place, unambiguous liquid pouring, **ambiguous instructions that require clarifying the user's intent**, and **dynamic tasks whose world state changes mid-execution**. The contribution is a like-for-like characterisation of where each architecture succeeds, where each fails, how each fails, and what each costs in data, compute, money and engineering effort.

**Explicitly out of scope:** simulated results of any kind, mobile manipulation, multi-arm, learned grasp synthesis, training a VLA from scratch, training or fine-tuning any speech model (an off-the-shelf ASR system is used as-is), wake-word detection and always-on listening (push-to-talk, see §7.5).

---

## 4 · Technical deliverables (T-series)

Each has a **definition of done** — a binary condition, checkable, no judgement call.

| ID | Deliverable | Definition of done |
|----|-------------|--------------------|
| **T1** | Physical cell, calibrated | Fixed RGB-D + wrist camera streaming; `T_base_cam` solved and saved to `config/calibration/`; validated residual RMS reported in mm against measured ground truth |
| **T2** | Architecture A, end to end | One command line turns a typed instruction into executed motion on the real arm, with per-stage logs; succeeds on Tier 0 at ≥70% over 20 trials |
| **T3** | Demonstration dataset | ≥50 real UR5e episodes per trained task, exported to a LeRobot dataset that loads without error, with fps asserted against the configured record rate |
| **T4** | Architecture B, trained and running | Fine-tuned SmolVLA checkpoint driving the arm through the same controller interface as T2, with a safety envelope on predicted actions; succeeds on Tier 0 at ≥50% over 20 trials |
| **T5** | Evaluation harness | Tiered task suite fixed and written down **before** either pipeline is tuned; scripted trial protocol; per-trial outcome + failure-stage logged to a machine-readable file |
| **T6** | Comparison result | The T5 harness run to completion on both architectures across all attempted tiers, with the results tables and figures that go into D4 |
| **T7** | Repo consolidation | README and vault docs match the tree; no dead references |

---

## 5 · Task suite and hypotheses

Fix this **before** tuning either pipeline. Once you start tuning, any change to the task set is fitting the benchmark to whichever pipeline you happened to build first.

### Tier 0 — Static pick-and-place *(verification)*
Spoken: `"Place that red cube into that metal tray."` Single unambiguous target, static scene, fixed tray. Deictic phrasing on purpose (§7.4).
**Role:** proves both pipelines are wired correctly. Not a headline result. If a pipeline cannot do Tier 0, nothing above it is meaningful.

### Tier 1 — Constrained-motion manipulation *(pouring, unambiguous)*
`"Pour the orange juice into the glass."` Both bottles present; the instruction names one. Requires grasping a bottle, transporting without spilling, and a controlled tilt over the glass.
**Why it matters:** the primitive is no longer a two-pose pick. Architecture A needs a hand-authored `pour` primitive with a tilt profile; Architecture B has to learn the tilt from demonstration. This is the cleanest test of *"who handles motion that is hard to script?"*

### Tier 2 — Ambiguous instruction *(the clarification tier)*
`"Pour me a drink."` Both orange juice and water are on the table. The instruction is under-determined.

- **Architecture A** exposes an `ask_user(question, options)` tool alongside the motion primitives. The planner emits it when detections are ambiguous relative to the instruction, receives the answer, and re-plans. Adding this is a schema change, not new architecture.
- **Architecture B has no channel to ask.** SmolVLA maps (images, state, instruction string) to joint action chunks. There is no output head that can produce a question. Under an ambiguous instruction it must do something — most likely collapse to whichever choice dominated its training data, or produce a hesitant/averaged trajectory between the two bottles.

**This asymmetry is a headline result, not a fairness problem.** Do not paper over it with an LLM wrapper. Measure it: run the ambiguous instruction N≥20 times against B and record *which bottle it picks and how consistently*. A model that picks orange juice 19/20 times regardless of what the user wanted is a concrete, quantified demonstration that an end-to-end policy substitutes its training prior for the user's intent. That single figure is worth more to the report than any success-rate table.

### Tier 3 — Dynamic, state-dependent task *(the drawer)*
`"Put the snack in the drawer."` The drawer starts closed. The robot opens it. **The human closes it again mid-task.** The robot must notice and re-open before placing.

**Why it matters:** this is the tier where the prediction flips. Architecture A must *detect* the state change and re-plan — which is exactly what the open-loop plan-then-execute pattern is bad at, and why V4 lists re-detection before each grasp (A7) as necessary work. Architecture B is a closed-loop visuomotor policy at 30 Hz; if the training demonstrations included re-closures, reactive recovery is the thing it is naturally good at.

### Tier 4 — Generalisation *(unseen objects and phrasings)*
Objects and instruction phrasings never seen in the demonstrations or the prompt examples. A's open-vocabulary detector should transfer; B should degrade.

### Hypotheses to state up front

| # | Hypothesis | Tier that tests it |
|---|-----------|-------------------|
| H1 | Both architectures reach comparable success on static, well-specified tasks | 0 |
| H2 | B produces smoother, more successful contact-rich and constrained motion than hand-authored primitives | 1 |
| H3 | A resolves ambiguous instructions; B cannot, and instead collapses to its training prior | 2 |
| H4 | B recovers from mid-task world-state changes without explicit state estimation; open-loop A does not, and closing A's loop costs latency | 3 |
| H5 | A generalises to unseen objects far better than B for the same engineering effort | 4 |
| H6 | A's failures are attributable to a specific stage; B's are not attributable at all | all |
| H7 | A is robust to spoken phrasing that drifts from the training/prompt wording; B degrades as the transcript drifts from its training instruction strings | all, via §7.4 |
| H8 | A grounds deictic reference ("*that* red cube") against detected scene context; B cannot | 0, 2 |

H6 is the one that most cleanly separates the architectures and it costs nothing extra to test — it falls straight out of the per-stage logging in §8-W4. H7 is nearly as cheap: run each tier's trials with a **held-out set of spoken paraphrases never used during recording**, and report success rate against phrasing distance.

---

## 6 · Evaluation protocol

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
| Effective control rate | plan-then-execute, report end-to-end latency | 30 Hz target, report chunk latency with/without async |
| ASR latency | shared — report separately, not charged to either | shared |
| Data cost | 0 demonstrations | episodes recorded, hours of lab time |
| Compute cost | 0 GPU-hours; log API tokens and SGD cost per episode | GPU-hours and wall-clock per fine-tune |
| Engineering cost | primitives + prompts + schemas, LOC and hours | dataset plumbing + training config, LOC and hours |
| Generalisation to unseen objects | ✓ | ✓ |

**Log the money.** Architecture A costs cents per command forever; Architecture B costs GPU-hours once and nothing per command. That tradeoff is a real axis of comparison, it is trivially measurable, and almost no student report includes it.

---

## 7 · Hardware decisions

### 7.1 Cameras — the answer to "do I need RGB-D on the wrist?"

**No. RGB is sufficient for the wrist. Here is why, and here is when to claim RGB-D anyway.**

SmolVLA consumes **multi-view RGB images, a robot state vector, and a language string**. It has no depth input. The LeRobot convention is `OBS_IMAGE_1` = top-down/overhead and `OBS_IMAGE_2` = wrist. Adding a depth stream to the wrist camera contributes literally nothing to Architecture B.

Architecture A does need depth, but only from the **fixed** camera — that is where detection happens and where pixels get back-projected into 3D and transformed into the base frame. The existing lab RGB-D, fixed as the supervisor recommends, covers it.

So the minimum correct configuration is:

| Camera | Type | Used by A | Used by B |
|--------|------|-----------|-----------|
| Fixed overhead/side | **RGB-D** (existing lab unit) | detection + depth → 3D → base frame | `OBS_IMAGE_1` (RGB only) |
| Wrist-mounted | **RGB is enough** | optional close-range re-detect | `OBS_IMAGE_2` (RGB only) |

**When to claim RGB-D on the wrist anyway.** If the cost falls on the lab rather than you, an RGB-D wrist camera buys optionality: close-range grasp verification and visual servoing in Architecture A, which would strengthen the Tier 3 re-planning story. If you do claim one, **specify a short-minimum-range model** — a standard D435-class sensor has a minimum depth range around 0.3 m, which is useless at grasp distance; the D405-class short-range units are the ones designed for wrist mounting.

**Hard rule either way: nothing in the pipeline may depend on wrist depth.** Design as if the wrist camera is RGB. If depth arrives, it is a bonus branch you can delete without touching anything else. This keeps you shippable if procurement stalls.

**Justify two cameras to the supervisor on evidence, not preference.** The LeRobot community finding is that policies trained with overhead + wrist produce noticeably smoother trajectories than wrist-only, and the reference SmolVLA fine-tuning setups use two views. Single-view fine-tuning is the likelier-to-underperform option, and if B underperforms because of a camera configuration you chose against the reference setup, the comparison is compromised and the result is unpublishable. That argument is worth making explicitly.

### 7.2 Teleoperation for demonstration capture — decide this in Week 1

This is the **most underestimated risk in the whole project**, because it silently determines whether T3 data is usable.

| Option | Cost | Setup | Problem |
|--------|------|-------|---------|
| **UR5e freedrive** (hand-guide the arm, record joint states) | £0 | hours | **Your hands and arms appear in both camera views on every frame.** A visuomotor policy can key on them, and at test time they are absent. This is a known contamination mode for kinesthetic teaching |
| **SpaceMouse / 3D mouse** → TCP velocity → `servoL` | ~S$200–500 | 1–2 days | Nothing in frame. Learning curve on 6-DOF input; needs a gripper button binding |
| **Leader-follower arm** | ~S$150+ | days | The LeRobot-native pattern, but a 5-DOF leader mapping to a 6-DOF UR5e is awkward and is not a solved thing you can copy |

**Recommendation:** prove the recording loop with **freedrive in the first lab session** — it needs no purchase and no wiring — but **order a SpaceMouse now**, and record the real training set with it. Hands-in-frame is not a cosmetic issue; discovering it after 200 episodes costs the project. If you do end up shipping freedrive data, say so in the report and treat it as a named confound.

**Separately, the recording path itself needs rewriting for hardware.** Real `moveJ`/`moveL` **block**; the sim recorder counts ticks inside a stepping loop. On hardware, poll `get_state` from a **separate thread on a clock-gated timer** at the configured rate, with camera frames captured in sync. Do not port `scripts/record_episode.py` — it still carries the tick-counting cadence bug that was fixed in `sim_server.py` and never back-ported. Write the hardware recorder fresh against `DemoRecorder`, which already takes an explicit `timestamp`.

### 7.3 Pouring rig — safety and mess

- Use **water tinted with food colouring** in both bottles for development. Real orange juice attracts pests, stains, and goes off in a shared lab.
- Everything happens **inside a shallow tray**. Spills are expected and a spill onto a UR5e controller or a RealSense is a project-ending event.
- **The robot pours into a glass standing on a table. It does not pour into a person.** "Feed the user" is the motivating scenario in the write-up, not a physical demo action. Any assisted-feeding framing in D4 should be explicitly labelled as motivation, with the safety case for why it was not physically demonstrated — that is a maturity signal to the examiner, not a weakness.
- Bottles should be **light, rigid and graspable by the 2F-85 within its stroke**. Check the gripper stroke against the bottle diameter before buying anything.

### 7.4 Speech front-end

The user speaks; the system acts. `"Place that red cube into that metal tray."`

**Architectural rule: ASR sits outside both architectures and is identical for both.** One microphone, one ASR model, one transcript, handed to A and to B unchanged. The moment you clean up the transcript for one and not the other, speech becomes a confound and the comparison is dead.

**Feed both the raw transcript.** This matters more than it sounds:

- **Architecture A** absorbs disfluent, verbose, conversational speech natively — that is what an LLM is. *"uh, can you put that red cube in the, the metal tray"* is not a problem for it.
- **Architecture B** takes the instruction as a **string conditioned on its training distribution.** SmolVLA was fine-tuned on whatever phrasings you stored in the episodes. A transcript that drifts from those phrasings degrades it, and there is no mechanism inside the policy to normalise the wording. If you normalise the transcript before handing it to B, you have inserted an LLM into B's pipeline and it is no longer end-to-end.

**Deixis is a second asymmetry, and it is free to test.** *"That* red cube" and *"that* metal tray" are deictic — there is no pointing input, so "that" can only be resolved from scene context. Architecture A resolves it naturally because the detected objects and their positions are already in the planner's prompt: if exactly one red cube is detected, "that red cube" is unambiguous, and if two are, the planner should emit `ask_user`. Architecture B has no grounding mechanism for "that" beyond token statistics from training. Use deictic phrasings deliberately in the evaluation — they cost nothing extra and they probe something real.

**Paraphrase the instructions at record time.** The instruction string is per-episode metadata in a LeRobot dataset, so varying it is free — just say the task differently while recording. Use **5–10 paraphrases per task** across the episode set rather than one templated string repeated 50 times. This is the single cheapest thing you can do to make B's comparison against spoken input fair, and if you skip it, B's failure at test time is partly your dataset's fault rather than the architecture's. Do this from the very first recording session; retrofitting means re-recording.

**You are not building speech recognition. You are calling one.** Nothing here is novel work and none of it belongs in the contribution claims — treat ASR as a solved off-the-shelf component, like the RGB-D driver.

**Cost is not a decision factor, so do not choose on price.** Total spoken audio across the entire project is roughly 500 utterances × ~5 s ≈ **45 minutes**. At 2026 cloud rates of roughly $0.004–0.017/min, that is **under one dollar for the whole FYP**. Choose on reliability and setup friction instead.

| Option | Type | Verdict |
|--------|------|---------|
| **`faster-whisper`** (Whisper via CTranslate2) | local, free | **Recommended.** No network dependency, no API key in the repo, runs on the GPU you already have. Whisper leads on accented English, which matters here |
| `whisper.cpp` | local, free | Equivalent; better if you want CPU-only or minimal dependencies |
| Vosk | local, free | Lower accuracy; only worth it if you need sub-500 ms latency, which push-to-talk means you do not |
| OpenAI / Deepgram / AssemblyAI / Google APIs | cloud, ~$0.004–0.017/min | Perfectly fine and cheaper to set up. **One real drawback: your live examiner demo would depend on lab wifi.** Keep one wired as a fallback, do not make it the primary |
| Browser Web Speech API | cloud, free | Zero install, but awkward to wire into a Python robotics stack. Not worth the integration |

**The deciding argument is the demo, not the technology.** All of these are accurate enough for short scripted commands in a quiet-ish room. The difference that matters is that a local model cannot fail because the campus network did, and you will be running this in front of an examiner. **Ship local, keep a cloud path behind a config flag as a fallback.**

**You do not need streaming.** Push-to-talk means you capture a complete 3–5 s clip and transcribe it in one shot. That sidesteps endpointing, partial hypotheses, and the entire streaming-latency discussion. Whisper-small on a local GPU returns a short clip in a few hundred milliseconds.

**Implementation:**

| Decision | Choice | Reason |
|----------|--------|--------|
| ASR model | `faster-whisper`, small or medium, run locally | See table above |
| Trigger | **Push-to-talk** (hold a key or footswitch) | Kills the whole endpointing/VAD problem. A robotics lab is noisy — fans, HVAC, other students. Always-on listening is a research project of its own and is out of scope |
| Microphone | **Headset or lapel mic**, not the laptop's | Laptop mics pick up the arm and the room. ~S$30–80 and it removes an entire class of failure |
| Spoken replies (TTS) | **Recommended, not required** | Tier 2's clarification is far more convincing as a spoken exchange than as text on a screen — the examiner watches the robot ask and you answer. Piper or an API TTS is an afternoon's work. If time is short, print the question instead |
| Compute type | **`compute_type="float16"` — set it explicitly** | See the Blackwell note below. Do not leave it on the default |
| GPU contention | Not an issue — ASR runs at demo/eval time, fine-tuning runs separately | If they ever do overlap, drop ASR to CPU; a 5 s clip is fine on CPU |

**Blackwell (RTX 50-series) gotcha — set this before you lose an afternoon to it.** CTranslate2, the runtime under `faster-whisper`, has a known incompatibility with the sm_120 architecture: the default **int8 quantisation crashes with `CUBLAS_STATUS_NOT_SUPPORTED`** because Blackwell's int8 tensor cores need padding that older CTranslate2 builds do not emit. **The fix is to force `compute_type="float16"`.** You lose int8's memory saving, which is irrelevant here — see below. Also ensure CUDA 12 and cuDNN 9; recent CTranslate2 requires both.

**VRAM is a non-issue for ASR.** `large-v3` at float16 needs roughly **3.4 GB**, and small/medium are well under that. On a 12 GB card you can run `large-v3` and still have ~8 GB free. Use `large-v3` — it is the most accurate on accented English and you are not memory-constrained at inference time.

**The GPU budget that actually needs thought is W6, not speech.** A 450M-parameter SmolVLA fine-tune on 12 GB is workable but not roomy: bf16 weights + gradients + AdamW optimiser states alone come to roughly 7 GB before activations, and you are feeding two camera streams. Plan for **small batch size with gradient accumulation, gradient checkpointing, an 8-bit optimiser, and freezing the vision backbone** if it does not fit. Confirm this early (§12-A4) — discovering it in W7 is expensive.

**Log ASR latency separately** from planning and execution latency in §6. It is shared by both architectures so it does not bias the comparison, but the examiner will ask about end-to-end responsiveness and "the robot took 4 seconds" is a different answer from "2 s of that was speech recognition."

### 7.5 Shopping list

| Item | Priority | Approx. | Notes |
|------|----------|---------|-------|
| Wrist camera (RGB, USB) | **must** | S$50–120 | Global shutter preferred; rolling shutter smears during fast motion |
| Wrist camera mount/bracket | **must** | S$0–60 | Print or fabricate; must not foul the gripper or the wrist-3 joint |
| USB cable routing / strain relief | **must** | S$20 | A cable snagging on a joint during a demo will end the demo |
| SpaceMouse Compact | **strong** | S$200–500 | See §7.2 |
| Tray, glasses, sealable bottles, food colouring | **must** | S$40 | |
| Small drawer unit for Tier 3 | **must** | S$40–100 | Handle must be graspable by the 2F-85 |
| **Headset or lapel microphone** | **must** | S$30–80 | §7.4. Laptop mic will not survive a noisy lab |
| Footswitch for push-to-talk | nice | S$30 | Frees both hands during a demo; a keyboard key works fine |
| Wrist RGB-D (D405-class) | optional | S$400+ | Only if the lab pays. Nothing may depend on it |

---

## 8 · Workstreams (W-series) — the actual steps

Ordered by dependency, not by week. §9 places them in time.

### W1 · Repo reset for hardware *(no hardware needed — start today)*

1. `shared/helpers/rotations.py`: add `euler_to_rotvec` and `euler_to_R` with round-trip tests against `rotvec_to_euler`. Nothing that commands an orientation can be written without these.
3. Fix `shared/helpers/config.py::get_config(path)` — the module-level `_CONFIG` singleton silently ignores `path` after the first call, so `URController.__init__` looks configurable and is not. Either honour the path or drop the parameter.
4. Fix `architecture_b/demos/hdf5_store.py::episode_paths` — it returns `sorted(*.h5) + sorted(*.hdf5)`, sorted within each extension but not across them. Mixed extensions give silently wrong episode order, which corrupts a dataset in a way you will not notice.
5. `export_lerobot.py`: fail loudly on truncated or malformed episodes; assert reported fps matches the configured record rate.

### W2 · Architecture A software *(no hardware needed)*

1. **`architecture_a/skills.py`** — `pick(xyz, ...)`, `place(xyz, ...)`, `pour(source_xyz, target_xyz, tilt_profile)`, `open_drawer(handle_xyz)`. Written against the **shared controller interface only**. If a primitive touches a ur_rtde-specific attribute directly instead of going through the controller, it is wrong. Parameterise approach height, descent speed, grasp width, tilt angle and tilt rate — the planner will vary them.
   *Note:* `server.py` currently reaches into five private controller attributes. The primitives will want some of the same. Promote what they need to a public API rather than adding a sixth reach-through.
2. **Anthropic SDK + tool schemas** — hello-world call, then schemas matching the W2.1 signatures *exactly*, plus `ask_user(question, options)`. Record rate limits and per-call cost; they are a §6 metric.
3. **`architecture_a/planner.py`** — system prompt lists available skills plus detected objects with base-frame positions; model returns an ordered tool-call plan; **the parser validates against the schemas before dispatch** (reject unknown skills, out-of-workspace coordinates, malformed args). A plan that reaches the controller unvalidated is how you break an arm.
4. **Clarification loop** — when the planner emits `ask_user`, surface the question, block for input, append the answer to context, re-plan. This is Tier 2's entire mechanism and it is maybe 40 lines.

### W2b · Speech front-end *(no hardware needed — build alongside W2)*

Deliberately a **separate module with no dependency on either architecture**, per §7.4.

1. **`shared/speech.py`** — it takes a held key and gives you a transcript string. Push-to-talk capture (`sounddevice`) → `faster-whisper` → text. Nothing else. No cleanup, no normalisation, no LLM. Put the model choice behind config so a cloud provider can be swapped in as the §7.4 fallback without touching callers.
2. **Wire it as the shared entry point.** Both A's orchestrator and B's inference loop accept an instruction string; speech just supplies it. Keep a `--text` flag so you can drive either architecture from the keyboard when debugging — you do not want to be talking to your laptop while chasing a transform bug.
3. **Log every transcript** alongside the audio, timestamped. Needed for §6 and for diagnosing whether a failure was speech or robotics.
4. **Optional TTS** for spoken clarification (§7.4). Ship it if Tier 2 is working by W9; it is an afternoon.
5. **Write the paraphrase sets now** — 5–10 phrasings per task, split into a record-time set and a held-out evaluation set. This has to exist *before* W5 recording starts, or the H7 experiment is impossible after the fact.

### W3 · Camera and calibration *(hardware-gated)*

1. **Camera driver** — RGB, depth, and real intrinsics straight from the SDK. If W2's interfaces were drawn correctly this is the *only* file Track A needs for real images.
2. **Decide the calibration marker.** Deferred since July, still blocking. Take the ArUco/ChArUco board — a printed board is more accurate than a detector-found geometric feature and takes an afternoon. Stop deferring this.
3. **Collection script** — move the arm to 15–20 poses spread across the workspace, record `(TCP pose from get_state, marker centroid in camera)` at each.
4. **Solve and save** `T_base_cam` via the existing `solve_rigid_transform` + `save`. The Kabsch solver already exists and is tested; only the procedure is missing.
5. **Validate.** Command the arm to a measured physical point via the transform; report residual RMS in mm. **A calibration you did not validate numerically is a calibration you do not have**, and every downstream failure will be blamed on it.
   *Trap carried over from V4:* the arm base carries `quat="0 0 0 -1"`, so **base ≠ world**. A world-frame target fed through a base-frame transform looks plausible and is wrong.

### W4 · Architecture A end to end *(hardware-gated)*

1. Wire W2 + W3 into one entry point.
2. **Per-stage logging, built properly the first time.** Every failure tagged detection / depth / transform / planning / execution. This is not housekeeping — per-stage diagnosability is the entire argument for Architecture A, it is H6, and this log *is* the evidence in D4.
3. Close the loop: re-detect before each grasp instead of trusting the plan-time position. Required for Tier 3. Document the failure modes you hit; they are results.
4. Run Tier 0 to the T2 threshold.

### W5 · Demonstration capture *(hardware-gated, critical path)*

1. Teleop rig per §7.2.
2. **Lock the dataset conventions and write down the reasons** — action space (joint positions, matching SmolVLA's joint-space convention and W2.1's primitives), observation set (2 RGB views + state vector), resolution, control rate, **and the per-episode instruction string drawn from the W2b.5 paraphrase set, not a single fixed template**. **The action space must match what Architecture A's primitives command, or the two architectures are not solving the same problem and the comparison is unfair.** This is the single most important paragraph in the methodology chapter.
3. Hardware recorder: threaded, clock-gated state polling + synchronised frames.
4. Record. **Budget: ~50 episodes per task minimum.** Community evidence says workspace *density* matters more than raw count — 50 episodes spread over a 30 cm region failed where 75 over 10 cm succeeded. **Constrain the workspace for the first fine-tune**, then widen it. At ~1.5–2 min per episode including reset, 250 episodes is 8–10 hours of lab time before retakes. Budget 15.
5. Export to LeRobot; confirm the dataset loads and fps asserts.

### W6 · Architecture B training and inference

1. Fine-tune on Tier 0 first. Log config, seed, wall-clock, GPU-hours. **Keep the checkpoint and the exact dataset version together** — a checkpoint whose dataset you cannot identify is not a result.
2. **Inference loop** — checkpoint loads, observation in, action chunk out, `chunk_size = 50` matching the SmolVLA config.
3. **Safety envelope before the controller** — workspace bounds and velocity clamp on every predicted action. Non-negotiable. A VLA under a novel observation can emit a chunk that slams the arm.
4. **Async inference** so the arm is not idle during the forward pass; the reported gain is roughly 30% off completion time. Read the RTC paper (Black et al. 2025) **before** designing this, not after.
5. Re-train per tier as demonstrations land.

### W7 · Evaluation and write-up

1. Build the T5 harness against the §6 protocol.
2. Run both architectures across all attempted tiers. **Freeze results before writing** — do not tune anything after the numbers start going into the report.
3. Figures: success rate by tier and architecture; A's failure-stage breakdown; Tier 2 choice distribution for B; cost table.
4. D3 → D4 → demo rehearsal.

### W8 · Reading *(interleave as breaks — all still outstanding)*

- **RTC** (Black et al. 2025) — async inference and action chunking. Needed **before** W6.4.
- **OXE** (Padalkar et al. 2023) — action/observation formats, UR5 inclusion. Feeds the W5.2 action-space decision.
- SmolVLA — read and summarised already.

---

## 9 · Schedule to 10 Nov 2026

Week 1 = 10 Aug 2026. Recess week = 28 Sep.

| Week | Dates | Focus | Milestone |
|------|-------|-------|-----------|
| **W1** | 10–16 Aug | W1 repo reset. **Chase lab access + procurement (§12-A1, §12-A2)** | Sim quarantined; orders placed |
| **W2** | 17–23 Aug | W2.1 skills, W2.2 SDK + schemas. **W2b speech front-end + paraphrase sets** | Primitives callable in isolation; speaking gives you a transcript |
| **W3** | 24–30 Aug | W2.3 planner, W2.4 clarification loop | **Spoken command drives A end-to-end** against the sim fixture |
| **W4** | 31 Aug–6 Sep | W3 camera driver + calibration on the real cell | **T1 done — `T_base_cam` validated, RMS reported** |
| **W5** | 7–13 Sep | W4 orchestrator + per-stage logging on hardware | **T2 done — A passes Tier 0** |
| **W6** | 14–20 Sep | W5 teleop rig + dataset conventions. **D1 due 14 Sep** | **D1 submitted** |
| **W7** | 21–27 Sep | Record Tier 0 demos (50–75, constrained workspace). First fine-tune | **T3 partial** |
| **Recess** | 28 Sep–4 Oct | W6.2–6.4 inference loop, safety envelope, async. Buffer | **T4 done — B passes Tier 0** |
| **W8** | 5–11 Oct | Tier 1 pouring: `pour` primitive for A; record + fine-tune for B | Tier 1 both architectures |
| **W9** | 12–18 Oct | Tier 2 ambiguity (A dialogue; B choice-distribution experiment). Tier 3 drawer demos + re-planning | Tiers 2–3 both architectures |
| **W10** | 19–25 Oct | W7 full evaluation sweep, all tiers + Tier 4 unseen objects. **Freeze results** | **T5, T6 done** |
| **W11** | 26 Oct–1 Nov | Write D4 (40–60 pp). Figures. T7 repo consolidation | Full draft |
| **W12** | 2–8 Nov | D3 draft **to supervisor early** for feedback (submission stays 25 Mar 2027). Turnitin dry run. Record D2 video. Rehearse the demo end-to-end | Demo-ready; draft with supervisor |
| **W13** | 9–15 Nov | **Submit D2 interim + video, 10 Nov.** D4 finished and parked. Outline D6 slides | **D2 submitted; D4 complete** |

**Critical path:** lab access → W3 calibration → W5 demo capture → W6 training → W7 evaluation. Everything in Track A sits off this path, which is exactly why W1–W2 should be finished before the arm is available.

### After W13 — holding pattern to submission

| When | Action |
|------|--------|
| Dec 2026 – Mar 2027 | **Monthly:** re-run the full demo end-to-end, fix what has rotted, log it. Act on supervisor feedback on the draft |
| Jan–Feb 2027 | Optional extra trials to tighten the §6 numbers, or recover the highest descoped tier (§11) |
| **25 Mar 2027** | Submit D3 draft final report |
| **09 Apr 2027** | Submit D4 final report |
| **12–16 Apr 2027** | D5 demonstration — arrange the date with the examiner **well in advance**, not in April |
| **10–12 May 2027** | D6 oral presentation, 15 + 10 min |
| **19 May 2027** | D7 full text to DR-NTU + revised final report to supervisor |

---

## 10 · Simulation

Removed from the project on 11 Aug 2026. The MuJoCo substrate is gitignored, dropped from `pyproject.toml`, `mujoco` is no longer a dependency, and no live file references it. Every result in this FYP comes from hardware.


---

## 11 · Risk register and descope ladder

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Lab access slips past W4** | Eats the Nov completion target and the Dec–Mar buffer with it | §12-A1 this week. Every week of delay descends one rung of the ladder below. Official deadlines are not at risk until the buffer is gone |
| Hands-in-frame contaminates demos | Retrain from scratch, ~2 weeks lost | SpaceMouse ordered W1; inspect the first 5 episodes' frames before recording 200 |
| **Demos recorded with one templated instruction string** | H7 becomes untestable and B looks artificially bad against spoken input; unfixable without re-recording | Paraphrase sets written in **W2b.5, before W5 recording starts**. Not a later problem |
| Lab noise wrecks ASR | Trials fail for reasons unrelated to either architecture | Headset mic + push-to-talk; transcript logged per trial so speech failures are separable |
| Calibration residual too large to grasp reliably | A fails everywhere; B unaffected | Validate numerically in W4 and report RMS. If >5 mm, re-run with more poses before proceeding |
| B fails to converge on 50 episodes | No comparison | Constrain the workspace first (density beats count); widen only after the constrained set works |
| Spill damages equipment | Project-ending | Tray, tinted water, never real juice, arm speed capped during pour |
| **12 GB VRAM too tight for the SmolVLA fine-tune** | W6 stalls on the critical path | Test a short fine-tune run in **W2**, long before real data exists — batch size 1 on the 2 existing episodes is enough to find an OOM. Mitigations in §7.4. Do not discover this in W7 |
| GPU time insufficient | Training bottleneck | Confirm GPU access W1; SmolVLA is 450M params and fine-tunes on a single consumer GPU, but budget the hours |
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

**Speech input is not on this ladder.** The project is titled *"LLM-Based Interface for Robotic Systems"* and the official summary promises *"more intuitive human–robot interaction"* — spoken command is the most direct reading of both. A text-only submission invites the examiner to ask why the interface is a terminal. Rung 1 already includes it; W2b is two days of work and it is on no critical path, so build it early and stop worrying about it.

---

## 12 · Open actions

| ID | Action | Owner | By |
|----|--------|-------|-----|
| **A1** | **Confirm lab access date and out-of-hours access to Robotics I (S2-B6a-01).** Demo capture is multi-hour and will not fit inside office hours. Highest-priority item in this document | JJ | W1 |
| **A2** | Confirm who pays for the wrist camera and teleop device, then order (§7.4) | JJ | W1 |
| **A3** | Confirm the existing lab RGB-D model and its SDK, and whether a mount point for the fixed camera already exists | JJ | W1 |
| **A4** | Confirm GPU availability and hours for fine-tuning | JJ | W2 |
| **A5** | Decide the calibration marker (§8-W3.2). Deferred since July — decide it, do not defer again | JJ | W3 |
| **A6** | **Ask whether the project title/summary needs updating on StaffLink to reflect the widened scope — the tiered task suite *and* the spoken interface.** Changes are locked around Week 8 of the first semester and **examiners are allocated based on the title and summary** — an examiner picked for "pick and place with typed commands" will be assessing a project about spoken interaction, clarification dialogue and dynamic tasks | JJ | **W4 — hard deadline, not movable** |
| **A7** | Agree the D5 demonstration date with the examiner. Arrange it months ahead, not in April 2027 | JJ | Jan 2027 |
| **A8** | ~~Update `README.md`~~ and ~~fix `pyproject.toml` description~~, both done 11 Aug. `SESSION_HANDOVER.md`, `ursim_setup_reference.md`, the V3/V4 plans and the old vault architecture note were **deleted** 11 Aug rather than updated, so this plan is now the single source of truth. Nothing further outstanding | JJ | done |
| **A9** | Send Prof Cheah a short written progress update **every two weeks**. This was skipped through July. It is directly worth 8 marks | JJ | fortnightly |

---

## 13 · Mapping deliverables to marks

| Assessment item | Marks | What earns it here |
|-----------------|-------|--------------------|
| Project Plan/Strategy | 7 | D1, drawn from §3, §5 and §9 |
| Interim Report | 13 | D2 — by Nov the work is largely complete, so this reads as results rather than progress |
| Report writing, structure, references | 11 | Cite SmolVLA, RTC, OXE, OWLv2 properly; §8-W8 is not optional reading |
| Intro, theory, review, discussion, conclusion | 12 | The A-vs-B framing is inherently a comparative review; §5's hypotheses give the discussion its spine, and §11 gives you "limitations of current work" for free |
| Demonstration + oral | 17 | D5 shows both architectures live on the same cell — Tier 2 side by side (A asks, B guesses) is the moment that lands. Finishing in Nov means five months of rehearsal, which is the whole point |
| Initiative + regular supervisor updates | 8 | §12-A9, fortnightly. **This was skipped through July.** It is 8 marks for an email |
| **Design, implementation, data, results, complexity** | **24** | §6's protocol is a designed experiment with stated hypotheses and controlled layouts; T3 is real data collection; §6's cost table shows understanding of tool *limitations*. This is the block to over-serve |
| Ability, independence, extending ideas | 8 | The widened task suite beyond pick-and-place *is* the extension. Say so explicitly in the report |

**One framing note for D4.** The official summary promises *"a user interface based on large language models… enabling more intuitive human–robot interaction."* Speech is the visible half of that promise and the comparison is the intellectual half. Lead the report with the interface — the user talks, the robot acts — then use the A-vs-B comparison to answer *what has to sit behind that interface for it to work.* That reads as a project with a thesis, rather than two pipelines benchmarked against each other.

---

*Sources for the deliverable dates and specifications: `docs/fyp_deliverables/fyp_guidelines.pdf`, `fyp_deliverables.png`, `assessment_criteria_for_final_report.pdf`, `SuggestedFormatInterimReport.pdf`, `project_background.png`. Repo state verified against commit `87f0f22` on 11 Aug 2026.*
