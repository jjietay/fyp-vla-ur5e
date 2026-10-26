Status: needs-triage

# Spec: Architecture A as a plan, perceive, act loop with one skill per turn

Written 24 Oct 2026 from a design discussion. Nothing here is built. It changes how A behaves, so
it must be decided before the 20-trial Tier 0 run (T2), or Tier 0 must be run again afterwards.

## Problem Statement

Today A is a fixed pipeline with a planner at the end. The LLM planner only plans. It never
controls perception, and it never sees the image. Four problems follow from that.

1. **The vocabulary is fixed at the start.** `ground()` extracts noun phrases once from the first
   transcript and caches them. A clarification answer goes to the planner, but grounding and
   perception never run again with new words. Tier 2 example: "pour me a drink" grounds to
   `bottle`, `glass`, `cup`. OWLv2 returns two objects both named `bottle`. The user answers
   "orange juice", and the planner has no object it can match to that answer. It can only ask
   again, stop, or guess, and guessing is the behaviour H3 says A avoids.
2. **The planner may run several skills per turn.** One reply can hold pick then place, and the
   planner sees the result of the pick only after the place has run. The 21 Oct runs happened to
   go one skill per turn, but nothing enforces it.
3. **Verification is thin.** The camera check runs after `pick` and `place` only, and tests only
   the distance from the object to the TCP (`verify.tolerance_mm`, 50 mm). It never checks where
   a placed object landed. `pour`, `open_drawer` and `home` have no check. An object the camera
   cannot see counts as OK ("unverified"), so on 21 Oct a place the operator judged failed was
   recorded by A as OK.
4. **"Done" is the planner's word only.** A reply with no tool calls sets `outcome.ok = True`,
   even when the planner's own summary says it could not confirm the result, or when it refused
   because an object was out of reach.

Recent modular systems treat perception as a tool the LLM calls, act one step at a time, and
check each step (ALRM, arXiv 2601.19510, Tool-as-Policy mode; MALLVi, arXiv 2602.16898, with a
VLM (vision-language model) Reflector). A weaker A invites the examiner's objection that B only
beat a dated baseline.

## Solution

A becomes a loop: **plan, perceive, act**, one tool call per turn.

1. **`perceive(queries)` is a tool.** It runs today's stages 2 to 5 (capture, OWLv2 detection,
   SAM mask and depth to a base-frame point, reach check) on the noun phrases the planner passes.
   It returns the objects in reach with xyz and the names of those out of reach.
2. **Grounding is the planner's choice of `queries`, not a separate LLM call (option A).** The
   planner already reads the command, so a second call to the same model to extract words adds a
   call and nothing else. After a clarification the planner calls, for example,
   `perceive(["orange juice bottle"])`. ADR 0004's rules move into the `perceive` tool
   description: keep the user's distinguishing words, add none, include the destination, short
   noun phrases only. `validate_call` enforces the cap (`architecture_a.grounding.max_queries`, 8).
3. **The first tool call must be `perceive`.** Any other first call is rejected and returned to
   the model as an error, the ADR 0010 pattern. This guarantees ground, then perceive, then act.
4. **One tool call per turn.** Enforced at the API with
   `tool_choice: {"type": "auto", "disable_parallel_tool_use": true}` (confirm against the current
   Anthropic API docs before relying on it), and in `validate_call` as a backstop.
5. **Plan and act merge.** Each turn the model writes a one-line intent, then makes its single
   call. The intent is logged in the trace, so a trace reads as a sequence of "thought, call,
   result".
6. **Every motion skill is followed by its own automatic check.** The check runs in code, not on
   the model's request, so it cannot be skipped and costs no planning turn. Its result goes back
   as `OK:` or `FAILED:` with the reason, and **includes the fresh scene the check already
   captured**, so the planner always plans on the current table without having to ask.
7. **A failed check goes back to the planner, it does not end the run.** The planner retries or
   gives up within `max_turns`. Ending on the first failure would contradict ADR 0010 and would
   make Tier 3 (drawer reclosed mid-task) impossible for A by design.
8. **`max_turns` rises.** perceive, pick, place, done is already 4 turns. Set it in
   `config.yaml` (it is a tunable).
9. **Code keeps a small state, and sends it as "State now" in place of "Scene now".** It
   replaces `_held` and holds three things only:
   - **In gripper:** the ID of the held object, with no xyz. The gripper hides it from the
     camera, and the state already says where it is.
   - **Objects:** each detection with an ID that stays the same between `perceive` calls
     (`bottle_1`, `bottle_2`). A new detection takes the ID of the nearest old object within
     `verify.tolerance_mm`. **Match by position, not by query.** After "orange juice" the
     planner calls `perceive(["orange juice bottle"])`, and a match by query would give the
     same bottle a new ID. Use the query only to break a tie. The automatic check matches by
     ID too: today it matches by name (`o.query == name`), so with two `bottle` detections it
     can measure the wrong one.
   - **Gone:** each object no longer seen, as `bottle_1: not seen since turn 3`, with **no
     xyz**. Same rule as out-of-reach objects: a stale position must not be something the
     planner can act on, because in Tier 3 a person moves things during the task.
     `validate_call` refuses a skill aimed at a gone object until a new `perceive` finds it.

   The step results stay in the chat history only. They are already there as tool_results.
   The chat history is the only memory of a run. It is bounded by `max_turns`: about 500
   tokens a scene, so about 5k tokens at 10 turns. Do not summarise old scenes or build
   retrieval.

## Traceability (H6, H9)

- Every `perceive` call is logged with its exact `queries`, what each query found and its score,
  and what was out of reach. A query that found nothing is visible at once, so H9 (wrong noun
  phrase) stays attributable.
- The trace stage formerly called GROUNDING records "planner chose queries" on each `perceive`.
  The `Stage` enum is the failure taxonomy in the report, so any change to it is deliberate.
- `Outcome.queries` collects every phrase used across the run, so the harness keeps working.

## Open questions

- **Checks for `pour`, `open_drawer`, `home`.** What does success look like for each, from the
  camera and RTDE (Real-Time Data Exchange) alone? Not needed for Tier 0.
- **Unverified place.** Once the fingers open they cannot hide the object, so "not seen after a
  place" may need to become FAILED. A pick can stay unverified, because the gripper does hide it.
- **Where a placed object landed.** Today's check only asks "is it away from the gripper". Tier 0
  success is "fully inside the box", which the check does not test.
- **Should the planner see the image?** The planner model is multimodal. Sending the frame would
  let it tell two `bottle` detections apart by colour. It also changes what A is, so it needs its
  own ADR.
- **Skills take an object ID in place of xyz?** With stable IDs, `pick(object_id)` and
  `place(object_id)` could look up the position in code, so the planner cannot invent or
  mis-copy coordinates. It changes the tool schemas and what the planner controls, so it needs
  its own ADR. Decide it with the image question above.
- **`outcome.ok` on a refusal.** A planner that stops because an object is out of reach is
  recorded as `ok = True`. Harmless while success is the operator's y/N, misleading to anyone who
  reads `outcome.ok`.
- **Operator verdicts in the printout.** `run_architecture_a.py` prints "A REPORTED DONE" even
  when the operator answered N to a step. One line such as `operator: 1 of 2 steps judged failed`
  would remove the confusion without feeding anything back to A.
- **`check_joint_move` scope.** It runs only in `home`. Slides and the diagram should not say all
  skills are joint-checked.

## Schedule risk

This rewrites A's loop. The plan has the 20-trial Tier 0 run in W11 (26 Oct to 1 Nov) and D2 on
10 Nov 2026. Building and bench-testing this first very likely pushes T2 past W11. The options to
put to Prof Cheah:

1. Build it before T2, and D2 reports the new A on Tier 0.
2. Run T2 on today's A for D2, build the loop in the vacation, and repeat Tier 0 in Semester 2
   (results before the change are not comparable, as with ADR 0010).

## Needs an ADR

One decision record for "A is a one-tool-per-turn loop with perception as a tool and automatic
per-skill checks", and an update to ADR 0004 (grounding moves into the planner's `perceive`
call).
