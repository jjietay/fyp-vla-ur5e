---
tags: [overview]
---

# Tasks

What the robot has to accomplish, fixed **before** either pipeline is tuned. Changing the task set after tuning begins means fitting the benchmark to whichever pipeline happened to get built first.

The widening beyond pick and place is the scope change agreed with the supervisor in Aug 2026, and it is the thing that makes the project interesting rather than routine.

| Tier | Task | Role |
|---|---|---|
| [[Tasks#Tier 0 Pick and Place\|Tier 0]] | place that red cube into that metal tray | verification only |
| [[Tasks#Tier 1 Pouring\|Tier 1]] | pour the orange juice into the glass | hard to script motion |
| [[Tasks#Tier 2 Ambiguity\|Tier 2]] | pour me a drink | the clarification result |
| [[Tasks#Tier 3 Drawer\|Tier 3]] | put the snack in the drawer, which gets closed | dynamic world state |
| [[Tasks#Tier 4 Generalisation\|Tier 4]] | unseen objects and phrasings | transfer |

Ordering note for descoping: Tier 2 is cheaper than Tier 3 and produces a more distinctive result, so cut the drawer before cutting the ambiguity experiment. See [[Risks]].

---

## Tier 0 Pick and Place

Spoken: *"Place that red cube into that metal tray."* Single unambiguous target, static scene, fixed tray. The deictic phrasing is deliberate.

**Verification, not a headline result.** It proves both pipelines are wired correctly and that they are commanding the same robot through the same interface. If a pipeline cannot do this, nothing above it means anything.

Targets: Architecture A at 70 percent over 20 trials, Architecture B at 50 percent. The thresholds differ because A is hand engineered and should be reliable on a task this simple, while B is learning from 50 demonstrations and will not be.

The first fine tune uses a **constrained workspace**, roughly 10 to 15 cm, rather than the full reachable area. Density beats raw episode count, and it widens only once the constrained version works.

Success: cube fully inside the tray, gripper clear, arm returned to home. Written into the harness source, not left to judgement, per [[Evaluation Protocol]].

---

## Tier 1 Pouring

Spoken: *"Pour the orange juice into the glass."* Both bottles present, the instruction names one, so there is nothing to clarify.

The primitive stops being a two pose pick. There is a grasp, a transport that must not spill, and a controlled tilt over a target. Architecture A needs a hand authored tilt profile, and Architecture B has to learn the tilt from demonstration. This is the cleanest test of who handles motion that is hard to script, which is one of the [[Hypotheses]].

### Rig and safety

- tinted water in both bottles, never real juice, since juice attracts pests and goes off in a shared lab
- everything happens inside a shallow tray, because a spill onto the controller or a camera ends the project
- bottles light, rigid, and within the gripper stroke, checked before buying anything
- arm speed capped during the pour

### Framing in the report

The motivating scenario is assisted feeding. **The robot pours into a glass on a table, never toward a person.** Label the feeding scenario as motivation and state why it was not physically demonstrated. That reads as maturity to an examiner rather than as a gap.

---

## Tier 2 Ambiguity

Spoken: *"Pour me a drink."* Both orange juice and water are on the table, so the instruction is under determined.

**This is the headline result of the project.**

Architecture A exposes `ask_user(question, options)` alongside the motion primitives. When detections are ambiguous relative to the instruction, the planner emits it, gets an answer, and re-plans. Adding this is a schema change, not new architecture.

Architecture B **has no channel to ask**. SmolVLA maps images, state and an instruction string to action chunks. There is no output head capable of producing a question. Faced with ambiguity it must still act.

### The experiment

Do not patch this asymmetry with an LLM wrapper. Measure it.

- [ ] run the ambiguous instruction at least 20 times against Architecture B
- [ ] record which bottle it picks each time
- [ ] report the distribution

If it picks orange juice 19 times out of 20 regardless of what the user wanted, that is a quantified demonstration that an end to end policy substitutes its training prior for the user's intent. One figure, and it is worth more than any success rate table.

Running A and B side by side on this instruction is the thirty seconds that explains the entire project, see [[Deliverables#D5 Demonstration]].

**Layout warning.** A hidden second candidate turns a clarification trial into Architecture A confidently picking the visible object, which would score a camera mounting choice as an architecture finding. See [[Decisions#Camera]].

---

## Tier 3 Drawer

Spoken: *"Put the snack in the drawer."* The drawer starts closed, the robot opens it, and **the human closes it again mid task**. The robot has to notice and re-open before placing.

This is the tier where the prediction flips, which is what makes it worth doing. Architecture A must detect the state change and re-plan, and plan-then-execute is bad at exactly this, which is why re-detection before each grasp is required work in W4. Architecture B is a closed loop visuomotor policy. If the demonstrations included re-closures, reactive recovery is what it is naturally good at. Expect B to win here.

### Being even handed

A comparison where one architecture wins everything is a comparison nobody believes. This tier is where Architecture B should look better, and reporting that honestly is what makes the Tier 2 result credible.

### Recording note

The demonstrations must **include** the perturbation. Record episodes where the drawer gets closed part way through and the operator re-opens it. If B only ever saw clean episodes it cannot recover, and the result measures the dataset instead of the architecture.

Hardware: a small drawer unit with a handle the gripper can actually grasp.

---

## Tier 4 Generalisation

Objects and instruction phrasings never seen in the demonstrations or the prompt examples.

Architecture A should transfer well. Its detector is open vocabulary, so a new object is a new text query and nothing needs retraining. Architecture B should degrade, since a new object is outside its training distribution and there is no mechanism to accommodate it short of collecting more data.

Two things get varied independently:

- **unseen objects**, testing visual and semantic transfer
- **unseen phrasings**, drawn from the held out paraphrase set

Keeping them separate matters. If both vary at once and B fails, you cannot say whether the object or the wording caused it.

Cheapest tier to run: no new primitives, no new training, no new rig, just different objects and different words. It is last on the descope ladder only because the tiers below it produce sharper results, not because it is expensive.
