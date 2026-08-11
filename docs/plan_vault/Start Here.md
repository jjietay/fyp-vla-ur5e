---
tags: [moc]
---

# Start Here

Navigation hub for the FYP plan. The authoritative document is `docs/fyp_plan.md` in the repo. This vault is the working surface for tracking it.

Project A1005-261, *Development of a Large Language Model-Based Interface for Robotic Systems*. Supervisor A/P Cheah Chien Chern, Robotics I lab.

## The one-sentence version

The user speaks a command, the UR5e does it, and the project compares two ways of making that happen: a modular LLM pipeline with scripted primitives, versus a fine-tuned vision-language-action model.

## Where to look

Planning and dates:

* [[Flowchart]] for the dependency and runtime diagrams
* [[Schedule]] for the week by week route to Nov 2026
* [[Deliverables Overview]] for what NTU actually wants and when
* [[Open Actions]] for the things blocking everything else

Doing the work:

* [[Workstreams Overview]] for the build order
* [[Task Tiers]] for what the robot has to accomplish
* [[Decisions Overview]] for choices that are made or still pending
* [[Detector Query Format]] for why the detector cannot be given a sentence

Thinking:

* [[Hypotheses]] for what the project claims
* [[Evaluation Protocol]] for how those claims get tested
* [[Risks]] for what goes wrong and what to do about it

## Status right now

Week 1, Aug 2026.

Architecture A is written end to end: perception, grounding, skills, planner, clarification loop and per stage tracing. The [[Evaluation Protocol|evaluation harness]] is built. None of it has run against hardware, because there is none yet.

Lab access is still unconfirmed and remains the blocker on everything downstream. The two decisions that will block work the moment access arrives, [[Calibration Marker]] and [[Action Space]], are both still open. See [[Open Actions]].
