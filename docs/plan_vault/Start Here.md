---
tags: [moc]
---

# Start Here

Navigation hub for the FYP plan. The authoritative document is `docs/fyp_plan.md` in the repo, and this vault is the working surface for tracking it.

Project A1005-261, *Modular vs. End-to-End Vision-Language-Action Models for Embodied Robotic Manipulation*. Supervisor A/P Cheah Chien Chern, Robotics I lab.

Title updated by Prof Cheah on 31 Aug 2026, from *Development of a Large Language Model-Based Interface for Robotic Systems*. The new one names the comparison directly, which is what the project argues.

## Summary

The user speaks a command, the UR5e does it, and the project compares two ways of making that happen: a modular LLM pipeline with scripted primitives, against a fine-tuned vision-language-action model.

## Where to look

#### 1. Planning

[[Schedule]] for the week by week route to Nov 2026.
[[Deliverables]] for what NTU wants, when, and where the marks are.
[[Open Actions]] for the things blocking everything else.
[[Flowchart]] for the dependency and runtime diagrams.

#### 2. Doing the Work

[[Workstreams]] for the build order and what state each part is in.
[[Tasks]] for what the robot has to accomplish.
[[Decisions]] for the choices made and why.

#### 3. Thinking
[[Hypotheses]] for what the project claims.
[[Evaluation Protocol]] for how those claims get tested.
[[Risks]] for what goes wrong and what to do about it.
[[Reading]] for the papers.

## Status, 31 Aug 2026

Week 4 of 13. **Nothing has run against hardware except speech.**

Both architectures are written. Architecture A is complete end to end, including perception, grounding, skills, planning, the clarification loop and per stage tracing. Architecture B's robot side is built on LeRobot's own `policy_server` and `robot_client` with a UR5e adapter. The evaluation harness exists. Speech works and drives Architecture A.

The calibration path is written and unrun: `check_camera.py`, `check_orientation.py`, `write_calibration.py` and `check_calibration.py`, in that order, and the order matters because each fails in a way that looks like the next one's problem.

There are **zero recorded episodes**, so nothing can be trained yet.

The one decision still open is the rotation format in [[Decisions#Action space]], and a ten second bench check settles it.

## How this vault is organised

Twelve files, flat, no folders. Each one is a subject rather than a note, so the D-notes live inside [[Deliverables]], the tiers inside [[Tasks]], and so on. It was 42 files across five folders until 31 Aug, and most of them were shorter than their own frontmatter.
