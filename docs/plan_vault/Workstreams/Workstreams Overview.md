---
tags: [overview]
---

# Workstreams Overview

Build order, arranged by dependency rather than by date. [[Schedule]] places these in time.

The split that matters is what needs the lab and what does not:

* [[W1 Repo Reset]], [[W2 Architecture A Software]] and [[W2b Speech Front End]] need no hardware, so they should be finished before the arm is available
* [[W3 Camera and Calibration]] onward are all hardware gated

Critical path to a result: lab access, then [[W3 Camera and Calibration]], then [[W5 Demonstration Capture]], then [[W6 Architecture B Training]], then [[W7 Evaluation and Write Up]]. Everything in Architecture A sits off that path, which is exactly why it goes first.

| Workstream | Needs lab | Blocks |
|---|---|---|
| [[W1 Repo Reset]] | no | done |
| [[W2 Architecture A Software]] | no | code complete |
| [[W2b Speech Front End]] | no | paraphrase sets done, speech pending |
| [[W3 Camera and Calibration]] | yes | driver written, calibration pending |
| [[W4 Architecture A End to End]] | yes | code complete |
| [[W5 Demonstration Capture]] | yes | W6 |
| [[W6 Architecture B Training]] | partly | W7 |
| [[W7 Evaluation and Write Up]] | yes | harness built |

## Where things stand, 11 Aug 2026

Almost everything that does not need the lab is now written. What is left divides cleanly:

* **needs the arm and the camera**: calibration, demonstration capture, every trial, every number in the report
* **needs a decision first**: [[Calibration Marker]] and [[Action Space]], both still open, both blocking hardware work the moment access arrives
* **needs neither**: the speech front end, and Architecture B's inference loop

None of the written code has run against hardware. Treat every "code complete" above as unverified, because that is what it is.
