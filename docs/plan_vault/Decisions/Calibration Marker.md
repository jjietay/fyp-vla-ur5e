---
status: pending
needed_by: W3
priority: high
---

# Calibration Marker

**Pending since July. This is the thing blocking [[W3 Camera and Calibration]], which blocks everything with a real object in it.**

## The options

* an ArUco or ChArUco board, printed and mounted on the gripper
* a distinctive geometric feature the existing detector can already find

## Recommendation

**Take the printed board.** It is more accurate than a detector found feature, it takes an afternoon, and marker detection is a solved problem with well tested library support. The geometric feature option trades accuracy for avoiding a printer.

## Why it keeps getting deferred

Because it is a decision rather than code, and code feels like progress. The collection script cannot be written until the marker is chosen, so deferring it has been quietly blocking the critical path since July.

Decide it. Do not defer again.
