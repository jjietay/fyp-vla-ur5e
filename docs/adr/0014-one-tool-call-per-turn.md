---
status: accepted
---

# The planner makes one tool call per turn

Decided 29 Oct 2026. The loop of plan, perceive and act only works if each call's result and a fresh look at the table reach the planner before it chooses the next one. A reply with a pick and a place planned the place before the pick's result existed, which was problem 2 of the 7 Oct review, yet the code still accepted several calls per reply. Each request now sets `tool_choice: {"type": "auto", "disable_parallel_tool_use": true}`, which on Sonnet 5.5 means at most one call, and `_read` refuses any reply that still holds more than one, so the model is told to send one and tries again inside the turn limit.

## Consequences

A task takes one planner turn per tool plus a last turn to say it is done, so a pick and place needs three of the six turns, and a planner that retries a failed step runs out sooner. Tier 3's recovery (open, pick, put the snack down on seeing the drawer closed, open, pick, place, done) needs seven, so `max_turns` must rise before Tier 3, for every tier alike. The planner no longer needs to answer calls it did not run, so that bookkeeping is gone.
