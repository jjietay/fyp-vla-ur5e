---
tags: [overview]
---

# Task Tiers

What the robot has to accomplish, fixed **before** either pipeline is tuned. Changing the task set after tuning begins means fitting the benchmark to whichever pipeline happened to get built first.

The widening beyond pick and place is the scope change agreed with the supervisor in Aug 2026, and it is the thing that makes the project interesting rather than routine.

| Tier | Task | Role |
|---|---|---|
| [[Tier 0 Pick and Place]] | place that red cube into that metal tray | verification only |
| [[Tier 1 Pouring]] | pour the orange juice into the glass | hard to script motion |
| [[Tier 2 Ambiguity]] | pour me a drink | the clarification result |
| [[Tier 3 Drawer]] | put the snack in the drawer, which gets closed | dynamic world state |
| [[Tier 4 Generalisation]] | unseen objects and phrasings | transfer |

Ordering note for descoping: Tier 2 is cheaper than Tier 3 and produces a more distinctive result, so cut the drawer before cutting the ambiguity experiment. See [[Risks]].
