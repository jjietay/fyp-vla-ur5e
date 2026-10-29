---
status: accepted
---

# The planner can look for objects by name with a look tool, and those queries stay for the rest of the run

Decided 29 Oct 2026. Grounding chose the queries once, from the instruction, and every later re-perception reused them. After "pour me a drink" and the answer "orange juice", the detector was never asked for the bottle, so the clarification in Tier 2 could not be acted on (problem 1 of the 7 Oct review, Section 4.1 of the interim report). The planner now has a `look(queries)` tool. Its queries join the run's queries, so later re-perception and verify search for them too, and the result says in words which were found. A look perceives non-strictly: a query it does not find is an answer for the planner, not a failed detection stage blamed on the trial.

## Consequences

The queries come from the planner's own reading of the instruction and the answer, never from a person, so A gains no information B lacks; B has no channel to ask or to name anything, and that asymmetry is the Tier 2 result. A look costs a planner turn inside `max_turns`. The query list only grows within a run, and grounding starts it afresh on the next trial.
