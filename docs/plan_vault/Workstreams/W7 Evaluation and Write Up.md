---
status: harness built, trials blocked on lab access
needs_lab: true
week: 10
---

# W7 Evaluation and Write Up

Where the project turns into a result.

## Steps

- [x] harness built in `src/fyp/evaluation/`, importing neither architecture, with a test enforcing that
- [ ] run both architectures across every attempted tier
- [ ] **freeze the results before writing**, and tune nothing afterwards
- [ ] figures: success rate by tier and architecture, Architecture A failure stage breakdown, Tier 2 choice distribution, cost table
- [ ] record the [[D2 Interim Report and Video]] video while the cell is still set up
- [ ] write [[D4 Final Report]]

## The discipline that matters

Once numbers start going into the report, stop tuning. Tuning after seeing results, then reporting the improved numbers, is the most common way an honest project becomes a dishonest one without anybody deciding to do that.

If something needs fixing after the freeze, re run **every** trial for both architectures, or report both sets separately and say why.

## Status 11 Aug 2026

The harness exists and was exercised end to end with fake agents, so the four results tables are known to render. Everything else here needs trials, which need the arm.

Worth remembering when the time comes: freeze the results before writing. Tuning after seeing numbers, then reporting the improved ones, is how an honest project becomes a dishonest one without anybody deciding to do that.
