---
tags: [reference]
---

# Risks

| Risk | Impact | What to do |
|---|---|---|
| lab access slips past week 4 | eats the Nov target and the buffer with it | chase it in week 1, see [[Open Actions]] |
| hands in frame contaminate demos | retrain from scratch, about 2 weeks | SpaceMouse ordered week 1, inspect the first 5 episodes |
| demos recorded with one templated instruction | phrasing hypothesis becomes untestable, unfixable without re recording | paraphrase sets written **before** recording starts |
| calibration residual too large to grasp reliably | A fails everywhere, B unaffected | validate numerically, re run with more poses if above 5 mm |
| B fails to converge on 50 episodes | no comparison at all | constrain the workspace first, widen later |
| 12 GB VRAM too tight for the fine tune | W6 stalls on the critical path | test a short run in week 2, long before real data exists |
| spill damages equipment | project ending | tray, tinted water, capped speed |
| lab noise wrecks speech recognition | trials fail for reasons unrelated to either architecture | headset mic, push to talk, transcript logged per trial |
| report underruns 40 pages | resubmission | start writing week 11, not week 13 |

## Descope ladder

Drop from the bottom. Each rung is still a complete piece of work.

1. Architecture A end to end, spoken input, Tiers 0 to 2. No training, no data collection, entirely within your control. **This alone is a passing FYP**
2. Architecture B on [[Tier 0 Pick and Place]] only
3. [[Tier 1 Pouring]] both architectures
4. [[Tier 2 Ambiguity]]
5. [[Tier 3 Drawer]]
6. [[Tier 4 Generalisation]]
7. spoken clarification replies, since text questions are an acceptable fallback
8. repo consolidation and documentation polish

Note the ordering. Tier 2 is cheaper than Tier 3 and produces a sharper result, so cut the drawer first.

**Speech input is not on this ladder.** It sits inside rung 1, because the project is titled an LLM based interface and a text only submission invites the question of why the interface is a terminal.
