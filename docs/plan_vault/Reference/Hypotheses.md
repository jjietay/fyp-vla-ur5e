---
tags: [reference]
---

# Hypotheses

Stated up front, before tuning. Each becomes a subsection of the discussion in [[D4 Final Report]].

| Number | Claim | Tested by |
|---|---|---|
| H1 | both architectures reach comparable success on static, well specified tasks | [[Tier 0 Pick and Place]] |
| H2 | B produces smoother, more successful constrained motion than hand authored primitives | [[Tier 1 Pouring]] |
| H3 | A resolves ambiguous instructions, B cannot and collapses to its training prior | [[Tier 2 Ambiguity]] |
| H4 | B recovers from mid task world state changes, open loop A does not, and closing A's loop costs latency | [[Tier 3 Drawer]] |
| H5 | A generalises to unseen objects far better than B for the same engineering effort | [[Tier 4 Generalisation]] |
| H6 | A's failures are attributable to a specific stage, B's are not attributable at all | every tier |
| H7 | A is robust to spoken phrasing that drifts from training wording, B degrades as it drifts | every tier |
| H8 | A grounds deictic reference against detected scene context, B cannot | Tiers 0 and 2 |

## The cheap ones

H6 costs nothing extra. It falls straight out of the per stage logging built in [[W4 Architecture A End to End]].

H7 is nearly as cheap. Run each tier with a **held out** set of spoken paraphrases never used during recording, then report success rate against phrasing distance.

H8 costs nothing at all. Phrase the commands deictically, as in "place *that* red cube", and the experiment runs itself.

## Being honest about H4

H4 predicts that Architecture B wins. Keeping it in, and reporting it if it holds, is what makes H3 believable. A comparison where one side wins everything reads as advocacy rather than research.
