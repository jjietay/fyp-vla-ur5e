---
tier: 4
status: not started
---

# Tier 4 Generalisation

Objects and instruction phrasings never seen in the demonstrations or the prompt examples.

## What is expected

Architecture A should transfer well. Its detector is open vocabulary, so a new object is a new text query and nothing needs retraining. Architecture B should degrade, since a new object is outside its training distribution and there is no mechanism to accommodate it short of collecting more data.

Two things get varied independently:

* **unseen objects**, testing visual and semantic transfer
* **unseen phrasings**, drawn from the held out paraphrase set in [[W2b Speech Front End]]

Keeping them separate matters. If both vary at once and B fails, you cannot say whether the object or the wording caused it.

## Cheapest tier to run

No new primitives, no new training, no new rig. Just different objects on the table and different words spoken. It is last on the descope ladder only because the tiers below it produce sharper results, not because it is expensive.
