---
tags: [reference]
---

# Evaluation Protocol

Fixed before tuning. Any change afterwards gets recorded with a reason in the report.

## Trials

* 20 per architecture per tier
* object start poses drawn from a written randomisation table, identical for both architectures, same layouts in the same order
* success defined per tier in one sentence, **in the harness source rather than in your head**

## Spoken commands

Scripted and shared. Write the 20 utterances in advance, half using training set phrasings and half held out paraphrases. Speak the same utterance to both architectures.

Do not improvise at the microphone. An unscripted phrasing that happens to suit one architecture is an uncontrolled variable.

**Record the transcript with every trial.** If the recogniser mis hears, that failure belongs to the speech front end rather than to either architecture, and you can only separate them if the text was kept.

## Per trial record

Architecture, tier, trial index, layout ID, spoken utterance, transcript, outcome, failure stage, wall clock duration, human interventions.

## Metrics

| Axis | A | B |
|---|---|---|
| success rate per tier | yes | yes |
| time to completion | yes | yes |
| failure attributable to a stage | yes | **no, and that is the finding** |
| ambiguity resolved | yes | expected no |
| robustness to held out phrasing | yes | expected to degrade |
| deictic reference resolved | yes | expected no |
| recovery from perturbation | measure | measure |
| data cost | zero demonstrations | episodes and lab hours |
| compute cost | zero GPU hours, log API spend | GPU hours per fine tune |
| engineering cost | primitives, prompts, schemas | dataset plumbing, training config |

## Log the money

Architecture A costs cents per command forever. Architecture B costs GPU hours once and nothing per command afterwards. That tradeoff is real, trivially measurable, and almost no student report includes it.

Claims being tested are in [[Hypotheses]].
