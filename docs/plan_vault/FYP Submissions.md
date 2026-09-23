---
tags: [overview]
---

# Deliverables

Everything NTU assesses, and where the marks are.

The distinction that matters: work is **complete** by Nov 2026, which is a personal deadline, and work is **submitted** on the official NTU dates, most of which fall in 2027. Only D1 and D2 are both finished and submitted in 2026.

| Deliverable | Submit on | Weight |
|---|---|---|
| [[FYP Submissions#D1 Project Plan\|D1 Project Plan]] | 14 Sep 2026 | 7 |
| [[FYP Submissions#D2 Interim Report and Video\|D2 Interim Report and Video]] | 10 Nov 2026 | 13 |
| [[FYP Submissions#D3 Draft Final Report\|D3 Draft Final Report]] | 25 Mar 2027 | part of 40 |
| [[FYP Submissions#D4 Final Report\|D4 Final Report]] | 09 Apr 2027 | part of 40 |
| [[FYP Submissions#D5 Demonstration\|D5 Demonstration]] | 12 to 16 Apr 2027 | part of 40 |
| [[FYP Submissions#D6 Oral Presentation\|D6 Oral Presentation]] | 10 to 12 May 2027 | part of 40 |
| [[FYP Submissions#D7 Library Submission\|D7 Library Submission]] | 19 May 2027 | none |

A further 40 marks sit outside the documents entirely, in the mapping at the bottom of this file.

---

## D1 Project Plan

**Submit 14 Sep 2026 to the supervisor. Worth 7 marks.**

A short plan and strategy report. Assessed on whether the research activities are planned with minimal supervision and whether the plan is clear and reasonable with proper background knowledge.

Required contents: the main objectives, background, the proposed way of carrying out the tasks, and a proposed weekly schedule **in chart form**, which is a stated requirement rather than a suggestion.

Almost all of it already exists and can be lifted. Objectives and background come from the scope statement in `docs/fyp_plan.md`, the approach section is [[Workstreams]], and the chart is [[Weekly Focus]] reformatted as a Gantt chart rather than a table.

Written 9 to 13 Sep. The master copy is the Google Doc, and `docs/FYP_A1005-261_Project_Plan_v5.docx` is version 5. Its schedule is the one re-baselined around the lab move, which [[Weekly Focus]] now matches.

---

## D2 Interim Report and Video

**Submit 10 Nov 2026 to the supervisor and examiner. Worth 13 marks, plus the video.**

The single most important date in 2026, and the one deliverable both finished and submitted this year.

- 5 to 10 A4 pages, which is short, so cutting matters more than writing
- Turnitin plagiarism check **before** submitting
- a separate video file, 2 to 3 minutes
- a form acknowledging any use of generative AI

Structure: introduction covering background, purpose, scope, objectives and timeline; work conducted, being what is done, the achievement and effort, and problems encountered; future work, being what is in progress and any changes to the original requirements; and an honest assessment of progress.

### The advantage of finishing early

By 10 Nov the evaluation is complete, so this report carries **results rather than progress**. Most interim reports say "I have built half a pipeline". This one can say "both architectures run and here is the comparison". Against a criterion that rewards content, understanding and logic, that is where the 13 marks are.

The video should show the robot working. Two minutes of a spoken command producing real motion is worth more than any slide, and it is recorded while the cell is already set up for evaluation.

---

## D3 Draft Final Report

**Submit 25 Mar 2027 to the supervisor. Draft finished by 2 Nov 2026.**

Written in weeks 11 and 12 of 2026, then handed to Prof Cheah early for feedback. Submission stays on the official date.

Why hand it over early rather than sitting on it: four months of supervisor feedback is worth more than four months of self editing, it counts as a regular progress update which is separately worth marks, and Turnitin surprises are cheap to fix in November and expensive in March.

Same content as D4, one revision behind.

---

## D4 Final Report

**Submit 09 Apr 2027 to the supervisor and examiner. Finished by 9 Nov 2026.**

### Hard formatting requirements

- main body 40 to 60 pages, excluding charts, diagrams, appendices, tables and references
- exceeding the limit can mean being asked to resubmit a condensed version
- Times New Roman 12, one and a half spacing
- margins 35 mm left, 30 mm top, bottom and right
- the same font throughout except when highlighting

### Framing

Lead with the interface, then use the comparison to answer what has to sit behind it. The official summary promises a user interface enabling more intuitive human robot interaction, so the report opens on the user speaking and the robot acting, not on two pipelines being benchmarked.

The high standard criterion explicitly asks for limitations of the current work to be highlighted, so [[Different Risk in Project]] and the failure analysis are marked content rather than padding.

### Content sources, by how much writing they save

- [[Evaluation of Architecture]] and its results tables
- [[Hypotheses for Comparison]], one subsection each
- the per stage failure logs from Architecture A
- [[Decisions to Make]], which becomes the methodology chapter

---

## D5 Demonstration

**12 to 16 Apr 2027, arranged directly with the examiner. Demo ready by Nov 2026.**

Compulsory, immediately after the final report goes in. The student arranges the date.

### The risk nobody plans for

A hardware demo that has not been run since November **will not work in April**. Cables shift, calibration drifts, a dependency updates, a camera enumerates on a different port. This is the main cost of finishing early, and it has one fix.

- [ ] re-run the full demo end to end in Dec 2026
- [ ] re-run in Jan 2027
- [ ] re-run in Feb 2027
- [ ] re-run in Mar 2027
- [ ] arrange the date with the examiner by Jan 2027, not in April

### What to show

The moment that lands is [[Tasks#Tier 2 Ambiguity]] run side by side. Say "pour me a drink" to Architecture A and it asks which one. Say it to Architecture B and it guesses. That contrast is the whole project in thirty seconds.

Keep a scripted running order and a fallback video of every tier, in case the arm misbehaves on the day.

---

## D6 Oral Presentation

**10 to 12 May 2027. 15 minutes presenting, 10 minutes of questions.**

Attended and assessed by both supervisor and examiner. Assessed on logical sequence, ability to highlight major outcomes, quality of explanation, effort in slide design, and answering questions. The high standard is answering *most* questions, so anticipate them.

Worth preparing for:

- why SmolVLA rather than a larger VLA
- why the demonstration count you chose, and what happens with more
- whether the comparison is fair given the architectures need different amounts of data
- what speech recognition contributes, given it is off the shelf
- what would change with a second arm, or a mobile base

Slides outlined in Nov 2026 while the work is fresh, then rehearsed from Mar 2027 on the same cadence as the demo.

---

## D7 Library Submission

**19 May 2027, one week after the oral.** Two things on the same date: the full text report submitted to the library through Digital Repository at NTU, and the revised final report incorporating anything raised at the oral, to the main supervisor only.

Clear the contents with the supervisor before submitting to the library, since the library copy is the permanent public record. Mechanical, and easy to forget once the oral is over.

---

## Where the marks actually are

From the EEE assessment criteria.

### Interim assessment, 20

| Item | Marks | Earned by |
|---|---|---|
| project plan and strategy | 7 | D1 |
| interim report | 13 | D2 |

### Report, demonstration and oral, 40

| Item | Marks | Earned by |
|---|---|---|
| writing, structure, references | 11 | cite the papers in [[Research Papers for Reading]] properly |
| introduction, theory, review, discussion, conclusion | 12 | [[Hypotheses for Comparison]] give the discussion its spine, [[Different Risk in Project]] supplies the limitations section |
| demonstration and oral | 17 | D5 and D6 |

### Approach, development and achievement, 40

| Item | Marks | Earned by |
|---|---|---|
| initiative and regular supervisor updates | 8 | a fortnightly email, tracked in [[To Do (Non-Technical)]] |
| **design, implementation, data, results, complexity** | **24** | [[Evaluation of Architecture]] |
| ability, independence, extending ideas | 8 | the widened task suite is the extension, say so explicitly |

### The block to over serve

The 24 mark item is the largest single line, and it explicitly rewards design of experiments, appropriate experimental technique, mastery of data collection, understanding of tools **and their limitations**, and new and interesting results.

That is a description of [[Evaluation of Architecture]]. Controlled layouts, stated hypotheses, held out phrasings, real data collection, and a cost table that shows where each architecture is weak.

### The cheapest 8 marks in the project

Regular supervisor updates. A short fortnightly email, which was skipped through July. It costs ten minutes and rewards nothing but consistency, and there is no way to reconstruct the trail in April.
