# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`docs/plan_vault/Decisions to Make.md`**: the project's existing decision record. Each decision is a `##` section with its reasoning. Read the sections that touch the area you are about to work in. This is the first place to look, because most project decisions (calibration, camera, teleoperation, detector queries) were recorded here before ADRs existed.
- **`GLOSSARY.md`** at the repo root.
- **`docs/adr/`**: ADRs for code-level design decisions that `/domain-modeling` records from now on.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## Where a new decision goes

- A **project** decision (scope, hardware, method, evaluation, anything the report must justify) goes in `docs/plan_vault/Decisions to Make.md` as a new `##` section, never as a new vault file. The vault has a strict house style: no em dashes or en dashes, and bullets never end in a full stop.
- A **code-level** design decision (a module seam, an interface, a data format) goes in `docs/adr/` as an ADR.
- Never record the same decision in both places. If an ADR depends on a vault decision, link to the vault section.

## File structure

Single-context repo:

```
/
├── GLOSSARY.md
├── docs/
│   ├── adr/
│   │   └── 0001-<slug>.md
│   └── plan_vault/
│       └── Decisions to Make.md
└── src/fyp/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag decision conflicts

If your output contradicts an existing ADR or a section of `Decisions to Make.md`, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
