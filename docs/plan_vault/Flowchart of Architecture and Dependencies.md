---
tags: [overview]
---

# Flowchart

Switch to reading view (Ctrl+E) to see these render. Mermaid is built into Obsidian, so nothing needs installing.

The graph view cannot show this. It is force directed and undirected, so it shows *that* two notes are linked but never which way the dependency runs. That is what these are for.

## Critical path

Everything that has to happen in order, and what can be done in parallel while waiting for the lab.

```mermaid
flowchart TD
    LAB([Lab access, from 22 Sep]):::done
    W2[W2 Architecture A Software]
    W2B[W2b Speech Front End]
    W3[W3 Camera and Calibration]:::doing
    W4[W4 Architecture A End to End]
    W5[W5 Demonstration Capture]:::crit
    W6[W6 Architecture B Training]:::crit
    W7[W7 Evaluation and Write Up]:::crit
    D4[D4 Final Report]:::deliver
    D2[D2 Interim Report 10 Nov]:::deliver

    W2 --> W4
    W2B --> W5
    LAB --> W3
    W3 --> W4
    W3 --> W5
    W4 --> W7
    W5 --> W6
    W6 --> W7
    W7 --> D4
    W7 --> D2

    classDef crit fill:#7f1d1d,stroke:#ef4444,color:#fff
    classDef blocker fill:#78350f,stroke:#f59e0b,color:#fff
    classDef doing fill:#14532d,stroke:#22c55e,color:#fff
    classDef deliver fill:#1e3a8a,stroke:#3b82f6,color:#fff
    classDef done fill:#374151,stroke:#9ca3af,color:#fff
```

Red is the critical path. Green is in progress. Grey is done.

Lab access was the amber blocker until about 22 Sep, when the arm became available after the lab move. The critical path now runs through W3 on the real cell, then demonstration capture, then training.

## Runtime, both architectures

What actually happens when you speak a command. Shared parts are the point: if these diverge, the comparison measures plumbing instead of architectures.

```mermaid
flowchart TD
    MIC([You speak]):::shared
    ASR[Whisper transcript]:::shared

    subgraph A[Architecture A, modular]
        DET[OWLv2 detect]
        DEPTH[Depth to 3D]
        TF[Camera to base frame]
        PLAN[LLM planner]
        ASK{Ambiguous?}
        SKILL[pick / place / pour]
    end

    subgraph B[Architecture B, SmolVLA]
        VLA[Images + state + string]
        CHUNK[Action chunk, 50 steps]
        SAFE[Safety envelope]
    end

    ARM([UR5e controller]):::shared

    MIC --> ASR
    ASR --> DET
    ASR --> VLA
    DET --> DEPTH --> TF --> PLAN
    PLAN --> ASK
    ASK -->|yes| ASKU[Ask the user] --> PLAN
    ASK -->|no| SKILL --> ARM
    VLA --> CHUNK --> SAFE --> ARM

    classDef shared fill:#1e3a8a,stroke:#3b82f6,color:#fff
```

Blue is shared by both. Note that **Architecture B has no path back to the user**: there is no arrow returning from B to the microphone, and that missing arrow is the entire [[Tasks#Tier 2 Ambiguity\|Tier 2 Ambiguity]] result.

## Decisions and what they block

```mermaid
flowchart LR
    MARK[Calibration marker: none]:::ok --> W3[W3 Calibration]
    CAM[Camera setup]:::ok --> W3
    TELE[Teleoperation method]:::ok --> W5[W5 Capture]
    ACT[Action space]:::ok --> W5
    ACT --> W2[W2 Primitives]
    SPEECH[Speech stack]:::ok --> W2B[W2b Speech]

    classDef ok fill:#14532d,stroke:#22c55e,color:#fff
    classDef pending fill:#78350f,stroke:#f59e0b,color:#fff
```

No decision is open. [[Decisions to Make#Action space\|Action Space]] was the last one, settled on 23 Sep when the rotation format was fixed, before any episode was recorded.

[[Decisions to Make#Calibration marker\|Calibration Marker]] closed on 12 Aug with no marker.

## Keeping these current

These are hand written, so they go stale if the plan changes and nobody updates them. Worth one minute after any scope change. The source of truth stays [[Weekly Focus]] and `docs/fyp_plan.md`.

## Other ways to draw things in Obsidian

* **Canvas** is a core plugin and already enabled: right click in the file explorer, New canvas, then drag notes in as live cards and draw arrows between them. Better than Mermaid when you want to rearrange by hand, worse when you want it version controlled and diffable
* **Excalidraw** is a community plugin for sketching, useful for figures in [[FYP Submissions#D4 Final Report\|D4 Final Report]]
* **Juggl** is a community plugin that replaces the graph view with one supporting directed layouts, if you specifically want the *whole vault* laid out rather than a curated diagram
