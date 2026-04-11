# Personal Context

This document records the project-relevant personal context behind the repo.

It is not meant to be a full biography.
It exists so roadmap decisions stay aligned with the actual learning goal rather than drifting toward generic ML-project habits.

Related source material:

- Full personal profile: [ME.md](/Users/marcoshernanz/dev/me/ME.md)
- Earlier low-level ML project: [BareTensor README](/Users/marcoshernanz/dev/baretensor/README.md)

## Core Profile

The working assumptions behind this repo are:

- first-principles understanding is the highest-leverage learning style,
- visualization and deep conceptual clarity matter more than fast surface progress,
- and low-level systems work is the long-term target, but only once the semantic target is stable enough to deserve that effort.

## What Matters Most

The main goal is not just "learn ML."
It is to become strong enough to build and optimize real AI systems across the stack:

- model semantics,
- training behavior,
- hardware execution,
- profiling,
- C++ systems work,
- CUDA kernels,
- and later product or company building in deep-tech directions.

## What This Implies For The Repo

This repo should prefer:

- understanding over speed,
- narrow, inspectable milestones over broad abstractions,
- stable experimental targets over open-ended framework building,
- and later low-level ownership over permanent dependence on high-level tooling.

## What To Avoid

The repo should avoid turning into:

- a generic framework project,
- a long-lived abstraction playground,
- or a place where low-level implementation starts before the high-level target is clear.

## Practical Interpretation

If a choice appears between:

- building more generic infrastructure,
- or learning one more part of the actual training stack deeply,

the second option is usually the right one.

That is why phase 2 existed, why phase 3 became a narrow handwritten systems reference, and why the active next phase is now a PyTorch-to-kernel path rather than more broad handwritten trainer work.
