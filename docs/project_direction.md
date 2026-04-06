# Project Direction

This document explains what this repo is really for and how decisions should be made when the roadmap is ambiguous.

It exists because the highest-leverage learning is not "build the biggest thing possible."
It is:

- learn the right concepts in the right order,
- freeze a real target,
- then rebuild the important parts from first principles at a lower level.

## Core Thesis

The current bottleneck is not raw C++ or CUDA ability.
It is semantic understanding of modern language-model training:

- model structure,
- optimization behavior,
- scaling behavior,
- data budget,
- hardware execution,
- profiling,
- and the relationship between all of them.

That is why this repo exists.
It is the high-level semantic lab that makes later low-level systems work worth doing.

## What This Repo Optimizes For

- Understanding before abstraction.
- Minimal experiments over reusable frameworks.
- First-principles explanations over cookbook replication.
- Stable targets before low-level optimization.
- Learning value over local cleverness.

## What This Repo Is Not

This repo is not trying to become:

- a production ML framework,
- a long-term general-purpose tensor library,
- or a dependency-free reimplementation of the whole ecosystem.

Those are tempting goals, but they are too broad for the current learning stage.

## Current Phase Logic

Phase 1 was about architecture semantics.
Phase 2 is about scaling, data, hardware, optimizer behavior, and profiling.

The point of phase 2 is to freeze a real target that can later be rebuilt at a lower level.

That means phase 2 should end only after the following are reasonably clear:

- one stable decoder architecture,
- one stable data path,
- one stable optimizer family,
- one stable hardware target,
- one stable training budget,
- and at least one real profile of where the time goes.

## Post-Phase-2 Direction

After phase 2, the right move is not "rewrite everything."
The right move is:

1. freeze one exact target trainer,
2. rebuild that trainer in C++ on CPU,
3. get correctness and behavioral parity,
4. profile that implementation,
5. then add CUDA only where profiling justifies it.

That path preserves first-principles learning while keeping scope narrow enough to finish.

## Learning Order After Phase 2

The high-level order should be:

1. freeze one real training target,
2. rebuild that target on CPU in C++,
3. learn and profile the real CPU bottlenecks,
4. study GPU fundamentals,
5. add CUDA for the measured hotspots,
6. only then move into more advanced kernel and distributed topics.

This matters because the repo should teach things in the order they become real bottlenecks, not in the order they seem exciting.

In practical terms, the priority stack after phase 2 is:

- CPU trainer correctness,
- profiling,
- GPU fundamentals,
- CUDA,
- then later Triton, FlashAttention, and distributed training.

Topics such as MoE, exotic kernel DSLs, or broad framework design should stay later because they add complexity before the dense single-trainer path is fully understood.

## Decision Rules

When deciding whether something belongs in this repo, ask:

1. Does it increase semantic understanding of the training target?
2. Does it freeze an interface or a workload that later low-level work will depend on?
3. Is it narrow enough to reason about clearly?
4. Would skipping it make later C++/CUDA work more confused?

If the answer is mostly no, it probably does not belong here.

## Framework Rule

Framework choice should be treated as a tool choice, not as identity.

For this repo:

- use the framework that best exposes the current learning target,
- prefer the simplest path that makes the relevant concept inspectable,
- and do not keep a tool only because a lot of work has already gone into it.

That means:

- JAX is useful when studying scaling, compilation, and multi-device behavior,
- PyTorch is useful when studying imperative training behavior and when keeping the bridge to low-level systems work as short as possible,
- and later C++/CUDA work should not be forced through a high-level framework shape if that shape now fights the learning goal.

The important thing is not loyalty to JAX or PyTorch.
It is whether the tool makes the current bottleneck easier to see.

## Anti-Rules

Avoid:

- rebuilding broad infrastructure before the target is stable,
- adding flexibility before it is earned,
- optimizing a system that does not yet have a clear bottleneck,
- and keeping a project alive only because a lot of effort has already gone into it.

## Relationship To Later Projects

This repo should produce:

- the semantic reference implementation,
- the experimental evidence,
- the metrics,
- and the scope boundaries

for the later low-level systems phase.

That later phase is described in [docs/phase_3_systems.md](./phase_3_systems.md).
Possible worthwhile later side projects are listed in [docs/future_projects.md](./future_projects.md).
