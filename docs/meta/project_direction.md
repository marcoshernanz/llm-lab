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
Phase 2 was about scaling, data, hardware, optimizer behavior, and profiling.
Phase 3 became a narrow handwritten CPU systems reference.
Phase 4 is now the active path and moves back toward the real modern workflow: PyTorch, real profiling, Triton, then raw CUDA/C++.

The point of phase 2 was to freeze a real target that later lower-level work could depend on.

That means phase 2 should end only after the following are reasonably clear:

- one stable decoder architecture,
- one stable data path,
- one stable optimizer family,
- one stable hardware target,
- one stable training budget,
- and at least one real profile of where the time goes.

## Current Direction

The current best path is not "keep rebuilding trainers by hand forever."
The better path is:

1. keep the small handwritten CPU trainer as a completed systems reference,
2. rebuild the target simply in PyTorch,
3. modernize it carefully,
4. profile the real PyTorch workload with production-style tools,
5. use Triton for the first custom-kernel layer,
6. then write raw CUDA/C++ kernels for measured hotspots.

That path stays aligned with the real modern workflow while still preserving the long-term kernel-learning goal.

## Learning Order Now

The high-level order should now be:

1. simple PyTorch semantic baseline,
2. modest modern-transformer upgrades,
3. production-style profiling,
4. Triton,
5. raw CUDA/C++,
6. then later more advanced kernel and distributed topics.

This matters because the repo should teach not only how kernels work, but also where kernels actually fit in a real stack.

In practical terms, the priority stack now is:

- PyTorch baseline clarity,
- profiling,
- Triton,
- raw CUDA/C++,
- then later FlashAttention-style work, deeper kernel specialization, and distributed systems.

## Decision Rules

When deciding whether something belongs in this repo, ask:

1. Does it increase semantic understanding of the training target?
2. Does it freeze an interface or a workload that later low-level work will depend on?
3. Is it narrow enough to reason about clearly?
4. Would skipping it make later Triton/CUDA work more confused?

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

The current active kernel-learning path is described in [phase4/roadmap.md](../phase4/roadmap.md).
The phase-3 handwritten systems reference remains documented in [phase3/systems.md](../phase3/systems.md).
Possible worthwhile later side projects are listed in [future_projects.md](./future_projects.md).
