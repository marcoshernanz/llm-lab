# Phase 3: Systems Rebuild

This document describes the planned phase after phase 2.

Historical note:
- phase 3 is now best read as the completed handwritten systems-reference plan,
- while the active next learning path of the repo is phase 4 in [phase4/roadmap.md](../phase4/roadmap.md).

Phase 3 starts only after phase 2 has frozen a real target well enough that low-level systems work is grounded in actual training behavior rather than abstraction taste.

## Goal

Rebuild one narrow, real decoder-training stack in C++ from scratch, then add CUDA after CPU correctness is solid.

The goal is not to build a framework.
The goal is to deeply understand and optimize one real trainer end to end.

## Why This Phase Exists

Phase 2 should answer:

- what model matters,
- what data format matters,
- what optimizer matters,
- what runtime budget matters,
- and what the bottlenecks actually are.

Once those answers exist, high-level experimentation stops being the bottleneck.
At that point the right learning move is to go down the stack.

## Scope

Phase 3 should target one frozen trainer, not a general library.

That trainer should include:

- token shard loading,
- embedding lookup,
- layer norm,
- linear layers,
- causal self-attention,
- feed-forward blocks,
- cross-entropy loss,
- AdamW,
- training loop,
- artifact logging,
- and enough evaluation/sampling to prove parity.

## Non-Goals

Do not begin phase 3 by trying to build:

- a generic autograd framework,
- a Python-first tensor library,
- a broad operator surface,
- or a dependency-free competitor to PyTorch.

Those goals are too large and will dilute the learning value.

## Recommended Implementation Order

### Stage 1: Freeze The Reference

Before writing C++, freeze:

- one model configuration,
- one optimizer configuration,
- one token-shard format,
- one training budget,
- one artifact format,
- and one baseline run from phase 2.

If these are still moving, phase 3 has started too early.

### Stage 2: Numerical And Systems Prerequisites

Before serious CUDA work, make the following explicit study targets part of the phase:

- floating-point behavior,
- numerical stability in softmax / log-sum-exp / normalization,
- memory layout and locality,
- GPU execution basics,
- and the relationship between arithmetic work and memory movement.

This stage exists so later low-level implementation decisions are explained by mechanism rather than copied from other systems.

### Stage 3: C++ CPU Reference Trainer

Build the trainer on CPU first.

Requirements:

- correctness over speed,
- clean numerical parity checks against the frozen reference,
- minimal dependencies,
- no Python bindings at first,
- no attempt at a general-purpose public API.

Notes:

- A small PyTorch semantic reference is acceptable if it shortens the path to trustworthy parity checks.
- The main implementation target is still the C++ trainer, not a parallel high-level framework port.

Exit criteria:

- one end-to-end training run works on CPU,
- loss trends match the reference well enough to trust semantics,
- tensor shapes and memory layout are fully understood.

### Stage 4: C++ CPU Optimization

Only after correctness:

- improve memory layout,
- reduce unnecessary allocations,
- consider threading where it is clearly justified,
- and profile the real training step.

This stage exists to expose what truly matters before touching CUDA.

### Stage 5: CUDA Bring-Up

Add CUDA only for the real hotspots found by profiling.

Likely early CUDA targets:

- matmul path,
- layer norm,
- softmax,
- attention sub-steps,
- and later fused kernels if profiling shows a clear win.

Exit criteria:

- one GPU run works end to end,
- kernel-level speedups are tied to measured bottlenecks,
- CPU and GPU paths remain behaviorally comparable.

### Stage 6: Later Kernel And Distributed Work

Only after the GPU trainer is real should phase 3 branch into more advanced areas such as:

- Triton,
- FlashAttention-style IO-aware kernels,
- and distributed training / collectives / sharding.

These are valuable topics, but they should sit on top of a working trainer and a measured bottleneck, not replace them.

## Project Shape

The right shape is probably:

- one narrow C++ project or subproject,
- one executable trainer,
- one small config surface,
- one artifact format,
- one target model family.

This should feel closer to "build a trainer" than "build a framework."

## Dependency Philosophy

Dependency-free should be treated as a bias, not as a religion.

Good rule:

- avoid dependencies that hide the learning,
- allow dependencies that remove pure drudgery without erasing understanding.

Examples:

- Python bindings are not phase-3-critical,
- broad framework abstractions are not phase-3-critical,
- a small amount of build tooling is fine,
- and if one carefully chosen low-level dependency prevents wasting months on solved problems, it should be considered honestly.

## Framework Reference Rule

Phase 3 should not be framework-centered.

Good use of high-level frameworks in this phase:

- as a semantic reference,
- as a parity oracle,
- or as a quick way to confirm that a low-level implementation is still doing the right math.

Bad use of high-level frameworks in this phase:

- porting the whole training stack just to mirror the existing repo in a different framework,
- keeping two equally serious high-level codepaths alive,
- or delaying the C++ trainer because the framework port keeps getting more polished.

## Relationship To Older Projects

This phase should not be forced through an older codebase if that codebase is now a learning drag.

Reusing ideas, tests, or isolated utilities is good.
Reusing architecture only because it already exists is not a good reason.

## Success Condition

Phase 3 succeeds if, by the end, you can say:

- I understand one real decoder trainer end to end,
- I can explain both the semantic model and the systems bottlenecks,
- I can profile and optimize the important kernels,
- and I did not get lost trying to build a framework instead of a trainer.
