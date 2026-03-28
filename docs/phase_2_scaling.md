# Phase 2: Scaling

This document defines the second learning phase of the repo.

Phase 1 ended with `experiments/018_decoder_refactor.py`, which established the first standardized tokenized decoder baseline.
Phase 2 starts from that baseline and keeps the focus on learning through controlled changes rather than jumping straight into broad architecture churn.
For the run history from this phase, see [docs/phase_2_learning_log.md](./phase_2_learning_log.md).

## Why This Phase Is Separate
Phase 1 was about reaching the first coherent decoder-only transformer through deliberately small learning steps.

That part has now done its job.

The next phase is different.
The goal is no longer to discover the basic architecture piece by piece.
The goal is to make the current transformer path easier to observe, easier to scale, and realistic enough that later choices about datasets, optimizers, profiling, and systems work become worth studying.

This phase guide is not a continuation of the old milestone numbering.
It is a new phase with a different emphasis:
- clearer experiment visibility,
- better data,
- new hardware targets,
- controlled scaling on the right hardware,
- and only then deeper training-recipe work.

## Status
As of 2026-03-28, phase 2 is empty.

Experiments in this phase:
- none yet

The first phase-2 experiment should take the `018` baseline from phase 1 and rerun it with:
- a better dataset,
- TPU `v5e-1` as the real training target,
- and as little architectural change as possible.

## Starting Baseline
The baseline inherited from phase 1 is:
- reusable transformer code under `models/`,
- a tokenized decoder experiment in `experiments/018_decoder_refactor.py`,
- pre-norm residual blocks,
- a final output normalization layer,
- tied token embedding / output projection instead of a separate LM head,
- and extracted helpers for setup, evaluation, plotting, and training-loop scaffolding.

The rule for the beginning of phase 2 is simple:
- preserve this architecture long enough to learn from better data and better hardware,
- and avoid changing multiple major axes at once.

## Global Rules
- Change one major axis at a time.
- Keep experiments runnable end to end.
- Keep a small local smoke-test path for correctness and debugging.
- Use TPU early for real runs once the dataset path is stable enough.
- Do not turn the repo into a framework.
- Do not start optimizer deep-dives before the scaled baseline is stable enough for optimizer differences to mean something.

## Opening Experiment
### Goal
Start phase 2 by validating that the `018` baseline can survive contact with a better dataset and the intended hardware target.

### What to change
- Replace the current tiny dataset with a clearly better language-modeling dataset.
- Run the experiment on TPU `v5e-1`.
- Keep the decoder architecture as close to `018` as possible.

### What to keep fixed
- The standardized `018` model path.
- The overall tokenized decoder training setup unless the dataset or TPU environment forces a small mechanical adjustment.
- The idea that this is still a baseline-establishing run, not yet a broad training-recipe rewrite.

### Why this is first
- A larger dataset makes scaling questions more real.
- TPU `v5e-1` is the right execution target for nontrivial runs.
- Keeping the model fixed makes the early phase-2 results easier to interpret.

### Exit criteria
- The repo can run the `018` baseline end to end on a clearly better dataset.
- The run works on TPU `v5e-1`.
- You can explain what broke, what stayed stable, and what the new bottlenecks are.

## After The Opening Experiment
### Track 1: Observability And Clean Experiment Outputs
- Make training behavior easy to see without cluttering experiment scripts.
- Keep plotting and artifact writing outside the experiment where possible.
- Prefer simple, readable loss histories and run outputs over elaborate tooling.

Why this matters:
- Better visibility is the highest-value improvement before scaling harder.
- It makes later dataset, optimizer, and hardware comparisons much easier to interpret.

Exit criteria:
- An experiment can save both training and validation loss curves with a small, readable call site.
- The plotting code and artifact-writing path live outside the experiment script where possible.

### Track 2: Controlled Scaling On TPU
- Scale sequence length, embedding size, decoder depth, runtime, batch size, or dataset size.
- Change only a small number of variables at once.
- Record what changed and what happened.

Exit criteria:
- There is at least one scaled TPU baseline that is clearly stronger or more informative than the current tiny runs.
- You know which scaling dimension is starting to matter most.

### Track 3: First Profiling Pass
- Study compile time vs steady-state step time.
- Measure steps per second, tokens per second, and obvious bottlenecks.
- Use profiling only when there is a concrete performance question to answer.

Why this comes later:
- Profiling is much more useful after at least one real TPU baseline exists.
- Earlier than this, timing and plotting are usually enough.

Exit criteria:
- You can use profiling to answer at least one real bottleneck question.
- Profiling output changes a concrete next decision.

### Track 4: Optimizers
- Study optimizers only after the scaled baseline is stable enough for differences to be interpretable.
- Suggested order:
  - SGD
  - SGD with momentum
  - Adam
  - AdamW

Why this order:
- It teaches optimizer ideas progressively instead of jumping straight to the standard answer.
- It keeps the learning value high.

Exit criteria:
- You can explain what each optimizer is doing differently.
- You have loss curves and run notes that make the comparison meaningful.

### Track 5: Training Recipe Improvements
- Study gradient clipping, warmup, learning-rate decay, weight decay, checkpointing, and repeatable run logging.
- Do this only after the optimizer work starts paying off.

Goal:
- Learn how training recipe choices interact with the now-scaled model and dataset.

## Later: Performance Track
Only after the model, data, and training loop are stable enough that performance work is grounded in real usage.

Areas to explore later:
- better input pipelines,
- profiling-driven speedups,
- JAX/XLA behavior,
- and eventually lower-level kernels and CUDA-oriented work.

Rule:
- correctness first,
- visibility second,
- scale third,
- profiling fourth,
- low-level optimization last.
