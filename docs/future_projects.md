# Future Projects

This document lists worthwhile later projects that are valuable, but not yet on the critical path.

These projects are worth doing only after the current phase has produced a stable enough target that the side project will not be invalidated immediately.

## GPU Fundamentals

### Why It Is Worthwhile

This is one of the highest-value next topics after the CPU trainer path is stable.

It teaches:

- memory hierarchy,
- warps and execution model,
- occupancy,
- memory coalescing,
- tiling intuition,
- and why many ML kernels are bottlenecked by memory movement rather than arithmetic.

### Why It Should Come Before Fancy Kernel Work

Without GPU fundamentals, later CUDA or Triton work easily turns into cargo-cult kernel tuning.

This topic should be treated as a prerequisite for serious CUDA work, not as an optional side interest.

## CUDA And Triton

### CUDA

CUDA is the core later systems topic once the CPU trainer and profiling work are solid.

It is worthwhile because it forces understanding of:

- launch configuration,
- shared memory use,
- synchronization,
- bandwidth limits,
- and the relationship between kernel design and model structure.

### Triton

Triton is also worthwhile, but later than basic CUDA understanding.

It is best treated as:

- a productive kernel-learning layer,
- a faster experimentation tool,
- and a bridge between high-level tensor thinking and low-level GPU implementation.

### Good Entry Condition

Start serious CUDA and Triton work once:

- one CPU trainer works end to end,
- one real profile exists,
- and the hot kernels are known rather than guessed.

## FlashAttention

### Why It Is Worthwhile

FlashAttention is an excellent later project because it teaches a genuinely important systems idea:

- same high-level attention semantics,
- much better memory behavior,
- and performance improvements driven by IO awareness rather than only arithmetic count.

### Why It Is Not Early

It should come only after:

- ordinary dense attention is understood,
- the baseline attention path has been implemented,
- and the attention bottleneck is real rather than hypothetical.

FlashAttention is high value, but only as a second-wave kernel topic.

## Distributed Programming

### Why It Is Worthwhile

Distributed programming matters for the later stages of ML systems work:

- data parallelism,
- sharding,
- collective communication,
- overlap of compute and communication,
- and multi-GPU or multi-node scaling.

### What Part Matters Most

The relevant version for this path is ML-systems distributed programming, especially:

- all-reduce,
- all-gather,
- reduce-scatter,
- data parallelism,
- parameter sharding,
- and communication bottlenecks.

### Why It Is Not Immediate

It should come after:

- single-device correctness,
- single-device performance understanding,
- and at least one real GPU training path.

Otherwise the communication layer is learned too far away from the computation it is supposed to support.

## Rust Tokenizer

### Why It Is Worthwhile

A Rust tokenizer is a very strong later project because it teaches:

- fast text processing,
- memory mapping,
- file formats,
- throughput-oriented engineering,
- and ownership/performance tradeoffs in a language well suited to systems work.

It also fits the long-term direction well:

- tokenizer work is real infrastructure,
- Rust is a language worth investing in,
- and data-path performance matters in practice.

### Why It Is Not Yet The Main Priority

Right now the higher-leverage work is:

- freezing the training target,
- understanding scaling behavior,
- and profiling the real model path.

If the target data format or workload is still moving, a tokenizer rewrite risks optimizing the wrong interface.

### Good Entry Condition

Start this once:

- the token shard format is stable,
- the downstream trainer is stable enough to consume it,
- and there is a real data-path bottleneck or a clear reason to own the tokenizer stack more deeply.

## High-Performance Data Loader

This is closely related to the Rust tokenizer but narrower.

Potential scope:

- mmap-backed shard readers,
- prefetching,
- shuffle/index strategies,
- threaded batch construction,
- and compact binary metadata.

This becomes worthwhile after profiling shows host/data overhead is meaningfully limiting training throughput.

## Standalone CUDA Kernel Experiments

These can be useful later, but only if tied to a real trainer bottleneck.

Good kernel experiments later:

- layer norm,
- softmax,
- fused attention pieces,
- optimizer update kernels,
- memory-bandwidth-sensitive fused operations.

Bad kernel experiments too early:

- isolated microbenchmarks with no stable training target,
- or kernels chosen because they are fashionable rather than measured bottlenecks.

## TileLang And Other Kernel DSLs

These are possible later topics, but they are lower priority than:

- GPU fundamentals,
- CUDA,
- Triton,
- and FlashAttention.

The rule is simple:

- do not learn a new kernel DSL before the underlying GPU execution model is already clear.

## Mixture Of Experts

MoE is worthwhile eventually, but it is not a near-term priority.

Why:

- it adds routing and systems complexity on top of already hard dense-training problems,
- it is less central than dense decoder training for the current learning path,
- and it is easy to spend a lot of time there without strengthening the core stack proportionally.

This should stay a later topic.

## Selective Reuse From Earlier Projects

Older projects may still be useful as source material for:

- tests,
- tensor-shape edge cases,
- API lessons,
- and implementation ideas.

But they should not automatically dictate the architecture of the next systems project.

## Rule For Future Projects

A future project becomes active only when it satisfies both:

1. it teaches something the current phase cannot teach as efficiently,
2. it depends on a target that is already stable enough to justify the work.

## Suggested Order

The current best order is:

1. Rust tokenizer and data path, later
2. GPU fundamentals, soon after the CPU trainer path is frozen
3. CUDA
4. Triton
5. FlashAttention
6. distributed ML systems work
7. MoE and other more advanced architecture/system combinations
