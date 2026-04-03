# Future Projects

This document lists worthwhile later projects that are valuable, but not yet on the critical path.

These projects are worth doing only after the current phase has produced a stable enough target that the side project will not be invalidated immediately.

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
