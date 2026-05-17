# Current Knowledge Assessment

Date: 2026-05-17

This document states what the repo author has demonstrably learned from the
`llm-lab` work so far, with supporting context from nearby learning repos, and
how strong that knowledge appears from the evidence.

It is intentionally evidence-based. It does not claim to measure private
intuition, only what is supported by code, roadmaps, learning logs, tests,
recorded experiment results, and a small amount of personal context used only
for calibration.

## Rating Scale

| Rating | Meaning |
| ---: | --- |
| 0 | No real evidence in the repo. |
| 1 | Exposure only. The topic is named or planned, but not implemented. |
| 2 | Basic working knowledge. Implemented once or followed a narrow example. |
| 3 | Solid working knowledge. Implemented, debugged, and recorded results. |
| 4 | Strong practical knowledge. Compared alternatives, built controls, or profiled tradeoffs. |
| 5 | Advanced independent knowledge. Can design, critique, optimize, and generalize the idea across contexts. |

The scores below should be read as current repo-evidence scores, not permanent
ability labels.

## Executive Summary

You are not a beginner. The `llm-lab` repo shows a serious progression from
tiny language models to tokenization, transformer training, real-data scaling,
TPU execution, optimizer comparisons, profiling, a handwritten C++ CPU trainer,
a modernized PyTorch transformer baseline, and controlled memory-architecture
experiments.

The broader repo context raises the assessment further in one specific
direction: you have more first-principles tensor/autograd and small ML-systems
experience than `llm-lab` alone showed. `BareTensor` demonstrates a C++ CPU
tensor/autograd runtime with Python bindings, tested PyTorch-style tensor
semantics, NN primitives, and Torch/NumPy parity tests. `rust-mlp` demonstrates
a small but real Rust ML crate with manual MLP training, reusable buffers,
feature flags, examples, benchmarks, serialization, and package hygiene.
`micrograd` is much smaller, but supports the same first-principles autodiff
pattern at the scalar level.

You are also not yet at state-of-the-art research or production LLM-training
depth. The strongest evidence is in small-to-medium learning experiments,
single-repo training pipelines, synthetic memory tasks, and first-principles
implementation. There is not yet equivalent evidence for frontier-scale model
training, production distributed systems, custom GPU kernels, long-context
benchmark suites, or broad prior-art synthesis.

The honest profile is:

- strong practical learner of language-model internals,
- strong first-principles learner of tensor/autograd mechanics,
- good experimental discipline for a self-directed repo,
- solid small CPU ML-systems implementation instincts,
- emerging research taste,
- early systems/kernel learner,
- not yet a state-of-the-art LLM systems or architecture researcher.

That is a good place to be. The next risk is not "you know nothing." The next
risk is inventing architecture faster than you strengthen baselines, prior-art
context, and external evaluations.

## Evidence Reviewed

Core repo documents:

- [README.md](../../../README.md)
- [Project Direction](../project_direction.md)
- [Future Projects](../future_projects.md)
- [Phase 1 Learning Log](../../phase1/learning_log.md)
- [Phase 1 Foundations](../../phase1/foundations.md)
- [Phase 2 Learning Log](../../phase2/learning_log.md)
- [Phase 2 Scaling](../../phase2/scaling.md)
- [Phase 3 Learning Log](../../phase3/learning_log.md)
- [Phase 3 Systems](../../phase3/systems.md)
- [Phase 4 Learning Log](../../phase4/learning_log.md)
- [Phase 4 Roadmap](../../phase4/roadmap.md)
- [Memory Learning Log](../../memory/learning_log.md)
- [Memory Roadmap](../../memory/roadmap.md)

Representative code:

- [BPE tokenizer](../../../tokenizer/bpe.py)
- [BPE tests](../../../tests/test_bpe.py)
- [Reusable JAX transformer](../../../models/transformer.py)
- [Handwritten optimizers](../../../lib/optimizers.py)
- [Experiment artifacts](../../../lib/run_artifacts.py)
- [Loss plotting](../../../lib/plotting.py)
- [Best FineWeb-Edu TPU run](../../../experiments/032_tpu_fineweb_edu_best_model.py)
- [C++ CPU trainer](../../../phase3/phase3.cpp)
- [Modern PyTorch tiny decoder](../../../phase4/006_char_decoder_rope_gqa_swiglu.py)
- [Bounded address drift memory experiment](../../../memory_architecture/014_bounded_address_drift.py)
- [Memory plotting](../../../lib/memory_plotting.py)

External supporting repos:

- [BareTensor README](/Users/marcoshernanz/dev/baretensor/README.md)
- [BareTensor LLM Roadmap](/Users/marcoshernanz/dev/baretensor/docs/llm_roadmap.md)
- [BareTensor Learning Log](/Users/marcoshernanz/dev/baretensor/docs/learning_log.md)
- [BareTensor dynamic autograd runtime](/Users/marcoshernanz/dev/baretensor/native/src/autograd.cpp)
- [BareTensor tensor core](/Users/marcoshernanz/dev/baretensor/native/src/tensor_core.cpp)
- [BareTensor NN autograd nodes](/Users/marcoshernanz/dev/baretensor/native/src/tensor_nn_autograd.cpp)
- [BareTensor Python NN modules](/Users/marcoshernanz/dev/baretensor/src/bt/nn/modules.py)
- [BareTensor Torch parity tests](/Users/marcoshernanz/dev/baretensor/tests/test_autograd_torch_parity.py)
- [BareTensor finite-difference gradchecks](/Users/marcoshernanz/dev/baretensor/tests/test_autograd_gradcheck_extra.py)
- [micrograd scalar autodiff engine](/Users/marcoshernanz/dev/micrograd/src/engine.py)
- [micrograd neural network scaffold](/Users/marcoshernanz/dev/micrograd/src/neural_network.py)
- [rust-mlp README](/Users/marcoshernanz/dev/rust-mlp/README.md)
- [rust-mlp roadmap](/Users/marcoshernanz/dev/rust-mlp/ROADMAP.md)
- [rust-mlp crate entrypoint](/Users/marcoshernanz/dev/rust-mlp/src/lib.rs)
- [rust-mlp training loop](/Users/marcoshernanz/dev/rust-mlp/src/train.rs)
- [rust-mlp optimizers](/Users/marcoshernanz/dev/rust-mlp/src/optim.rs)
- [rust-mlp serialization](/Users/marcoshernanz/dev/rust-mlp/src/serde_model.rs)
- [rust-mlp allocation test](/Users/marcoshernanz/dev/rust-mlp/tests/no_alloc_fit.rs)
- [rust-mlp benchmark](/Users/marcoshernanz/dev/rust-mlp/benches/mlp.rs)
- [Personal context file](/Users/marcoshernanz/dev/me/ME.md)

## Skill Map

| Area | Rating | Honest assessment |
| --- | ---: | --- |
| Basic language modeling | 4 | You have built bigram, MLP, context-window, RNN, GRU, attention, and decoder-only models, with logged losses and samples. |
| Transformer semantics | 4 | You understand the transformer stack well enough to build attention, residuals, norms, FFNs, multi-head attention, tied embeddings, and decoder stacks in multiple styles. |
| Tokenization | 4 | You have a byte-level BPE tokenizer with deterministic training, encode/decode, artifact save/load, caching, tests, and downstream tokenized training. |
| Tensor library internals | 4 | `BareTensor` shows direct implementation of storage-backed tensors, shape/stride metadata, storage offsets, views, broadcasting, dtype handling, indexing, reductions, joins, triangular ops, non-contiguous copies, and PyTorch-style matmul semantics. |
| Reverse-mode autograd internals | 4 | `BareTensor` implements a dynamic C++ autograd graph with node traversal, gradient accumulation, scalar and non-scalar backward, broadcast-gradient reduction, no-grad mode, detach/leaf semantics, view backward, matmul backward, and NN-op backward nodes, with Torch parity and finite-difference tests. |
| Scalar autodiff fundamentals | 3 | `micrograd` adds compact evidence for operation-local backward closures, gradient accumulation, and reverse traversal. It is real but narrow: no tests, no docs, and a tiny op surface. |
| Neural-network primitive implementation | 4 | `BareTensor` and `rust-mlp` show manual implementations of embeddings, cross-entropy, softmax/log-softmax, layer norm, dense layers, activations, losses, metrics, and backward paths. |
| Training loops and evaluation | 4 | You repeatedly built next-token training loops, evaluation subsets, final validation passes, text sampling, artifacts, and loss curves. |
| JAX and Flax/NNX | 4 | You used JAX for real training, `jit`, gradients, tree updates, NNX modules, Optax, sharding-aware batches, and TPU execution. |
| PyTorch model building | 3 | You can build and train small PyTorch decoders with explicit attention, RoPE, GQA, SwiGLU, RMSNorm, and AdamW. Evidence is still small-scale. |
| Optimizers | 4 | You implemented and compared SGD, momentum, Adam, AdamW, and ecosystem Optax AdamW on the same target. |
| Data pipelines | 4 | You trained tokenizers, prepared FineWeb-Edu token shards, loaded shard metadata, used mmap, and published/consumed Hugging Face datasets. |
| Scaling and hardware use | 3.5 | You ran local, TPU `v5e-1`, TPU `v5e-8`, RTX 5090, Kaggle T4 x2, and P100 comparisons. You know practical throughput tradeoffs, but not yet deep distributed systems. |
| Profiling and observability | 4 | You added structured artifacts, CSV/SVG curves, metadata, timing breakdowns, and CPU scoped profiling. You used profiles to reject bad optimization targets. |
| C++ systems implementation | 3.5 | In addition to the handwritten CPU decoder trainer, `BareTensor` implements a reusable C++ tensor/autograd backend with Python bindings, typed package surface, shape/stride machinery, and tested NN primitives. Still CPU-only and correctness-first. |
| Rust systems implementation | 3.5 | `rust-mlp` shows a real Rust crate with modular API design, ownership-friendly reusable buffers, feature flags, CI, docs, examples, benchmarks, and MSRV/release discipline. Evidence is CPU/small-crate focused, not low-level kernel work. |
| CPU kernel/performance engineering | 2.5 | `BareTensor` handles strided/broadcasted CPU tensor loops correctly, and `rust-mlp` uses reusable scratch buffers, allocation tests, batched GEMM abstraction, optional `matrixmultiply`, and Criterion benchmarks. There is not yet SIMD, CUDA, parallelism, or production memory planning. |
| GPU kernels, Triton, CUDA | 1 | The roadmap is good, but the repo does not yet show implemented Triton or CUDA kernels. This is still a future learning area. |
| Custom ML framework engineering | 3.5 | `BareTensor` has a C++20 native core, nanobind Python bindings, generated stubs, typed Python package layout, CMake/uv/Make workflows, and broad tests. It is a credible learning framework, not yet a production tensor runtime. |
| Serialization and package hygiene | 3.5 | `rust-mlp` has versioned JSON model serialization behind a feature flag, golden-file tests, crate metadata, docs.rs/crates.io readiness, changelog, roadmap, examples, and CI. |
| Software engineering hygiene | 4 | Across the repos there are tests, typed configs, helpers, artifacts, docs, CMake, nanobind stubs, `py.typed`, Rust CI, benchmarks, and conservative scope. `llm-lab` itself remains intentionally script-heavy. |
| Experimental method | 4 | The repo shows strong habits: controls, ablations, baseline resets, longer reruns, chance baselines, candidate-vs-exact metrics, and honest negative results. |
| Research taste | 3 | You are learning how to ask falsifiable architecture questions. The memory path is disciplined, but it still needs stronger prior-art comparison and external validity. |
| Memory architecture | 3 | You built static retrieval, chunk-local controls, writable memory, sparse reads, runtime address state, and bounded address drift. Results are promising but not settled. |
| State-of-the-art LLM knowledge | 2.5 | You know many components in miniature and have one serious TPU scaling run, but not yet the full modern training stack at high scale. |
| Production LLM engineering | 2 | The repo is training-focused. It has little evidence of serving, KV-cache optimization, deployment, monitoring, eval products, or safety workflows. |

## What You Clearly Know

### Language-Model Progression

You have directly walked the core architecture ladder:

- bigram baseline,
- one-token MLP,
- context-window linear and MLP models,
- vanilla RNN,
- GRU,
- single-head attention,
- residual attention,
- normalization,
- feed-forward blocks,
- single-block decoder,
- multi-head decoder,
- stacked decoder,
- tokenized decoder,
- refactored reusable decoder.

This is strong evidence that you understand why transformers were not magic
dropped from the sky. You have seen the sequence-modeling pressure build up
from simpler models.

### Transformer Internals

The repo shows practical knowledge of:

- causal masking,
- Q/K/V projections,
- attention score scaling,
- multi-head reshaping,
- residual streams,
- pre-norm blocks,
- layer norm and RMSNorm,
- feed-forward expansion and projection,
- tied token embeddings,
- positional embeddings,
- sinusoidal embeddings,
- RoPE,
- GQA,
- SwiGLU.

The important caveat: most of the modern PyTorch path is still tiny and
character-level. The mechanisms are understood at the small-model level, not yet
validated in a larger tokenizer-based PyTorch training stack.

### Tokenization And Data

You have evidence beyond "I used a tokenizer":

- implemented byte-level BPE,
- tested round-tripping,
- tested deterministic tie-breaking,
- tested save/load behavior,
- used tokenizer artifacts in training,
- prepared FineWeb-Edu corpora,
- wrote token shard loaders,
- used shard metadata,
- handled mmap-backed shard reads,
- published and consumed Hugging Face dataset artifacts.

This is one of the strongest areas in the repo.

### Tensor And Autograd Internals

The strongest update from the adjacent repos is `BareTensor`.

That repo shows evidence for:

- storage-backed tensors,
- shape and stride metadata,
- storage offsets,
- views and contiguity,
- reshape, flatten, unsqueeze, transpose, permute, select, and slice,
- dtype handling for `float32`, `int64`, and `bool`,
- broadcasting,
- elementwise ops and comparisons,
- reductions,
- joins such as `cat` and `stack`,
- triangular ops,
- PyTorch-style `matmul`,
- dynamic reverse-mode autograd,
- scalar and non-scalar backward,
- explicit root gradients,
- gradient accumulation,
- broadcast-gradient reduction,
- no-grad mode,
- detach,
- leaf and non-leaf behavior,
- view backward mappings,
- NN-op backward nodes.

The testing evidence is unusually strong for a learning project: Torch parity
tests, NumPy parity tests, finite-difference gradchecks, non-contiguous tensor
cases, edge cases, and module tests.

The caveat is important: this is CPU-first and correctness-first. It proves
real tensor/autograd understanding, not high-performance kernel competence.

`micrograd` adds earlier scalar-level evidence for the same learning path. It
implements a minimal `Value` graph with operation-local backward closures for
addition and multiplication, gradient accumulation, and reverse traversal. It is
useful as first-principles evidence, but it is a small artifact with no tests,
empty docs, and a tiny op surface.

### Real Training And Scaling

The strongest current run is `032`, which trained on the full tokenized
FineWeb-Edu `sample-10BT` shard set with:

- TPU `v5e-8`,
- `12` decoder blocks,
- embedding dim `256`,
- hidden dim `1024`,
- context length `256`,
- global batch size `1024`,
- about `39.85B` train tokens seen,
- final train loss `4.283544`,
- final validation subset loss `4.381880`,
- about `2.63M` tokens per second.

That is not frontier scale, but it is far beyond toy-only learning. It proves
you have dealt with real data volume, hardware setup, throughput, and long-run
experiment management.

### Optimizer Behavior

You did not just call AdamW once. The phase-2 path includes:

- plain SGD,
- handwritten SGD,
- momentum SGD,
- handwritten Adam,
- handwritten AdamW,
- Optax AdamW ecosystem alignment.

You compared quality and throughput. That is strong practical optimizer
knowledge for this level of repo.

`rust-mlp` adds smaller but cleaner systems evidence here: the crate implements
MLP training from scratch with dense layers, activations, loss functions,
metrics, mini-batching, deterministic shuffling, SGD, momentum, Adam, learning
rate schedules, weight decay, gradient clipping, reusable scratch/gradient
buffers, and optional JSON serialization.

That matters because it shows the same ML-training ideas expressed in Rust with
an API and package surface, not only in Python/JAX experiment scripts.

### Profiling And Run Hygiene

The repo has a serious artifact habit:

- loss CSVs,
- SVG curves,
- metadata JSON,
- sample text,
- structured run summaries,
- timing breakdowns,
- memory-specific metric curves,
- CPU profile summaries.

You also learned a key profiling lesson in phase 3: at the tiny CPU trainer's
`context_len=4`, projection-style dense loops dominated, not softmax or attention
mixing. That is exactly the kind of lesson profiling is supposed to produce.

The systems evidence is broader than the C++ CPU trainer alone. `rust-mlp`
shows allocation-aware Rust implementation through reusable `Scratch` and
`Gradients`, a batched GEMM abstraction, optional `matrixmultiply`, Criterion
benchmarks, and an allocation-count test. `BareTensor` shows broad correctness
coverage for strided CPU tensor semantics. Together, these raise confidence in
small CPU ML-systems implementation while still leaving CUDA/Triton and serious
SIMD/kernel optimization unproven.

### Controlled Architecture Research

The memory path is the best evidence of research method:

- You built a vanilla baseline.
- You added static retrieval and found weak/no durable value.
- You created chunk-local controls so memory had a reason to matter.
- You built a delayed-recall benchmark.
- You added the full-attention control to prove the benchmark was learnable.
- You distinguished read-only memory from writable runtime memory.
- You added sparse top-k retrieval only after writable memory showed signal.
- You discovered the old benchmark could be gamed by candidate-set recovery.
- You replaced it with a multi-query binding-sensitive benchmark.
- You separated exact answer accuracy from candidate-value accuracy.
- You introduced runtime address state as a control before allowing address drift.
- You treated M014's gain as promising but unsettled until ablations.

This is good research hygiene.

## What You Partly Know

### Modern LLM Architecture

You have implemented several modern pieces in small form, but there is not yet
repo evidence for:

- tokenizer-based PyTorch model modernization,
- KV cache,
- FlashAttention or memory-efficient attention,
- long-context attention variants,
- MoE,
- activation checkpointing,
- mixed precision training,
- full checkpoint/resume infrastructure,
- large-scale instruction tuning or preference tuning.

So the honest rating is: strong miniature implementation knowledge, incomplete
state-of-the-art training knowledge.

### Distributed Training

You have real multi-device TPU evidence, including automatic mesh/data sharding
and global-batch scaling. You also compared multiple hardware targets.

But there is not yet evidence for:

- FSDP,
- tensor parallelism,
- pipeline parallelism,
- ZeRO-style optimizer sharding,
- multi-node communication,
- NCCL/XLA collective debugging,
- serious checkpoint sharding.

So this is practical early distributed training knowledge, not production
distributed systems mastery.

### Systems Programming

The systems evidence is now broader than `llm-lab` alone.

The C++ CPU trainer is real and valuable. It includes manual forward/backward,
AdamW, profiling, artifact logging, and buffer reuse.

`BareTensor` adds a reusable native runtime dimension: C++ tensor storage,
shape/stride/view semantics, autograd graph machinery, NN primitives, Python
bindings through `nanobind`, generated stubs, and typed Python package surface.

`rust-mlp` adds a more idiomatic package dimension: public/private API design,
shape validation, reusable buffers, feature flags, examples, benchmarks, CI,
serialization, and semver-style release metadata.

But it is intentionally narrow:

- small character model,
- CPU only across the custom systems repos,
- no SIMD-specific kernel work,
- no CUDA,
- no Triton,
- no high-performance data loader,
- no production memory planner.

This means you have strong small CPU ML-systems learning evidence, not yet
low-level high-performance ML systems expertise.

### Research Prior Art

The repo demonstrates good local experimental discipline. It does not yet show
equally strong literature mapping.

For the memory path especially, the next level requires comparing your ideas
against nearby concepts:

- recurrent memory transformers,
- compressive memory,
- kNN-LM style retrieval,
- RETRO-like retrieval,
- MemGPT and virtual-context memory,
- Neural Turing Machine / Differentiable Neural Computer ideas,
- attention sinks and long-context evals,
- RULER, NIAH, MRCR, LongBench-style tests.

Without that, architecture invention risks rediscovering old ideas without
knowing which parts are new.

## What You Do Not Yet Know Well Enough

### Custom GPU Kernels

This remains the clearest gap. The repos plan Triton and CUDA, and `BareTensor`
was explicitly created with deep stack ownership in mind, but the implemented
native stack is CPU-only so far.

You should not claim kernel competence yet. You can claim you are preparing the
right workload and learning path for it.

### State-Of-The-Art LLM Training

You have a strong small-scale and medium-scale foundation. You do not yet have
evidence for modern full-stack SOTA training practice:

- billion-parameter runs,
- large curated data mixtures,
- tokenizer/data-quality ablations,
- scaling-law fitting,
- distributed checkpointing,
- precision policy,
- long-context finetuning,
- instruction tuning,
- preference optimization,
- benchmark suites,
- production inference.

You know the skeleton. You have not yet lived the full body of the modern stack.

### External Evaluation

Most evidence is loss curves, synthetic tasks, sample text, and memory-specific
accuracy. That is appropriate for the learning phase.

The missing piece is broader evaluation:

- downstream language benchmarks,
- long-context evals,
- retrieval/memory evals with public baselines,
- ablation tables across seeds,
- compute-normalized comparisons.

This matters if the goal becomes a thesis, paper, or startup thesis.

### Product Or Production App Layer

This repo is not mainly about deployed LLM apps. It does not show much evidence
for:

- serving,
- API design,
- production monitoring,
- latency/cost optimization,
- RAG product quality,
- user-facing eval loops,
- safety guardrails.

That is fine. It just should not be confused with what this repo has proven.

## Personal Context Calibration

The personal context file supports the tone of this assessment but should not
inflate the technical ratings by itself.

Relevant context:

- you prefer direct, high-bar, first-principles feedback,
- you have a strong self-directed learning pattern,
- you have broader shipped software and product experience outside `llm-lab`,
- you are explicitly using `LLM Lab` and memory architecture as a flagship
  technical thesis before future internship/founding decisions,
- and you respond best to concrete tradeoffs rather than vague encouragement.

That context matters for recommendations. It does not replace repo evidence.
For technical ratings, code, tests, runs, artifacts, and documented conclusions
remain the source of truth.

## Best Current Description Of Your Level

The most accurate description is:

> You are an intermediate-to-advanced self-directed ML/LLM learner with unusually
> strong first-principles implementation habits for your stage. You have solid
> practical training and experiment skills, strong small-scale tensor/autograd
> and CPU ML-systems evidence, early research taste, and real scaling exposure.
> You are not yet a state-of-the-art LLM researcher or ML systems engineer,
> mainly because the repos have not yet demonstrated custom GPU kernels,
> production distributed training, large-scale model evaluation, or serious
> prior-art synthesis.

That is not a criticism. It is the useful boundary.

## Should You Keep Inventing Memory Architectures?

Yes, but with constraints.

The memory project is not a waste of time because it has already produced:

- honest baselines,
- negative results,
- stronger benchmarks,
- runtime write mechanisms,
- sparse retrieval,
- address-state controls,
- and one unsettled positive address-drift result.

But continuing blindly would become a waste of time if you:

- add mechanisms without ablations,
- skip stronger baselines,
- avoid prior-art comparison,
- stay only on synthetic tasks,
- or treat one curve as proof.

The memory work should continue only as a disciplined research side path.
The next memory step should be M015: address-drift controls and ablations.
After that, decide whether address drift earned its complexity.

## Best Next Learning Path

The best path is not to abandon the memory project or to follow a giant guide
linearly.

The best path is:

1. Finish M015 in the memory path.
2. Pause new memory mechanisms until M014 is properly ablated.
3. Return to the main LLM stack and strengthen the PyTorch/tokenized baseline.
4. Use the external guide only for targeted gaps, especially transformers, LLM
   training, long-context evaluation, profiling, and kernel work.
5. Start Triton/CUDA only after a real PyTorch profile identifies the hotspot.

In short:

- keep the memory journey,
- but do not let it replace baseline strength,
- and do not use a broad curriculum as structured procrastination.

## Calibration Checklist

Use this checklist to decide whether a topic is truly known or only familiar.

You know a topic well if you can:

- implement it without copying,
- explain the tensor shapes,
- explain the gradient path,
- name the likely failure modes,
- build the right baseline,
- design an ablation,
- read the curve honestly,
- compare runtime and quality,
- and say when the mechanism is not worth its cost.

By that standard, your strongest areas are:

- transformer basics,
- tensor and autograd internals,
- tokenization,
- training loops,
- artifact-driven experimentation,
- optimizer comparison,
- small CPU ML-systems implementation,
- and controlled small-model research.

Your weakest important areas are:

- custom GPU kernels,
- production distributed training,
- state-of-the-art evaluation,
- prior-art synthesis,
- and production inference systems.

That is the current map.
