# Phase 2: Scaling

This document defines the second learning phase of the repo.

Phase 1 ended with `experiments/018_decoder_refactor.py`, which established the first standardized tokenized decoder baseline.
Phase 2 starts from that baseline and shifts the goal from discovering the architecture to making it usable on better data and better hardware.
For the run history from this phase, see [docs/phase_2_learning_log.md](./phase_2_learning_log.md).

## Why This Phase Is Separate
Phase 1 was about learning the model family piece by piece.

Phase 2 is different:
- the architecture is already good enough to study,
- the next learning value comes from data, hardware, scaling, and training behavior,
- and the work should now be organized around explicit milestones again, not loose themes.

The emphasis of phase 2 is:
- better data,
- cleaner experiment visibility,
- TPU-first execution for real runs,
- controlled scaling,
- then profiling,
- then optimizer comparisons,
- and only after that deeper training-recipe work.

## Status
As of 2026-03-29:
- `019` is complete as the first local FineWeb-Edu shard baseline,
- `020` is complete as the local FineWeb-Edu multi-shard baseline,
- `021` is the next milestone,
- phase 2 is now active rather than empty.

## Starting Baseline
The baseline inherited from phase 1 is:
- reusable transformer code under `models/`,
- a tokenized decoder experiment in `experiments/018_decoder_refactor.py`,
- pre-norm residual blocks,
- a final output normalization layer,
- tied token embedding / output projection instead of a separate LM head,
- and extracted helpers for setup, evaluation, plotting, and training-loop scaffolding.

The baseline added during early phase 2 is:
- `HuggingFaceFW/fineweb-edu` `sample-10BT`,
- a `16384`-vocab BPE tokenizer,
- `uint16` token shard storage,
- and the first one-shard local experiment in `019`.

## Global Rules
- Change one major axis at a time.
- Keep experiments runnable end to end.
- Keep a small local smoke-test path for correctness and debugging.
- Use TPU early for real runs once the local multi-shard path is stable.
- Do not turn the repo into a framework.
- Do not start optimizer comparisons before the scaled SGD baseline is stable enough for the differences to mean something.
- Keep the learning log aligned with milestones, not just with ad hoc runs.

## Data Setup
Phase 2 assumes the tokenizer and tokenized shards already exist.

### Tokenizer
Use `tokenizer/prepare_fineweb_edu_corpus.py` to build a capped local tokenizer corpus:

```bash
uv run python tokenizer/prepare_fineweb_edu_corpus.py \
  --dataset-config sample-10BT \
  --max-chars 100000000 \
  --output-path datasets/fineweb_edu/sample10bt_tokenizer_corpus.txt
```

Then train the tokenizer:

```bash
uv run python tokenizer/bpe.py \
  --data-path datasets/fineweb_edu/sample10bt_tokenizer_corpus.txt \
  --vocab-size 16384 \
  --output-path artifacts/tokenizers/fineweb_edu_sample10bt_bpe_16384.json
```

### Tokenized Shards
Use `tokenizer/tokenize_fineweb_edu.py` to build reusable token shards:

```bash
uv run python tokenizer/tokenize_fineweb_edu.py \
  --dataset-config sample-10BT \
  --tokenizer-path artifacts/tokenizers/fineweb_edu_sample10bt_bpe_16384.json \
  --output-dir datasets/fineweb_edu/sample10bt_bpe_16384 \
  --shard-tokens 10000000 \
  --validation-fraction 0.01 \
  --max-train-shards 10
```

When the vocab fits in `uint16`, shard storage uses `uint16` to keep disk usage down.

## Milestones
Each milestone belongs mainly to one track, but the roadmap is milestone-first.

### Milestone 019: Local FineWeb Single-Shard Baseline
Track: Data bring-up

Goal:
- Prove that the `018` architecture runs locally on real FineWeb token shards.

What changes:
- Move from Tiny Shakespeare text files to FineWeb-Edu token shards.
- Train on one train shard and evaluate on one validation shard.

What stays fixed:
- The standardized `018` model.
- SGD.
- Local execution.

Exit criteria:
- The experiment runs end to end.
- Loss decreases on real FineWeb tokens.
- Sampling works on continuation-only output.

Status:
- Complete via `experiments/019_fineweb_edu_shards.py`.

### Milestone 020: Local FineWeb Multi-Shard Baseline
Track: Data bring-up

Goal:
- Move from a one-shard sanity check to a real local multi-shard baseline.

What changes:
- Rotate across multiple train shards.
- Keep validation fixed on a small validation shard or subset.

What stays fixed:
- Same model.
- Same optimizer family.
- Local execution.

Exit criteria:
- A local run trains across multiple train shards without manual intervention.
- The run remains understandable and debuggable.

Questions to answer:
- Does the loss trend stay coherent when train data spans multiple shards?
- What new loading or bookkeeping friction appears?

### Milestone 021: TPU Multi-Shard Baseline
Track: Hardware + Data

Goal:
- Port the current local multi-shard baseline cleanly to TPU `v5e-1`.

What changes:
- Execution target moves from local CPU to TPU.

What stays fixed:
- Multi-shard train loading.
- Fixed validation shard.
- Same model family.
- Same optimizer family.

Exit criteria:
- TPU training works across multiple train shards.
- The data path is stable enough that hardware, not plumbing, becomes the main topic.
- You can explain the TPU-specific friction clearly.

Practical note:
- For Colab, prefer staging token shards into local `/content` storage before training rather than reading them directly from mounted Drive.

### Milestone 022: First Controlled Scaling Pass
Track: Scaling

Goal:
- Scale one major axis while keeping the rest of the setup stable.

What to vary:
- context length,
- embedding size,
- decoder depth,
- batch size,
- runtime,
- or number of train shards.

Rule:
- Change one major axis at a time.

Exit criteria:
- At least one scaled SGD baseline is clearly more informative than the local bring-up runs.
- You can say which scaling dimension matters most so far.

### Milestone 023: Observability And Run Artifacts
Track: Observability

Goal:
- Make phase-2 runs easier to compare and easier to learn from.

What to improve:
- loss curves,
- run metadata,
- artifact naming,
- and notes that connect experiment changes to outcomes.

Exit criteria:
- The learning log can compare phase-2 runs cleanly.
- An experiment’s outputs are enough to understand what changed and how it behaved.

### Milestone 024: Profiling First Pass
Track: Profiling

Goal:
- Use profiling only after there is a real performance question to answer.

What to study:
- compile time vs step time,
- train vs eval cost,
- steps per second,
- tokens per second,
- obvious host or device bottlenecks.

Exit criteria:
- Profiling answers at least one real bottleneck question.
- The output changes a concrete next decision.

### Milestone 025: SGD Baseline Lock-In
Track: Optimizers

Goal:
- Freeze a clear SGD baseline before comparing optimizers.

What changes:
- Nothing major architecturally.
- The emphasis is on documenting the stable SGD reference point clearly.

Why this milestone exists:
- Optimizer comparisons are only meaningful if the SGD reference is explicit in the log.

Exit criteria:
- You have a stable scaled SGD baseline with enough logging to compare against later optimizers.

### Milestone 026: SGD With Momentum
Track: Optimizers

Goal:
- Learn what momentum changes relative to plain SGD.

What stays fixed:
- Dataset subset,
- model shape,
- hardware target,
- and run bookkeeping.

Exit criteria:
- You can explain the behavioral difference relative to milestone `026`.

### Milestone 027: Adam
Track: Optimizers

Goal:
- Compare Adam against the now-established SGD family baselines.

What stays fixed:
- Same baseline conditions used in the earlier optimizer milestones as much as possible.

Exit criteria:
- You can explain where Adam helps, where it changes training behavior, and whether the difference is worth it in this regime.

### Milestone 028: AdamW
Track: Optimizers

Goal:
- Separate adaptive optimization from decoupled weight decay.

Why this milestone is optional but likely worth doing:
- AdamW is the standard modern reference point, so it is useful to know whether it matters here.

Exit criteria:
- You can compare Adam vs AdamW cleanly and say whether the distinction matters yet.

### Milestone 029: Training Recipe Improvements
Track: Training recipe

Goal:
- Study recipe choices only after the optimizer sequence starts paying off.

What to explore:
- gradient clipping,
- warmup,
- learning-rate decay,
- checkpointing,
- and repeatable run logging.

Rule:
- One recipe variable at a time.

Exit criteria:
- You can tie at least one recipe choice to a concrete improvement or failure mode.

## Track Summary
Tracks still exist, but they are secondary to milestones:
- Data bring-up: `019`, `020`
- Hardware: `021`
- Scaling: `022`
- Observability: `023`
- Profiling: `024`
- Optimizers: `025`, `026`, `027`, `028`
- Training recipe: `029`

## Later
Only after the model, data path, and training loop are stable enough that lower-level performance work is grounded in real usage.

Later areas:
- better input pipelines,
- profiling-driven speedups,
- JAX/XLA behavior,
- and eventually lower-level kernels and CUDA-oriented work.
