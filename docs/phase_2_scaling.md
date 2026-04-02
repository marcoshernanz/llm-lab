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
- then optimizer implementation and comparisons,
- then explicit multi-core execution and profiling,
- and only after that deeper training-recipe work.

## Status
As of 2026-04-02:
- `019` is complete as the first local FineWeb-Edu shard baseline,
- `020` is complete as the local FineWeb-Edu multi-shard baseline,
- `021` is complete as the TPU multi-shard baseline,
- `022` is complete as the first aggressive scaled TPU baseline,
- `023` is complete as the first self-describing scaled-run baseline,
- `024` is complete as the batch-size recovery pass,
- `024` selected `batch_size=128` as the default scaled SGD baseline,
- `025` now has a from-scratch SGD experiment path wired into the baseline script,
- phase 2 now has a stable enough baseline to start learning optimizers from first principles.

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

Status:
- Complete via `experiments/021_tpu_fineweb_edu_multi_shard.py`.

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

Status:
- Complete via `experiments/022_tpu_fineweb_edu_scaling_baseline.py`.

Note:
- In practice, the current `022` result scaled batch size, runtime, width, and depth together more aggressively than the original milestone wording intended. That was useful for learning what the TPU can do, but it means the next milestone should restore experimental discipline rather than continue compounding changes.

### Milestone 023: Observability And Run Hygiene
Track: Observability

Goal:
- Make scaled runs self-describing enough that the artifact directory, not notebook stdout, becomes the source of truth.

Why this is next:
- `022` produced the strongest result so far, but too much of the important context still lived in notebook logs and manual notes.
- Before more hardware or optimizer work, the repo needs clearer run bookkeeping.

Concrete work:
- Save `train_loss`, `train_subset_loss`, and `validation_subset_loss` in one canonical CSV for scaled runs.
- Save a small run metadata file next to the CSV/SVG/sample with at least:
  - script name
  - execution target
  - JAX device count
  - batch size
  - learning rate
  - train steps
  - context length
  - embedding dim
  - decoder depth
  - train shard count
  - train subset shard index
- Make notebook export steps consistent across Colab and Kaggle.
- Make the learning log easy to update from local artifacts without re-reading notebook output.

Exit criteria:
- A finished run can be understood from its artifact directory alone.
- Milestone logs no longer depend on notebook stdout for essential metrics.
- Artifact export from Colab and Kaggle is routine rather than ad hoc.

Implementation:
- `experiments/023_tpu_fineweb_edu_observability.py` keeps the scaled baseline behavior but saves `run_metadata.json` next to the CSV, SVG, and sample.
- `lib/run_artifacts.py` holds the reusable metadata, summary, and artifact-writing helpers so the experiment stays thin.

### Milestone 024: Batch-Size Recovery Pass
Track: Scaling

Goal:
- Recover a clean single-axis scaling result after the intentionally aggressive `022` run.

Why batch size:
- Batch size was the most important knob in the recent TPU runs.
- `022` strongly suggests the TPU likes bigger batches, but it does not tell you whether `128` is actually the best tradeoff.

What stays fixed:
- Same dataset and shard set.
- Same model shape as the current `022` baseline.
- Same optimizer family.
- Same learning-rate choice for the sweep.
- Same logging and artifact format from `023`.

What changes:
- Batch size only.

Planned sweep:
- `batch_size=32`
- `batch_size=64`
- `batch_size=128`

What to compare:
- `train_subset_loss`
- `validation_subset_loss`
- steps per second
- tokens per second
- stability of the curve

Exit criteria:
- You can justify one batch size as the default scaled SGD baseline.
- You can say whether the largest batch is actually helping optimization, or only helping throughput.

Status:
- Complete via `experiments/024_tpu_fineweb_edu_batch_size_sweep.py`.
- The current default scaled SGD baseline is `batch_size=128`.

### Milestone 025: From-Scratch SGD Baseline Lock-In
Track: Optimizers

Goal:
- Freeze one clear scaled SGD reference before comparing optimizers.
- Replace the current Optax-backed SGD path with a repo-owned SGD implementation so the optimizer becomes part of the learning surface.

Why this comes now:
- `024` already selected the default scaled SGD setup, so the next useful learning step is to make SGD explicit rather than treating it as a library black box.
- Optimizer comparisons will be easier to interpret if plain SGD is both behaviorally stable and implemented by hand in the repo.

What stays fixed:
- Same dataset and shard set.
- Same model shape as the `024` default baseline.
- Same hardware target.
- Same artifact and subset-loss logging.
- Same default `batch_size=128` reference point from `024`.

What changes:
- Optimizer implementation only.
- The run should move from `optax.sgd(...)` to a small repo-owned SGD implementation.

Concrete work:
- Add a minimal optimizer module that implements plain SGD from first principles.
- Wire the scaled baseline experiment to use that implementation.
- Run at least one smoke test and one real baseline run with the new SGD path.
- Record the resulting run as the canonical reference for `026`, `027`, and `028`.

Exit criteria:
- One scaled SGD run is the agreed reference point for later optimizer work.
- The repo no longer depends on Optax for the plain SGD baseline.
- The artifact format and hardware target are stable enough that optimizer differences are interpretable.

### Milestone 026: SGD With Momentum
Track: Optimizers

Goal:
- Learn what momentum changes relative to plain SGD.

What stays fixed:
- Same scaled model shape.
- Same dataset and shard set.
- Same hardware target.
- Same artifact and subset-loss logging.

Exit criteria:
- You can explain the difference between plain SGD and momentum-SGD in this training regime.

### Milestone 027: Adam
Track: Optimizers

Goal:
- Compare Adam against the locked SGD family baselines.

What stays fixed:
- Same scaled model shape.
- Same dataset and shard set.
- Same hardware target.
- Same artifact and subset-loss logging.

Exit criteria:
- You can explain where Adam helps, where it changes training behavior, and whether the difference is worth it in this regime.

### Milestone 028: AdamW
Track: Optimizers

Goal:
- Separate adaptive optimization from decoupled weight decay.

What stays fixed:
- Same scaled model shape.
- Same dataset and shard set.
- Same hardware target.
- Same artifact and subset-loss logging.

Exit criteria:
- You can compare Adam vs AdamW cleanly and say whether the distinction matters yet.

### Milestone 029: Multi-Core JAX TPU Baseline
Track: Hardware

Goal:
- Move from single-device TPU execution to explicit multi-core JAX execution on `v5e-8`.

Why this comes after optimizer work:
- First learn training behavior under one-device execution.
- Then change the execution model once the optimizer story is clearer.

What stays fixed:
- Same dataset.
- Same scaled model shape.
- Same chosen optimizer baseline.
- Same artifact format and subset-loss logging.

What changes:
- Execution model only.
- Per-device batch handling.
- Global batch semantics.
- JAX parallelism strategy.

Questions to answer:
- What speedup do you actually get from `1` device to `8` devices?
- Does the loss behavior stay comparable at the same global batch size?
- What new host/device friction appears?

Exit criteria:
- One multi-core run completes end to end.
- Throughput improvement is measured clearly.
- You can explain the main conceptual changes required to go multi-core.

### Milestone 030: Profiling First Pass
Track: Profiling

Goal:
- Profile only after there is a real performance question to answer.

What to profile:
- single-device vs multi-core compile time
- single-device vs multi-core step time
- host overhead
- eval cost
- tokens per second
- obvious input-pipeline bottlenecks

Exit criteria:
- Profiling answers at least one concrete bottleneck question.
- The results change a real next decision.

### Milestone 031: Training Recipe Improvements
Track: Training recipe

Goal:
- Study recipe choices only after scaling, observability, hardware execution, and optimizer baselines are stable enough to support it.

What to explore:
- gradient clipping,
- warmup,
- learning-rate decay,
- checkpointing,
- and restartable long-run logging.

Rule:
- One recipe variable at a time.

Exit criteria:
- You can tie at least one recipe choice to a concrete improvement or failure mode.

## Track Summary
Tracks still exist, but they are secondary to milestones:
- Data bring-up: `019`, `020`
- Hardware: `021`
- Scaling: `022`, `024`
- Observability: `023`
- Optimizers: `025`, `026`, `027`, `028`
- Hardware: `029`
- Profiling: `030`
- Training recipe: `031`

## Later
Only after the model, data path, and training loop are stable enough that lower-level performance work is grounded in real usage.

Later areas:
- better input pipelines,
- profiling-driven speedups,
- JAX/XLA behavior,
- and eventually lower-level kernels and CUDA-oriented work.
