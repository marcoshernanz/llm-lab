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
- then profiling,
- then multi-core execution on the current ecosystem baseline,
- then a best-possible long-run scaling pass on that multi-core baseline,
- and then hand off to the later systems rebuild.

## Status
As of 2026-04-06:
- `019` is complete as the first local FineWeb-Edu shard baseline,
- `020` is complete as the local FineWeb-Edu multi-shard baseline,
- `021` is complete as the TPU multi-shard baseline,
- `022` is complete as the first aggressive scaled TPU baseline,
- `023` is complete as the first self-describing scaled-run baseline,
- `024` is complete as the batch-size recovery pass,
- `024` selected `batch_size=128` as the default scaled SGD baseline,
- `025` is complete as the locked from-scratch SGD baseline,
- `026` is complete as the handwritten momentum-SGD baseline,
- `027` is complete as the handwritten Adam baseline,
- `028` is complete as the handwritten AdamW baseline,
- `029` is complete as the ecosystem-aligned baseline,
- `030` is complete as the first profiling pass on the single-device ecosystem baseline,
- `030` showed that steady-state training dominates wall-clock, while shard loading and subset evaluation are small enough that they do not justify a systems rewrite yet,
- `031` is complete as the first working multi-core JAX TPU baseline,
- `031` showed that the multi-core path trains correctly and that throughput scales strongly once per-device batch is increased enough to use the `v5e-8` slice productively,
- `032` is now the next milestone and focuses on one best-possible long run on TPU `v5e-8`,
- `032` should start from the full tokenized `sample-10BT` dataset, not the current `10`-shard local subset,
- phase 2 now has both a first-principles implementation path and a production-style implementation path to compare against each other.

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

Status:
- Complete via `experiments/025_tpu_fineweb_edu_sgd_baseline.py`.
- The locked SGD reference is `batch_size=128`, `learning_rate=0.1`, `train_steps=100000`.

### Milestone 026: SGD With Momentum
Track: Optimizers

Goal:
- Learn what momentum changes relative to plain SGD.

Why this is next:
- `025` established a clear plain-SGD reference, so the next learning step is to add exactly one new optimizer idea: velocity.
- Momentum is the smallest useful optimizer extension because it introduces optimizer state without yet introducing adaptive per-parameter scaling.

What stays fixed:
- Same scaled model shape.
- Same dataset and shard set.
- Same hardware target.
- Same artifact and subset-loss logging.
- Same default baseline run from `025`: `batch_size=128`, `learning_rate=0.1`, `train_steps=100000`.

What changes:
- Optimizer only.
- Add a velocity tree with the same structure as the parameter tree.
- Update parameters using momentum SGD instead of plain SGD.

Concrete work:
- Copy the `025` experiment into a `026` experiment scaffold.
- Add a minimal momentum-state initializer to the optimizer module.
- Implement the momentum update rule in repo code from first principles.
- Add one new momentum hyperparameter, likely `momentum=0.9`, while leaving the rest of the setup fixed.
- Run at least one smoke test, one shorter parity-style run, and one real long baseline run.

Exit criteria:
- You can explain the difference between plain SGD and momentum-SGD in this training regime.
- One logged `026` run can be compared directly against the locked `025` baseline.

Status:
- Complete via `experiments/026_tpu_fineweb_edu_sgd_momentum.py`.
- The locked momentum reference is `batch_size=128`, `learning_rate=0.1`, `momentum=0.9`, `train_steps=100000`.

### Milestone 027: Adam
Track: Optimizers

Goal:
- Compare Adam against the locked SGD family baselines.

Why this is next:
- `026` added optimizer state in the simplest possible way through velocity.
- Adam is the next useful step because it keeps the first-moment idea while adding adaptive per-parameter scaling through a second moment.

What stays fixed:
- Same scaled model shape.
- Same dataset and shard set.
- Same hardware target.
- Same artifact and subset-loss logging.
- Same default baseline run from `026`: `batch_size=128`, `train_steps=100000`.

What changes:
- Optimizer only.
- Replace the single momentum velocity tree with two Adam state trees:
  - first moment
  - second moment
- Add the Adam bias-correction step counter.

Concrete work:
- Copy the `026` experiment into a `027` experiment scaffold.
- Add minimal Adam-state initializers to the optimizer module.
- Implement Adam from first principles with:
  - first-moment updates
  - second-moment updates
  - bias correction
  - epsilon stabilization
- Start with an Adam-appropriate learning rate rather than reusing the SGD rate blindly.
- Run one smoke test and one real baseline run against the locked `025` and `026` references.

Exit criteria:
- You can explain where Adam helps, where it changes training behavior, and whether the difference is worth it in this regime.
- One logged `027` run can be compared cleanly against the locked `025` and `026` baselines.

Status:
- Complete via `experiments/027_tpu_fineweb_edu_adam.py`.
- The locked Adam reference is `batch_size=128`, `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, `train_steps=100000`.

### Milestone 028: AdamW
Track: Optimizers

Goal:
- Separate adaptive optimization from decoupled weight decay.

Why this is next:
- `027` established a handwritten Adam reference, so the next useful question is whether decoupled weight decay matters in this regime.
- AdamW changes one optimizer idea only: how shrinkage is applied relative to the adaptive update.

What stays fixed:
- Same scaled model shape.
- Same dataset and shard set.
- Same hardware target.
- Same artifact and subset-loss logging.
- Same default baseline run from `027`: `batch_size=128`, `train_steps=100000`.

What changes:
- Optimizer only.
- Keep the Adam first moment, second moment, and bias correction.
- Add decoupled weight decay as a separate parameter-shrink step rather than folding it into the gradient.

Concrete work:
- Copy the `027` experiment into a `028` experiment scaffold.
- Extend the optimizer module with a minimal AdamW update.
- Add one new weight-decay hyperparameter while leaving the rest of the Adam setup fixed.
- Run one smoke test and one real baseline run against the locked `027` Adam reference.

Exit criteria:
- You can compare Adam vs AdamW cleanly and say whether the distinction matters yet.
- One logged `028` run can be compared directly against the locked `027` Adam baseline.

Status:
- Complete via `experiments/028_tpu_fineweb_edu_adamw.py`.
- The locked AdamW reference is `batch_size=128`, `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, `weight_decay=0.01`, `train_steps=100000`.

### Milestone 029: Ecosystem Alignment Refactor
Track: Engineering

Goal:
- Refactor the learning-built training stack into a clean, professional baseline that leans on the JAX/Flax/Optax ecosystem wherever doing so improves clarity and maintainability.

Why this comes here:
- `025` through `028` were about understanding optimizers from first principles by implementing them directly.
- After that learning arc is complete, the next useful step is to contrast the hand-built path with the ecosystem-native path professionals would usually ship and maintain.
- This milestone is not about changing the training recipe again. It is about changing implementation style while preserving the learned concepts and keeping the repo minimal.

What stays fixed:
- Same scaled model shape.
- Same dataset and shard set.
- Same hardware target.
- Same artifact format and subset-loss logging.
- Same recent optimizer conclusions from `025` through `028`.

What changes:
- Implementation style only.
- Prefer `flax.nnx` or other Flax/JAX ecosystem building blocks over hand-rolled infrastructure when the ecosystem version is clearer and production-leaning.
- Prefer `optax` optimizers over handwritten optimizer implementations in the production-style baseline.
- Prefer cleaner ecosystem-native module boundaries over custom learning scaffolding where that scaffolding is no longer the thing being studied.

Concrete work:
- Add one new production-style experiment baseline after the handwritten optimizer arc.
- Replace handwritten optimizer code in that path with the matching `optax` implementation.
- Evaluate where model components should remain handwritten for learning value versus where ecosystem-native blocks are now the cleaner choice.
- Keep the script thin, readable, and pedagogical even while making it more professional.
- Document explicitly which custom pieces were intentionally retired and which still remain because they carry real learning value.
- Start from the locked `028` baseline and change implementation style, not the training target.

Exit criteria:
- One baseline run exists that uses the ecosystem to the greatest practical extent without turning the repo into a framework.
- The difference between the learning-first implementation style and the production-style implementation style is clear in the code.
- The resulting code is cleaner and more maintainable without losing the repo’s minimal learning-oriented character.

Status:
- Complete via `experiments/029_tpu_fineweb_edu_ecosystem_refactor.py`.
- The standard `models/transformer.py` now uses NNX attention, embeddings, layer norms, linear layers, tied embedding logits, Optax AdamW, and an Optax loss helper.
- The handwritten transformer path remains available under `models/transformer_manual.py` for earlier milestones.

### Milestone 030: Profiling First Pass
Track: Profiling

Goal:
- Profile only after there is a real performance question to answer.

Why this comes here:
- The attempted multi-core bring-up introduced implementation complexity before there was a clear measurement of where the current single-device ecosystem baseline is spending time.
- Before changing the execution model again, the next useful step is to measure where time is actually going in the current `029` path.

What to profile:
- compile time
- train-step time
- train-subset eval time
- validation-subset eval time
- sampling cost
- tokens per second
- obvious input-pipeline or host-overhead bottlenecks

What stays fixed:
- Same dataset.
- Same scaled model shape.
- Same chosen optimizer baseline.
- Same artifact format and subset-loss logging.

What changes:
- Measurement only.
- Add timing and profiling instrumentation around the current baseline.

Questions to answer:
- Which part of the current baseline dominates wall-clock time?
- Is the bottleneck in training, evaluation, sampling, compilation, or input handling?
- What is the most defensible next systems change after measurement?

Concrete work:
- Add targeted timing and profiling instrumentation to the current baseline or notebook workflow.
- Measure at least one representative run cleanly enough that the time breakdown is trustworthy.
- Write down the one or two biggest bottlenecks before changing execution strategy again.

Exit criteria:
- Profiling answers at least one concrete bottleneck question.
- The results change a real next decision.

Status:
- Complete via `experiments/030_tpu_fineweb_edu_profiling.py`.
- The first profiling pass showed that steady-state training, not shard loading or subset evaluation, dominates wall-clock on the current single-device ecosystem baseline.
- The measured bottleneck did not by itself require a systems rewrite, but the next milestone is still moving to explicit multi-core execution in order to study the execution model directly before the next scaling pass.

### Milestone 031: Multi-Core JAX TPU Baseline
Track: Hardware

Goal:
- Move from single-device TPU execution to real multi-core JAX execution on `v5e-8`.

Why this comes here:
- `030` already established the current single-device runtime behavior clearly enough that the next remaining systems question is the explicit multi-core execution model itself.
- The profiling results do not force a multi-core rewrite, but they also remove the main uncertainty about hidden input or evaluation bottlenecks.
- Doing the multi-core bring-up now keeps the next scaling pass grounded in the actual execution model you want to study.

What stays fixed:
- Same dataset.
- Same scaled model shape as the current ecosystem baseline, unless the multi-core bring-up exposes one narrow change that is required for correctness.
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

Concrete work:
- Add the simplest data-parallel execution path that the current baseline can support cleanly.
- Keep global batch semantics explicit and easy to inspect.
- Run one representative multi-core training baseline end to end.
- Compare throughput, artifact quality, and loss behavior against the single-device `029` and `030` references.

Exit criteria:
- One multi-core run completes end to end.
- Throughput improvement is measured clearly.
- You can explain the main conceptual changes required to go multi-core.

Rule:
- Prefer the simplest data-parallel implementation that the current learning goal justifies.

### Milestone 032: Best-Model Long-Run Scaling Pass
Track: Scaling

Goal:
- Build the strongest model and training setup that can run productively for about `10h` on TPU `v5e-8`.

Why this comes here:
- `031` should establish the execution model you actually want to use for the next serious scaling pass.
- Once the multi-core path is working, the next useful question is how much model and data the real hardware budget can support when you stop optimizing for short convenience runs and instead optimize for the best attainable result in one long run.
- Recent curves still suggest the current model may be small for the available runtime budget, while the widening train/validation gap suggests the current data budget is also too small.

What stays fixed:
- Same hardware target established in `031`.
- Same multi-core ecosystem baseline from `031` unless the execution results justify one narrow change first.
- Same optimizer family.
- Same artifact format and subset-loss logging.

What changes:
- Model size, train-data budget, and possibly schedule length within a long-run `10h` budget.
- Prefer increasing available train shards before or alongside model growth, rather than scaling model size against a too-small repeated dataset.

Questions to answer:
- With a roughly `10h` TPU `v5e-8` budget, what is the best model you can actually train on the new execution baseline?
- How much does using more FineWeb-Edu train shards reduce the train/validation gap?
- Given that long budget, is the limiting factor more about model capacity, data budget, optimization schedule, or some combination of all three?

Exit criteria:
- One long-run configuration is clearly the strongest practical baseline the repo can support on `v5e-8` without turning into an unbounded training project.
- You can justify the chosen model size, batch size, data budget, and schedule length as the best use of the `10h` window.
- The resulting run is strong enough to serve as the final phase-2 scaling reference before the later systems rebuild.

Concrete work:
- Build or stage the full tokenized `sample-10BT` shard set so the run is not limited to the current `10`-shard local subset.
- Start from the current multi-core baseline with the strongest observed throughput setting: `global_batch_size=1024`.
- Use a single chosen scaled model target for the one long run instead of a sweep, with the decision justified from the `031` benchmark results.
- Compare end-of-run train loss, validation subset loss, train/validation gap, tokens per second, total tokens seen, and overall sample quality.

## Track Summary
Tracks still exist, but they are secondary to milestones:
- Data bring-up: `019`, `020`
- Hardware: `021`
- Scaling: `022`, `024`
- Observability: `023`
- Optimizers: `025`, `026`, `027`, `028`
- Engineering: `029`
- Profiling: `030`
- Hardware: `031`
- Scaling: `032`

## Next
After `032`, the planned next phase is the systems rebuild described in [docs/phase_3_systems.md](./phase_3_systems.md).
The broader project thesis and decision rules live in [docs/project_direction.md](./project_direction.md).
The personal context behind those choices is summarized in [docs/personal_context.md](./personal_context.md).
Later worthwhile side projects, such as a Rust tokenizer, are listed in [docs/future_projects.md](./future_projects.md).

## Later
Only after the model, data path, and training loop are stable enough that lower-level performance work is grounded in real usage.

Later areas:
- better input pipelines,
- profiling-driven speedups,
- JAX/XLA behavior,
- and eventually lower-level kernels and CUDA-oriented work.
