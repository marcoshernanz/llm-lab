# Memory Architecture Roadmap

This document defines the memory-architecture research path of the repo.

It is separate from phases 1 through 4 because the goal here is not mainly scaling, systems, or kernel work.
The goal is to study whether a language model can use an internal addressable memory in a way that is:

- mechanistically understandable,
- experimentally falsifiable,
- and ambitious enough to grow into a thesis or startup direction if the evidence becomes strong.

For the run history from this path, see [learning_log.md](./learning_log.md).

## Why This Path Is Separate

The current repo already has:

- a foundations path,
- a scaling path,
- a handwritten systems-reference path,
- and a framework-to-kernel path.

The memory-architecture path is different.
It asks a narrower research question:

- can we build a model that stores and retrieves latent information through an internal memory system,
- and can that system do something attention alone does not do well enough?

This path should therefore stay:

- small,
- explicit,
- benchmarked honestly,
- and organized around mechanism rather than hype.

## Core Goal

Build from a tiny transformer baseline toward a model with an internal latent-space addressable memory.

The desired end state is:

- the model emits memory queries into a learned latent address space,
- retrieves memory values by similarity or neighborhood in that space,
- writes useful information back into memory over time,
- and does so in a way that improves behavior on tasks where persistent internal memory should matter.

This is the path that is most relevant to:

- a serious undergraduate thesis,
- a credible architecture research direction,
- and a startup thesis around internal memory rather than plain context extension.

## Current Status

As of 2026-05-17:

- `001_char_decoder.py` is complete as the clean vanilla baseline.
- `002_memory_retrieval.py` is complete as the static retrieval scaffold on top of full-sequence attention.
- `003_chunk_local.py` is complete as the first honest chunk-local baseline.
- `004_chunk_memory_retrieval.py` is complete as the first chunk-local model with static memory retrieval.
- the longer `003L` and `004L` follow-up runs show that read-only static retrieval is not yet earning its cost in this setup.
- `005_memory_task_harness.py` is complete as the first chunk-local synthetic delayed-recall benchmark.
- `006_full_attention_task_harness.py` is complete as the matching full-attention control for that benchmark.
- `007_dense_latent_address_read.py` is complete as the first dense latent-address read path on the delayed-recall benchmark.
- `008_writable_fixed_address_memory.py` is complete as the first writable fixed-address memory model on the delayed-recall benchmark.
- `009_sparse_neighborhood_retrieval.py` is complete as the first sparse top-k read model over writable memory.
- `010_binding_sensitive_task_harness.py` is complete as the multi-query chunk-local binding baseline.
- `011_full_attention_binding_sensitive_task_harness.py` is complete as the matching multi-query full-attention control.
- `012_sparse_memory_binding_sensitive_task_harness.py` is complete as the sparse writable memory run on the multi-query binding benchmark.
- `013_runtime_address_state_control.py` is complete as the control that turns fixed addresses into per-example runtime address state without moving addresses yet.

That means the fixed-slot read-only line has done its job:

- it taught the mechanics of retrieval,
- it established a fair chunk-local control,
- and it gave a negative or at least weak result.

That also means the task-harness/control pair has now done its job:

- the delayed-recall task runs end to end under a chunk-local bottleneck,
- the chunk-local baseline stays near chance,
- and the full-attention control learns materially better on the same task.

The dense latent-address read milestone has also done its job:

- the address-space read path runs end to end,
- and read-only address memory does not solve a task that requires per-example runtime storage.

The writable fixed-address milestone gives the first strong positive result:

- runtime memory values are updated across chunks,
- answer accuracy jumps from the near-chance `005`/`007` level to about the full-attention-control level,
- and the result shows that the missing ingredient was runtime writing, not more static read capacity.

The sparse neighborhood milestone shows that the read path can be made local without losing the behavioral gain:

- each token reads only `8` of `64` memory slots,
- final answer accuracy stays essentially tied with dense writable memory and full attention,
- and the naive sparse implementation is slower at this scale, so the win is mechanistic rather than computational for now.

The old delayed-recall benchmark exposed a weakness:

- with `4` stored facts, guessing among stored candidate values gives `1 / 4 = 0.25` exact accuracy,
- the strongest full-attention and writable-memory runs clustered around that value,
- so the old benchmark could not separate true key-value binding from candidate-set recovery.

The multi-query binding benchmark is now the main synthetic benchmark:

- each example stores `8` key-value facts,
- every stored key is queried once,
- exact answer accuracy and candidate-value accuracy are reported separately,
- chunk-local stays near random value-token behavior,
- full attention learns exact binding above candidate guessing,
- and sparse writable memory beats the no-memory baseline while still trailing full attention.

The runtime address state control has also done its job:

- reads and writes now consume batched runtime addresses,
- the addresses are still copied from the learned base table and not updated,
- and final exact answer accuracy stays close to the fixed-address sparse-memory run (`0.2380` for `M-013` vs `0.2134` for `M-012`).

## What This Path Is Trying To Learn

There are really three different questions here:

1. Can a model read from a latent memory in a useful way?
2. Can it write to that memory in a stable and selective way?
3. Can the addressing mechanism scale beyond a tiny fixed slot table and become something more like an internal learned memory space?

The first question has been explored enough for now.
The second and third are the real research frontier for this path.

## Global Rules

- Optimize for learning and first-principles understanding, not for local benchmark wins.
- Keep each milestone narrow enough that one new mechanism is being tested at a time.
- Do not scale a mechanism before it has shown a real signal on a small controlled setup.
- Every natural-text experiment should have a matching synthetic or semi-synthetic memory task when possible.
- Separate read-path questions from write-path questions.
- Separate addressing questions from content-update questions.
- Prefer differentiable retrieval first, then sparse or hard retrieval once the dense version is understood.
- Keep honest controls. A memory model should be compared against the right no-memory baseline, not just against a weaker unrelated model.
- Treat runtime cost and memory cost as first-class results, not footnotes.
- Do not let “it probably works at scale” replace mechanism-level evidence.

## Working Hypothesis

The fixed-pool key-value memory idea was a useful starting point, but it is probably not the main research destination.

The stronger long-term direction is:

- a latent address space,
- explicit geometric or similarity-based retrieval,
- writable memory contents that persist across chunks or segments,
- and sparse neighborhood lookup once there is useful runtime memory to retrieve from.

In plain terms:

- Path A taught us how to think about fixed-slot recurrent memory.
- Path B is the more innovative direction and the one most worth pursuing now.

## Milestones

The first thirteen milestones already exist in code and are part of the roadmap.
The later milestones move toward latent-space addressable memory in small steps.

### Milestone 001: Vanilla Decoder Baseline
Track: Baseline

Goal:
- Establish the clean no-memory decoder baseline for the memory path.

What changes:
- Nothing memory-specific yet.
- This is the reference point for later comparisons.

What stays fixed:
- TinyStories,
- small model size,
- standard next-token loss.

Status:
- Complete via `memory_architecture/001_char_decoder.py`.

### Milestone 002: Static Retrieval Scaffold
Track: Retrieval mechanics

Goal:
- Prove that a separate retrieval branch can be wired into the decoder cleanly.

What changes:
- Add a shared static key-value memory bank.
- Let tokens query that bank.

What stays fixed:
- Full-sequence self-attention.
- No chunk bottleneck yet.
- No writes yet.

Status:
- Complete via `memory_architecture/002_memory_retrieval.py`.

Main lesson:
- Retrieval mechanics alone are not enough when full token-token attention already solves the task.

### Milestone 003: Chunk-Local Baseline
Track: Honest control

Goal:
- Create the first setting where memory could actually matter.

What changes:
- Replace full-sequence self-attention with chunk-local self-attention.

What stays fixed:
- Same tiny model family.
- Same language-model objective.
- No memory branch yet.

Status:
- Complete via `memory_architecture/003_chunk_local.py`.

Main lesson:
- Removing cross-chunk attention creates a real performance gap that later memory models should try to recover.

### Milestone 004: Chunk-Local Static Retrieval
Track: Retrieval under bottleneck

Goal:
- Test whether read-only static memory helps once the chunk bottleneck is real.

What changes:
- Add static memory retrieval to the chunk-local model.

What stays fixed:
- Same chunk-local attention pattern.
- No writes yet.

Status:
- Complete via `memory_architecture/004_chunk_memory_retrieval.py`.

Main lesson:
- Over longer runs, static read-only retrieval does not clearly beat the chunk-local baseline and does not justify its cost.

### Milestone 005: Memory Task Harness
Track: Evaluation

Goal:
- Build tiny tasks where internal memory should matter unambiguously.

Candidate tasks:
- associative recall,
- delayed key-value lookup across chunks,
- repeated entity-state tracking,
- document facts introduced early and queried late.

Why this milestone is next:
- Natural text alone is too noisy to debug a new memory mechanism cleanly.
- If a memory mechanism cannot win on a task designed for memory, it should not be trusted on open-ended language modeling.

Exit criteria:
- At least one tiny memory-sensitive task runs end to end.
- `003` and later memory models can be compared on exactly the same task.
- The task is simple enough that failure is interpretable.

Questions to answer:
- Does the task truly require cross-chunk state?
- Can a chunk-local baseline solve it without memory?
- What kind of memory should the model need to store?

Status:
- Complete via `memory_architecture/005_memory_task_harness.py`.

Main lesson:
- The delayed-recall harness creates a clear cross-chunk pressure point: the chunk-local baseline stays near chance.
- Viewed alone, this still leaves one ambiguity: the task might be unlearnable or poorly specified rather than specifically blocked by chunk-local attention.

### Milestone 006: Full-Attention Delayed-Recall Control
Track: Evaluation control

Goal:
- Prove that the delayed-recall task from milestone 005 is learnable when the model has a direct cross-token information path.

What changes:
- Replace chunk-local self-attention with full causal self-attention over the whole sequence.

What stays fixed:
- Same synthetic delayed-recall data generator.
- Same vocabulary and sequence length.
- Same tiny model size.
- Same answer-position-only objective.
- No memory branch yet.

Why this milestone matters:
- It separates two different failure modes:
  - the chunk-local model cannot access the stored facts,
  - or the task itself is flawed or too hard for this model family.
- If full attention learns while chunk-local attention stays near chance, the harness is a credible memory benchmark.

Exit criteria:
- A matching full-attention control beats the chunk-local baseline on the same delayed-recall task.
- The result is reported with answer loss, answer accuracy, and wall-clock cost.

Questions to answer:
- Does unrestricted causal attention make the task materially easier?
- Is the gap large enough to justify using this harness for future memory models?
- Is the full-attention result an upper reference point or already close to saturation?

Status:
- Complete via `memory_architecture/006_full_attention_task_harness.py`.

Main lesson:
- The delayed-recall harness is credible enough to use for memory experiments because the chunk-local baseline stays near chance while the full-attention control learns materially better.
- `006` is therefore not a memory architecture step. It is the control that makes the next memory architecture steps interpretable.

### Milestone 007: Dense Latent Address Read Path
Track: Addressing

Goal:
- Move from “memory keys” toward a real latent address space.

What changes:
- Represent each memory entry as:
  - an address vector in a latent space,
  - and a value vector stored at that address.
- Replace plain projected key matching with an explicit similarity or distance-based read rule over addresses.

What stays fixed:
- No writes yet.
- Dense differentiable retrieval over all memory entries.
- Small model and small memory bank.

Why this milestone matters:
- It is the first real Path-B step.
- It turns memory into a geometric address space rather than just an extra learned table.

Exit criteria:
- Retrieval runs correctly and is explainable in terms of address-space geometry.
- The model can be compared honestly against the chunk-local and full-attention controls from milestones 005 and 006.

Questions to answer:
- Is dot-product similarity enough, or is an explicit distance kernel more interpretable here?
- Should address dimension equal model dimension?
- Does this behave differently from the earlier static key-value table in practice?

Status:
- Complete via `memory_architecture/007_dense_latent_address_read.py`.

Main lesson:
- Dense latent-address reading is implemented, but read-only memory remains near the chunk-local baseline on delayed recall.
- This is expected because the task's key-value facts are sampled per sequence and cannot be stored in static learned memory values.
- The result supports moving to writable values next; sparse retrieval is more informative after the model has useful runtime memory to retrieve from.

### Milestone 008: Writable Values At Fixed Addresses
Track: Writing

Goal:
- Introduce online memory state without yet changing the address system itself.

What changes:
- Memory addresses stay fixed and learned.
- Memory values become per-example runtime state.
- The model updates memory values after each chunk.
- Later chunks can read from memory values written by earlier chunks.

What stays fixed:
- Same delayed-recall task harness.
- Same chunk-local token attention.
- Same dense address read rule from milestone 007.
- No address updates or allocation yet.

Why this milestone is next:
- The current benchmark samples new key-value facts for every sequence.
- A static read-only memory cannot store those per-example facts.
- Writable values are the smallest change that gives the model a real cross-chunk state path.

Suggested first implementation:
- Start from `007_dense_latent_address_read.py`.
- Process the sequence chunk by chunk instead of all chunks at once.
- Initialize runtime memory values as zeros with shape `[batch, memory_slots, embedding_dim]`.
- For each chunk:
  - run local token processing,
  - read from the current runtime memory values,
  - compute a chunk summary,
  - write an update back into the runtime memory values.
- Keep addresses global learned parameters.

Exit criteria:
- Memory values are updated across chunks.
- The write path is differentiable and stable.
- The model can beat its read-only predecessor on at least one memory-sensitive task.

Questions to answer:
- What chunk summary should be written?
- Should updates be additive, interpolated, or overwrite-based?
- How selective should the write weights be?
- Should the model read before writing, after writing, or both?
- Does runtime memory improve answer accuracy beyond the near-chance `005` and `007` baselines?

Status:
- Complete via `memory_architecture/008_writable_fixed_address_memory.py`.

Main lesson:
- Writable fixed-address memory is the first strong positive result on the delayed-recall harness.
- It improves answer accuracy from the near-chance read-only result `M-007` (`0.0698`) to `0.2480`.
- It nearly matches the full-attention control `M-006` (`0.2505`), while preserving chunk-local token attention.
- This supports the working hypothesis that runtime memory writing matters more than static read-only retrieval for this benchmark.

### Milestone 009: Sparse Neighborhood Retrieval
Track: Addressing

Goal:
- Make retrieval local in address space rather than dense over the whole bank.

Candidate retrieval rules:
- top-k nearest addresses,
- soft neighborhood over a radius-like kernel,
- or a differentiable approximation that becomes sparse enough to matter.

Why this comes after writable values:
- Dense retrieval is already understood well enough for the current scale.
- Sparse retrieval is easier to interpret once memory values contain per-example state.
- A sparse read-only model would mostly test efficiency, not the missing memory behavior.

What stays fixed:
- Writable runtime memory values.
- Address geometry.
- Small model.
- Same task harness and controls.

Exit criteria:
- The model retrieves only a small neighborhood of memory entries.
- The sparse read path remains numerically stable.
- Runtime and quality can be compared against dense writable retrieval honestly.

Questions to answer:
- Does sparse retrieval actually help, or only make the model harder to train?
- Is the best neighborhood defined by top-k, thresholding, or a learned temperature?
- How does sparsity affect gradient quality?
- Does sparse retrieval preserve any behavioral gain from writable dense memory?

Status:
- Complete via `memory_architecture/009_sparse_neighborhood_retrieval.py`.

Main lesson:
- Sparse top-k reads preserve the writable-memory behavior on the delayed-recall harness.
- With `8` reads out of `64` slots, final answer accuracy reaches `0.2510`, essentially tied with dense writable memory `M-008` (`0.2480`) and the full-attention control `M-006` (`0.2505`).
- The naive sparse top-k implementation is slower than dense retrieval at this small scale, so the current result is about address locality, not runtime efficiency.
- This result later motivated the multi-query benchmark reset because `0.2510` is too close to the `1 / 4 = 0.25` candidate-guessing baseline.

### Milestone 010: Multi-Query Chunk-Local Binding Baseline
Track: Baseline

Goal:
- Replace the old single-query delayed-recall benchmark with the multi-query binding benchmark.

What changes:
- Start from `005_memory_task_harness.py`.
- Store `8` key-value facts instead of `4`.
- Query every stored key once, so each sequence has `8` answer positions.
- Report exact answer accuracy and candidate-value accuracy separately.

What stays fixed:
- Same chunk-local architecture as `M-005`.
- Same sequence length, chunk size, optimizer, model size, and synthetic vocabulary.
- No cross-chunk attention and no memory path.

Why this milestone matters:
- The old benchmark allowed candidate-set recovery to look like useful memory.
- This benchmark makes exact key-value binding visible.
- It gives the memory model a fair no-memory baseline on the final benchmark.

Status:
- Complete via `memory_architecture/010_binding_sensitive_task_harness.py`.

Main lesson:
- Chunk-local attention does not solve the multi-query binding task.
- Final exact answer accuracy is `0.0715`, close to random value-token behavior and below the `1 / 8 = 0.125` candidate-guessing baseline.
- Final candidate-value accuracy is `0.4387`, below the `8 / 16 = 0.5000` random candidate-value baseline.

### Milestone 011: Multi-Query Full-Attention Binding Control
Track: Control

Goal:
- Establish that the final benchmark is learnable when the model has direct attention access to all stored facts.

What changes:
- Start from `006_full_attention_task_harness.py`.
- Apply the same multi-query benchmark changes used in `M-010`.
- Report exact answer accuracy and candidate-value accuracy separately.

What stays fixed:
- Same full-attention architecture as `M-006`.
- Same model size, optimizer, sequence length, and vocabulary as `M-010`.

Why this milestone matters:
- A memory model should not be judged on a task that full attention cannot learn.
- Full attention is the positive control for exact binding.

Status:
- Complete via `memory_architecture/011_full_attention_binding_sensitive_task_harness.py`.

Main lesson:
- Full attention is a positive exact-binding control on the multi-query benchmark.
- Final exact answer accuracy is `0.3431`, clearly above the `0.1250` candidate-guessing baseline.
- Final candidate-value accuracy is `1.0000`.

### Milestone 012: Sparse Writable Memory On Multi-Query Binding
Track: Writing + Addressing

Goal:
- Rerun the strongest current memory mechanism on the final multi-query binding benchmark.

What changes:
- Start from `009_sparse_neighborhood_retrieval.py`.
- Apply the same multi-query benchmark changes used in `M-010`.
- Keep sparse top-k reads and writable runtime memory values.
- Report exact answer accuracy and candidate-value accuracy separately.

What stays fixed:
- Chunk-local token attention.
- Fixed learned memory addresses.
- Runtime writable memory values.
- Sparse top-k read setting: `8` of `64` slots.

Why this milestone matters:
- This is the first real test of whether sparse writable memory learned binding rather than only candidate-set recovery.
- It creates the fixed-address memory baseline that address dynamics must beat.

Status:
- Complete via `memory_architecture/012_sparse_memory_binding_sensitive_task_harness.py`.

Main lesson:
- Sparse writable memory beats the chunk-local baseline and candidate guessing on exact binding.
- Final exact answer accuracy is `0.2134`, compared with `0.0715` for chunk-local and `0.3431` for full attention.
- Final candidate-value accuracy is `1.0000`.
- The memory path is useful, but still loses exact binding information relative to full attention.

### Milestone 013: Runtime Address State Control
Track: Addressing

Goal:
- Convert global fixed addresses into per-example runtime address state without allowing the addresses to move yet.

What changes:
- Start from `012_sparse_memory_binding_sensitive_task_harness.py`.
- Keep the learned base address table.
- At the start of each forward pass, copy the base addresses into runtime addresses with shape `[batch, memory_slots, address_dim]`.
- Make the read and write paths consume runtime addresses instead of only the global address table.

What does not change:
- Runtime addresses are not updated yet.
- Memory values still update as in milestone 012.
- Sparse top-k reads stay fixed at `8` of `64` slots.
- Same multi-query binding benchmark and controls.

Why this milestone matters:
- It isolates the tensor and API change needed for dynamic addresses.
- If this control changes behavior, the later dynamic-address results would be hard to interpret.
- The expected result is near-equivalence with milestone 012.

Exit criteria:
- Runtime addresses are threaded through reads and writes correctly.
- The model remains numerically stable.
- Final accuracy stays close to `M-012` rather than collapsing.

Questions to answer:
- Does merely making addresses batched/runtime state change the result?
- Are there any hidden assumptions in the current code that require global shared addresses?
- Is the code still easy enough to explain before adding address movement?

Status:
- Complete via `memory_architecture/013_runtime_address_state_control.py`.

Main lesson:
- Making addresses batched runtime state does not collapse the sparse-memory model.
- Final exact answer accuracy is `0.2380`, close to and slightly above `M-012` at `0.2134`.
- Final candidate-value accuracy remains `1.0000`.
- This validates the address-state API change before adding real address movement.

### Milestone 014: Bounded Address Drift
Track: Writing + Addressing

Goal:
- Let existing memory slots move slightly in address space while keeping the number of slots fixed.

What changes:
- Runtime addresses now update after each chunk.
- Each updated address is old address plus a small learned delta.
- Addresses are normalized after the update so the similarity geometry remains bounded.
- Address updates should be gated by write strength, so slots that receive no meaningful write do not drift arbitrarily.

What stays fixed:
- Same memory size.
- Same writable memory values.
- Same sparse top-k read.
- Same multi-query binding benchmark.
- No slot creation, deletion, free list, or replacement rule yet.

Why this milestone matters:
- It tests the first real address-dynamics question without introducing allocation.
- It answers whether address movement helps the model organize memory or destabilizes retrieval.

Exit criteria:
- Address updates run without numerical collapse.
- Accuracy is compared against `M-012` (`0.2134`) and `M-013` (`0.2380`).
- There is at least one simple inspection of address movement magnitude over training or during evaluation.

Questions to answer:
- Does address drift destroy retrieval stability?
- Do addresses actually move, or does the model learn to keep them fixed?
- Are moved addresses more useful than fixed learned addresses on this task?

### Milestone 015: Address Drift Controls And Ablations
Track: Evaluation + Addressing

Goal:
- Determine whether address movement itself is responsible for any observed behavior.

Candidate controls:
- freeze address updates after initialization,
- use a much smaller address-update scale,
- detach address-update gradients through the write path,
- or update addresses with gates disabled.

Why this milestone matters:
- A dynamic-address model can appear to work for the wrong reason.
- Before adding allocation, we need to know whether address drift is useful, harmless, or actively avoided by the model.

Exit criteria:
- At least two targeted controls are run against milestone 014.
- The learning log clearly states whether address movement earned its complexity.
- The next allocation step is either justified or explicitly deprioritized.

Questions to answer:
- Is address movement doing real work, or is the value memory still carrying everything?
- How sensitive is the model to address-update scale?
- Does the learned system prefer stable addresses even when movement is available?

### Milestone 016: Bounded Slot Allocation
Track: Writing + Addressing

Goal:
- Let the model choose when a memory slot should represent a new memory rather than only updating an existing one.

What changes:
- Add a bounded usage or freshness signal per memory slot.
- Let the writer choose between updating an existing nearby slot and overwriting a low-usage slot.
- When a slot is overwritten, update both its value and its runtime address.

What stays fixed:
- Memory size remains bounded.
- Retrieval remains sparse top-k.
- The task harness stays synthetic and controlled.

Why this comes after drift:
- Allocation changes both where a memory lives and whether old content is preserved.
- It should only be tested after address drift has been isolated.

Exit criteria:
- The model can use bounded allocation without immediate collapse.
- Collisions and overwrites are measurable.
- There is evidence this buys something beyond fixed slots or simple address drift.

Questions to answer:
- What does memory allocation mean when memory size is bounded?
- How should collisions be handled?
- Does allocation improve behavior, or only add instability?

### Milestone 017: Longer-Context Pressure Test
Track: Scaling

Goal:
- Test whether the memory mechanism becomes more useful as context pressure increases.

What changes:
- Increase sequence length,
- maybe reduce chunk size relative to sequence length,
- and enlarge the memory bank modestly.

Rule:
- Scale only after the mechanism has shown a real signal on the small task harness.

Exit criteria:
- At least one memory-enabled model shows better scaling behavior than the no-memory chunk-local baseline as context pressure increases.

Questions to answer:
- Does the value of memory grow with longer contexts?
- Does the address space stay usable at larger memory counts?
- Does runtime scale acceptably?

### Milestone 018: Natural-Text Evaluation Pass
Track: External validity

Goal:
- Move from synthetic memory tasks back to natural language modeling with a more credible memory mechanism.

Candidate evaluations:
- TinyStories with longer chunked contexts,
- task-specific document recall setups,
- or lightweight multi-document memory settings.

Why this milestone matters:
- A memory mechanism should eventually matter on natural text, not only on toy recall tasks.

Exit criteria:
- The model is tested on at least one natural-text setting where memory plausibly matters.
- Results are interpreted against both quality and runtime cost.

Questions to answer:
- Does success on synthetic tasks transfer at all?
- Is the memory system learning semantic state or only benchmark tricks?

### Milestone 019: Thesis-Grade Freeze
Track: Research packaging

Goal:
- Freeze one architecture and one evaluation suite that are strong enough to support a thesis or paper-style writeup.

Deliverables:
- one frozen model definition,
- one frozen training recipe,
- one frozen benchmark set,
- one ablation table,
- one clear failure analysis section,
- and one clear claim that is narrow enough to defend.

Why this milestone matters:
- “Memory is interesting” is not a thesis.
- A precise, defensible claim about one mechanism is.

Exit criteria:
- You can state exactly what was tested, what worked, what failed, and why the result matters.

Questions to answer:
- What is the narrowest honest claim the evidence supports?
- Is the contribution in addressing, writing, scaling, or evaluation?
- What would make the work thesis-worthy rather than just exploratory?

## Recommended Order

The intended order from the current point is:

1. milestone 014: bounded address drift,
2. milestone 015: address drift controls and ablations,
3. milestone 016: bounded slot allocation,
4. milestone 017: longer-context pressure test,
5. milestone 018: natural-text evaluation,
6. milestone 019: thesis-grade freeze.

This order matters.
It keeps the research disciplined:

- first prove memory-sensitive tasks with the right controls,
- then prove latent-space reading,
- then prove writing,
- then compare sparse retrieval once there is useful runtime memory,
- then tighten the benchmark so exact binding is measured directly,
- then test whether addresses can move or be allocated,
- then test scale,
- then package the result honestly.

## Non-Goals

This path should not become:

- a generic vector database,
- a giant agent-memory framework,
- a production RAG system,
- a broad neuroscience-inspired memory zoo,
- or a vague “memory AI” project with no falsifiable milestones.

Those directions may become relevant later.
They are not the right way to learn from this repo now.

## Success Condition

This path succeeds if, by the end, you can say:

- I built a model with a real internal addressable memory mechanism.
- I can explain how it reads, how it writes, and why the address space behaves the way it does.
- I have evidence about when it helps and when it does not.
- I can distinguish weak static retrieval ideas from stronger persistent memory ideas.
- I have one narrow, defensible architecture contribution that could support a thesis or startup direction.

That is the standard.
Not “memory sounds exciting.”
Real mechanism, real controls, real evidence.
