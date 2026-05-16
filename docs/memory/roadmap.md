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

As of 2026-05-16:

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

The next roadmap step is address updates or allocation.

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

The first nine milestones already exist in code and are part of the roadmap.
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
- The next useful mechanism question is whether memory addresses can be updated or allocated, because fixed addresses are now doing enough to justify testing address dynamics.

### Milestone 010: Address Updates Or Allocation
Track: Writing + Addressing

Goal:
- Let the model change where memories live, not only what values they contain.

Candidate mechanisms:
- update existing address vectors,
- allocate new addresses from a controller,
- merge into nearby addresses,
- or use a bounded free-list / replacement rule.

Why this is later:
- Changing addresses and values at the same time is much harder to interpret.
- This step should only happen after fixed-address writable memory is understood.

Exit criteria:
- The model can modify or allocate addresses without immediate collapse.
- The address space remains interpretable enough to visualize and inspect.
- There is evidence this buys something beyond fixed-address writable values.

Questions to answer:
- Does address drift destroy retrieval stability?
- How should collisions be handled?
- What does memory allocation mean when memory size is bounded?

### Milestone 011: Longer-Context Pressure Test
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

### Milestone 012: Natural-Text Evaluation Pass
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

### Milestone 013: Thesis-Grade Freeze
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

1. milestone 010: address updates or allocation,
2. milestone 011: longer-context pressure test,
3. milestone 012: natural-text evaluation,
4. milestone 013: thesis-grade freeze.

This order matters.
It keeps the research disciplined:

- first prove memory-sensitive tasks with the right controls,
- then prove latent-space reading,
- then prove writing,
- then compare sparse retrieval once there is useful runtime memory,
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
