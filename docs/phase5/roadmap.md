# Phase 5: Modern Architecture From Scratch

This document defines the fifth learning phase of the repo.

Phase 5 exists because the repo has never built a *current* language model.
It has built a correct one, a scaled one, a handwritten one, and a modestly modernized one, but the architecture that frontier labs actually shipped in 2026 contains roughly ten mechanisms that this repo has never implemented.

The goal of phase 5 is to close that gap the same way phase 1 closed the transformer gap:
start from a vanilla decoder-only transformer and add one mechanism at a time until the model is the modern thing.

For the run history from this path, see `learning_log.md`, which is created when the first milestone completes.

## Current Status

As of 2026-08-16:

- The roadmap is written and the target architecture is chosen.
- Milestones 501, 502, and 503 are complete, recorded as `P5-001`, `P5-002`, and `P5-003`.
- Validation loss so far: `3.0537` for the vanilla post-norm baseline, `1.0161` for pre-norm, `0.9563` for RMSNorm with no biases.
- The baseline does not learn at the control learning rate. It collapses to character-unigram loss by step `250`, with the smallest gradient norms of any configuration tested. A control sweep confirms the code is correct: the same script reaches `2.15` at lr `3e-4`.
- RMSNorm plus bias removal improved loss, parameter count, and wall-clock simultaneously, which makes it the cheapest win in the ladder so far.
- Milestone 504 is next.
- Still outstanding: the supplementary `post-norm, lr 3e-4` run at full length, so the `501` to `502` comparison is not credited entirely to norm placement.

## Why This Phase Is Separate

Phase 4 is the framework-to-kernel path: PyTorch baseline, profiling, Triton, CUDA.
Its first two milestones are complete; `401` built the tiny PyTorch decoder and `402` added RoPE, GQA, SwiGLU, and RMSNorm.

That leaves phase 4 blocked on a question it cannot answer itself: **profile what?**

`phase4/006_char_decoder_rope_gqa_swiglu.py` is a 2023-era architecture.
Profiling it, then writing Triton kernels for it, would teach the workflow against a workload nobody runs anymore.
The hot paths in a 2026 model are not the hot paths in a dense multi-head decoder: they are sparse expert dispatch, compressed KV attention, and chunked linear-attention recurrences.

So phase 5 comes first and produces the thing phase 4 profiles.
This matches the repo thesis in [project_direction.md](../meta/project_direction.md): freeze a real target, then rebuild the important parts at a lower level.

Phase 5 is also deliberately narrow in a second way: **it is about model architecture and nothing else.**
No tokenizer work, no data pipeline work, no scaling work, no distributed work.
Character-level TinyStories and one fixed trainer, held constant across every milestone, so that every measured difference comes from the model.

## Core Goal

Build a vanilla decoder-only transformer, then modernize it one mechanism at a time until it is a current-generation architecture:

1. vanilla decoder (2017/GPT-2 shape),
2. normalization and position modernization,
3. attention modernization,
4. feed-forward and sparsity modernization,
5. objective and optimizer modernization,
6. one integrated modern reference model with an ablation table.

Every milestone must be explainable from mechanism.
If a change cannot be explained beyond "the frontier labs do it," it does not belong in this phase yet.

## What The 2026 Frontier Converged On

This section is the evidence base for the target architecture.
It was compiled from the Kimi K3 and DeepSeek-V4 technical reports directly, plus secondary coverage of GLM-5, Qwen3.5, MiniMax, and the open-weight architecture galleries.

Near-universal across every serious 2026 model:

| Component | Status | Notes |
| --- | --- | --- |
| Pre-norm + RMSNorm, no biases | universal | LayerNorm survives only in legacy models |
| RoPE | universal | often partial RoPE, or NoPE on a subset of layers |
| SwiGLU feed-forward | universal | Kimi K3 caps both branches (SiTU-GLU) to control activation outliers |
| GQA as the floor, MLA at flagship scale | universal | KV-cache pressure is the driver |
| QK-Norm on queries and keys | near-universal | dropped only where it interacts badly with very long context |
| Sparse MoE with fine-grained routed experts plus shared experts | universal above roughly 100B | first block or blocks stay dense |
| Sigmoid router with auxiliary-loss-free load balancing | universal in MoE models | a per-expert bias steers dispatch without touching the mixture weights |
| Layerwise hybrid attention, roughly 3 local to 1 global | the dominant 2026 shift | local mixer is sliding-window, linear, or delta-rule |
| Gated attention output | rapidly standard | data-dependent gate on the attention output; mitigates attention sinks |
| Multi-token prediction | common, not universal | training-time auxiliary head, optionally reused for speculative decoding |
| Muon for matrix parameters | frontier standard | Kimi K3 uses a per-head variant; DeepSeek-V4 adopted Muon in V4 |
| Untied input and output embeddings | modern default at scale | small models still tie |

Concrete anchors:

- **Kimi K3** (2.78T total, 104.2B active, 93 layers): 3 Kimi-Delta-Attention layers to 1 Gated MLA layer, plus one extra global layer at the end of the backbone; **NoPE everywhere**, with position carried by the linear-attention recurrence; 896 routed experts with 16 active per token and 2 shared experts; sigmoid router with Quantile Balancing instead of an auxiliary loss; SiTU-GLU; an RMSNorm inserted between routed-expert aggregation and the up-projection; attention residuals across depth; 1 MTP layer; 1 dense layer; per-head Muon.
- **DeepSeek-V4** (V4-Pro 1.6T total / 49B active, V4-Flash 284B / 13B): keeps DeepSeekMoE and the V3 MTP configuration unchanged; changes the router affinity function from sigmoid to `sqrt(softplus)`; keeps auxiliary-loss-free balancing plus a small sequence-wise balance loss; replaces the dense FFN in the first several blocks with hash-routed MoE; introduces manifold-constrained hyper-connections (`mHC`) in place of plain residual connections; interleaves Compressed Sparse Attention with Heavily Compressed Attention; uses Muon.
- **GLM-5** (744B / 40B active): MLA plus DeepSeek Sparse Attention, partial RoPE, QK-Norm, 256 routed experts with top-8 and 1 shared expert, sigmoid loss-free routing.
- **Qwen3.5** (397B / 17B active): hybrid Gated DeltaNet with gated full attention, sparse MoE.

The useful counter-signal:

- **MiniMax M2 and M2.5 went back to plain full attention.** They trained hundreds of billions to trillions of tokens on sliding-window and hybrid variants and found all of them worse on retrieval, multi-hop reasoning, and in-context learning, with the gap widening above 32K context after fine-tuning. They only shipped sparse attention in M3, once it was production-ready.

That matters for this phase.
It means the hybrid-attention milestone is a **measurement**, not a foregone conclusion, and the honest result may be that full attention wins at this scale.

## Target Architecture

This is the frozen end state for milestone 516.
Every milestone before it is a step along one of these rows.

| Component | Vanilla start (M-501) | Phase-5 end state (M-516) |
| --- | --- | --- |
| Normalization | LayerNorm, post-norm, biases everywhere | RMSNorm, pre-norm plus final norm, no biases anywhere |
| Positions | learned absolute position embedding | RoPE on local layers, NoPE on global layers |
| Attention layout | dense multi-head attention in every layer | 3 local to 1 global, last layer always global |
| Local mixer | — | gated linear attention with a delta-rule update |
| Global mixer | — | gated MLA with QK-Norm and a data-dependent output gate |
| Feed-forward | GELU MLP at 4x width | block 0 dense SwiGLU; the rest fine-grained MoE with routed plus shared experts, sigmoid top-k router, auxiliary-loss-free balancing bias, RMSNorm before the up-projection |
| Residual stream | plain residual | attention residuals or hyper-connections, kept only if measured to earn it |
| Objective | next-token cross-entropy | next-token cross-entropy plus a multi-token-prediction auxiliary loss |
| Embeddings | tied input and output | untied |
| Optimizer | AdamW | Muon on 2D matrices, AdamW on everything else |

## The Frozen Control

Everything in this list is identical across all sixteen milestones.
A milestone that changes the control is not a valid milestone.

Data:

- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Representation: character-level vocabulary built from both splits
- No tokenizer, no shards, no data pipeline work

Trainer:

- Device: `mps`
- Seed: `1337`
- Sequence length: `256`
- Batch size: `32`
- Train steps: `3000`
- Optimizer: `AdamW`, learning rate `3e-3`, until milestone 515 changes it deliberately
- Gradient clipping: global norm `1.0`
- Eval interval: `250` steps
- Eval batches: `32`

Model size envelope:

- Embedding dim: `128`
- Attention heads: `4`, head dim `32`
- Decoder blocks: `8`, divisible by four so the 3:1 hybrid pattern is exact
- Dense feed-forward hidden dim: `512`
- Roughly `1.6M` parameters at the vanilla starting point

The size was chosen by measurement, not by taste.
On the development machine (Apple M4, `16GB`), a timing sweep over candidate shapes gave `230ms` per step at `128`-dim and six blocks, `294ms` at the chosen shape, `549ms` at `192`-dim and eight blocks, and `935ms` at `256`-dim and eight blocks.
The chosen shape puts a full `3000`-step run at roughly `15` minutes, which keeps sixteen milestones tractable on a laptop.
Naive implementations of MoE dispatch and chunked linear attention will be several times slower than that, which is expected and must be reported rather than hidden.

The learning rate deserves a specific note, because it is the one control setting that is unfair to milestone 501 on purpose.
A short probe at `400` steps on the chosen shape gave:

| Configuration | Validation loss at step 400 |
| --- | ---: |
| post-norm, learning rate `3e-3` | about `3.08`, stalled |
| post-norm, learning rate `1e-3` | `2.303` |
| post-norm, learning rate `3e-4` | `2.332` |
| pre-norm, learning rate `3e-3` | `1.713` |

Post-norm at eight blocks cannot use the learning rate that the rest of the ladder wants, and lowering the learning rate to rescue it makes every later milestone worse.
So the control keeps `3e-3`, and milestone 501 reports the stall as its actual result.
That is the honest historical lesson rather than a flaw in the baseline: pre-norm is what made depth trainable at aggressive learning rates without warmup.

Reporting, for every run:

- final train loss and validation loss,
- total parameters and active parameters per token,
- wall-clock seconds and tokens per second,
- loss history CSV and SVG under `artifacts/phase5/`, via `lib/plotting.py`.

## Global Rules

- Optimize for mechanism understanding, not for benchmark wins.
- One mechanism per milestone. Never change the trainer and the model in the same step.
- Keep the parameter budget roughly constant across milestones so that loss differences mean something. When a mechanism changes the natural width, match parameters explicitly, as with the `2/3` rule for SwiGLU and active-parameter parity for MoE.
- Prefer explicit tensor math over fused framework calls while the mechanism is the lesson. `scaled_dot_product_attention` and other fused paths belong to phase 4 profiling work, not here.
- Report wall-clock honestly. At this scale most modern efficiency mechanisms are slower, because their value is asymptotic and their fast implementations are kernels this repo has not written yet.
- A modernization that does not improve loss at this scale is still kept if it is a genuine 2026 standard, but the learning log must say plainly that it did not pay for itself here and why.
- Do not build a framework. Standalone numbered scripts, module-level configuration constants, shared code only for artifacts.
- Keep the learning log tied to completed milestones.

## Milestones

### Milestone 501: Vanilla Decoder-Only Transformer
Track: Baseline

Goal:
- Establish the pre-modern reference point that every later milestone is measured against.

What changes:
- Nothing yet. This is the starting architecture.

What it contains:
- Learned absolute position embeddings.
- Dense multi-head causal self-attention with biased q/k/v/o projections.
- Post-norm LayerNorm around both sublayers.
- A GELU feed-forward block at 4x width, with biases.
- Tied input and output embeddings.
- AdamW.

Exit criteria:
- One run completes end to end.
- Parameter count and wall-clock are recorded.
- Every tensor path in the file can be explained without reference to any other file.

Status:
- Complete via `phase5/001_vanilla_decoder.py`, recorded as `P5-001`.

Main lesson:
- At the control learning rate the baseline collapses to character-unigram loss (`3.0537`) by step `250` and never recovers, so "loss decreases smoothly" was not met.
- The collapse is an optimization failure, not a code bug and not a divergence: gradient norms are the smallest of any tested configuration, clipping fires on `1%` of steps, and the same script reaches `2.15` at lr `3e-4`.
- Post-norm puts the normalization on the residual stream itself, so there is no identity path through depth and no single learning rate that works. That is the mechanism milestone 502 removes.

Questions to answer:
- What does post-norm actually do to the gradient path?
- Where does the learned position table break down, and why did the field move away from it?

### Milestone 502: Pre-Norm Residual Stream
Track: Normalization

Goal:
- Move the normalization inside the residual branch and add a final norm before the output projection.

Why it comes first:
- Pre-norm is the change that makes every later depth-related mechanism trainable at all.
- It is also the cleanest single-variable demonstration of why normalization placement is a gradient-flow decision, not a preprocessing decision.

Exit criteria:
- Loss curve compared against `M-501` at identical seed and budget.
- The residual stream identity path is explicit in the code.

Status:
- Complete via `phase5/002_pre_norm.py`, recorded as `P5-002`.

Main lesson:
- Validation loss falls from `3.0537` to `1.0161` for four lines of code and `+256` parameters.
- The identity path is measurable outside training: activations scaled by `50x` leave a post-norm block at standard deviation `1.000` and a pre-norm block at `50.265`.
- Pre-norm trains with much larger gradients, clipping on `50%` of steps against `1%` for the collapsed post-norm run, so heavy clipping here is a sign of health.

Questions to answer:
- Why does post-norm need warmup that pre-norm does not?
- What does the final norm before the LM head actually protect?

### Milestone 503: RMSNorm And Bias Removal
Track: Normalization

Goal:
- Replace LayerNorm with RMSNorm and remove biases from every linear layer.

Why these are one milestone:
- Both are the same decision: drop the mean-centering and shift degrees of freedom that modern models found unnecessary.

Exit criteria:
- Parameter count drops and the loss does not.
- The exact arithmetic difference between LayerNorm and RMSNorm is written down.

Status:
- Complete via `phase5/003_rms_norm.py`, recorded as `P5-003`.

Main lesson:
- Everything improved at once: validation loss `1.0161` to `0.9563`, parameters down by `11392`, wall-clock down `9.5%` from `663.90s` to `601.10s`.
- The claim that re-centering contributes little is reproduced directly: dropping the mean subtraction and all biases cost no quality.
- Implementation trap: normalizing by `x.var(correction=0)` instead of the mean square agrees with RMSNorm only when the per-token mean is zero, and pre-norm is exactly the setting where the residual stream drifts away from it.

Questions to answer:
- What does mean subtraction buy, and why was losing it free?
- Which biases actually mattered, if any?

### Milestone 504: Rotary Position Embeddings
Track: Positions

Goal:
- Replace learned absolute positions with RoPE applied to queries and keys.

What stays fixed:
- Dense attention. Only the position mechanism changes.

Exit criteria:
- The position embedding table is gone.
- Relative-position behavior is explained from the rotation math, not asserted.

Questions to answer:
- Why does rotating q and k give relative position for free?
- What breaks at positions beyond the training length, and what do the frequency-scaling tricks actually do about it?

### Milestone 505: Grouped-Query Attention
Track: Attention

Goal:
- Share key and value heads across query heads.

Exit criteria:
- KV parameter and KV-cache footprint reductions are computed explicitly.
- Loss impact at fixed budget is reported.

Questions to answer:
- Is GQA a quality change or a memory change?
- Why is the KV cache, and not the parameter count, the thing that forced this?

### Milestone 506: SwiGLU Feed-Forward
Track: Feed-forward

Goal:
- Replace the GELU MLP with a gated SwiGLU feed-forward block at matched parameter count.

Rule:
- Apply the `2/3` width rule so the three-matrix gated block has the same parameters as the two-matrix block it replaces. Otherwise the comparison measures width, not gating.

Exit criteria:
- Matched-parameter comparison against `M-505`.
- The gating mechanism is explained as multiplicative feature selection.

Questions to answer:
- What does the gate branch let the block express that a single nonlinearity cannot?
- Why did the field standardize on this specific variant?

### Milestone 507: QK-Norm And Gated Attention
Track: Attention stability

Goal:
- Add RMSNorm to queries and keys, and a data-dependent sigmoid gate on the attention output.

Why these are one milestone:
- Both are stability mechanisms that act on the attention path, and both became near-standard in 2026 for the same reason: controlling outlier magnitudes.

Exit criteria:
- Attention logit magnitudes are inspected before and after QK-Norm.
- The gate's learned behavior is inspected, not just its loss effect.

Questions to answer:
- What failure mode does QK-Norm actually prevent, and would it appear at this scale?
- What is an attention sink, and how does an output gate relate to it?

### Milestone 508: Layerwise Hybrid Attention
Track: Attention layout

Goal:
- Interleave three sliding-window local attention layers with one full global attention layer, and remove RoPE from the global layers.

What stays fixed:
- All attention is still standard softmax attention. Only the receptive field and the position handling change per layer.

Why this ordering:
- Sliding-window plus global is the simplest honest version of the 2026 hybrid pattern, and it isolates the layout question from the linear-attention question that follows in `M-513`.

Exit criteria:
- The 3:1 pattern is exact and the last layer is global.
- Loss and wall-clock are compared against full attention everywhere.
- The MiniMax counter-result is explicitly tested at this scale rather than assumed.

Questions to answer:
- How much does the model lose when most layers cannot see the whole sequence?
- Why does NoPE work in the global layers when the local layers still carry RoPE?

### Milestone 509: Multi-Head Latent Attention
Track: Attention

Goal:
- Replace the global layers' key/value projections with a compressed latent representation that is up-projected during attention.

Ordering note:
- `M-508` already makes the global layers NoPE, which would let MLA skip the position problem entirely.
- Implement the decoupled-RoPE form anyway, because that conflict is the whole reason MLA looks the way it does, and then compare it against the NoPE form that this ladder can actually afford to use.

Exit criteria:
- KV-cache footprint per token is computed and compared against GQA.
- The decoupled position handling required by KV compression is implemented correctly and explained.
- Decoupled-RoPE MLA and NoPE MLA are both run, and the difference is recorded.

Questions to answer:
- Why does compressing KV into a latent conflict with RoPE, and what are the ways around it?
- Is MLA a better GQA, or a different tradeoff entirely?

### Milestone 510: Sparse Mixture-Of-Experts Feed-Forward
Track: Sparsity

Goal:
- Replace the dense feed-forward block with fine-grained routed experts plus a shared expert, keeping block 0 dense.

Configuration principle:
- Match active parameters per token to the dense baseline so the comparison isolates sparsity from capacity.

Exit criteria:
- Total parameters and active parameters per token are both reported.
- Routing is correct: every token reaches exactly top-k routed experts plus the shared expert.
- Wall-clock is reported honestly against the dense baseline.

Questions to answer:
- What is a fine-grained expert, and why did expert counts grow while expert width shrank?
- What does the shared expert absorb that the routed experts should not have to?

### Milestone 511: Auxiliary-Loss-Free Load Balancing
Track: Sparsity

Goal:
- Balance expert load with a per-expert routing bias instead of an auxiliary loss, and instrument the routing.

Why it is separate from `M-510`:
- Routing correctness and routing balance are different problems, and collapsed routing is easy to hide behind a working loss curve.

Exit criteria:
- Per-expert load, routing entropy, and dead-expert count are tracked across training.
- The balanced and unbalanced runs are compared on both loss and load statistics.

Questions to answer:
- Why does a bias applied only to selection, and not to the mixture weights, avoid distorting the gradient?
- What does expert collapse look like before it shows up in the loss?

### Milestone 512: Multi-Token Prediction
Track: Objective

Goal:
- Add an auxiliary head that predicts the token after next, and blend its loss into the objective.

Exit criteria:
- Main-task loss is reported separately from the auxiliary loss.
- The effect of the auxiliary loss weight is checked at more than one value.

Questions to answer:
- Why does predicting further ahead improve the representation used for the next token?
- What is the relationship between this training-time head and speculative decoding at inference?

### Milestone 513: Gated Linear Attention With A Delta Rule
Track: Attention

Goal:
- Replace the sliding-window local layers with a gated linear-attention recurrence using a delta-rule state update, keeping the 3:1 pattern against the global MLA layers.

Why it comes late:
- It is the least settled mechanism in the target architecture and the hardest to implement correctly, and it only makes sense once the hybrid layout from `M-508` exists to drop it into.

Requirements:
- Implement both the recurrent form and the chunked parallel form, and check that they agree numerically.
- Remove RoPE from these layers; position comes from the recurrence.

Exit criteria:
- The two forms match to numerical tolerance.
- Loss, wall-clock, and state-size behavior are compared against the sliding-window variant.

Questions to answer:
- What does the delta rule change relative to plain linear attention?
- What does the gate control, and what happens to the state when it saturates?
- What does this actually give up relative to softmax attention?

### Milestone 514: Residual-Stream Upgrades
Track: Depth

Goal:
- Test one modern replacement for the plain residual connection.

Candidates:
- Attention residuals, where each layer attends over the outputs of all preceding layers.
- Hyper-connections, where the residual stream is widened and mixed with learned, constrained maps.

Rule:
- Implement one, measure it, and keep it only if it earns its complexity at this scale. Both are real 2026 mechanisms, and both are plausible losers at `8` layers.

Exit criteria:
- The mechanism runs stably and its cost in memory and time is measured.
- The decision to keep or drop it is recorded with the reason.

Questions to answer:
- What information does a plain residual stream destroy?
- Why does depth-wise mixing need explicit stability constraints?

### Milestone 515: Optimizer And Initialization Modernization
Track: Optimization

Goal:
- Move from AdamW everywhere to Muon on 2D matrices with AdamW on embeddings, norms, and gains, and untie the input and output embeddings.

Why it comes last among the mechanisms:
- Optimizer changes shift every earlier comparison. Doing this earlier would invalidate the ladder.

Exit criteria:
- The Newton-Schulz orthogonalization step is implemented explicitly, not imported.
- A matched-budget comparison against AdamW is recorded.
- The parameter-group split is justified per group.

Questions to answer:
- What does orthogonalizing the update actually do to the learning dynamics?
- Why do embeddings and norms stay on AdamW?
- Does untying help or hurt at a vocabulary of about one hundred characters, and why is that different from a vocabulary of 160K?

### Milestone 516: Modern Reference Model
Track: Integration

Goal:
- Produce one integrated model containing every mechanism that earned its place, plus the ablation table for the whole phase.

Deliverables:
- One frozen model definition.
- One frozen training recipe.
- One table of every milestone with loss, parameters, active parameters, and wall-clock.
- One explicit list of mechanisms that were implemented and then dropped, with reasons.

Exit criteria:
- The final model is the phase-4 profiling target.
- Every mechanism in it can be explained from first principles.
- The ablation table is honest about which changes did nothing at this scale.

Questions to answer:
- Which mechanisms were quality changes and which were purely systems changes?
- Which ones only pay off at scales this repo has not reached?
- What is the shortest defensible description of a 2026 architecture?

## Recommended Order

The intended order is exactly `501` through `516`.

The order is not arbitrary:

- normalization first, because it makes everything else trainable,
- positions next, because they are independent of the attention layout,
- attention layout before attention internals, because the layout decides what the internals operate on,
- sparsity after the dense feed-forward is understood,
- the objective after the model,
- the optimizer last, because it changes the meaning of every earlier comparison,
- integration only once each mechanism has a recorded, individual result.

## Non-Goals

Phase 5 should not become:

- a tokenizer or data-pipeline project,
- a scaling run,
- a distributed-training project,
- a kernel-optimization project, which is phase 4's job with phase 5's output,
- a race to reimplement every named mechanism from every technical report,
- a model that is modern but that the author cannot explain.

## Success Condition

Phase 5 succeeds if, by the end, the following are all true:

- I can build a current-generation language model architecture from scratch, without copying a reference implementation.
- I can explain every mechanism in it from mechanism, including what it costs.
- I know which of those mechanisms help at small scale, which are purely about inference economics, and which I only believe because a technical report said so.
- I have one frozen modern workload that the phase-4 profiling, Triton, and CUDA path can attack next.
