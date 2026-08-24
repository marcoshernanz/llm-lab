# Phase 5: Modern Architecture From Scratch

This document defines the fifth learning phase of the repo.

Phase 5 exists because the repo has never built a *current* language model.
It has built a correct one, a scaled one, a handwritten one, and a modestly modernized one, but the architecture that frontier labs actually shipped in 2026 contains roughly fifteen mechanisms that this repo has never implemented.

The goal of phase 5 is to close that gap the same way phase 1 closed the transformer gap:
start from a vanilla decoder-only transformer and add one mechanism at a time until the model is the modern thing.

Every milestone below is written to the same five-part shape, because the *why* is the part that survives:

1. **Goal** — the one sentence version.
2. **The problem** — what is actually broken in the architecture we have.
3. **What the field tried** — the dead ends and partial fixes, in order, because the solution only makes sense against them.
4. **The solution that won** — thorough, with the mathematics.
5. **Implementation decisions** — the small forks resolved in advance, so a later session does not re-litigate them.

For the run history, see [learning_log.md](learning_log.md).

## Milestone Index

| # | Milestone | One line | Status |
| --- | --- | --- | --- |
| 501 | Vanilla decoder-only transformer | The 2017/GPT-2 reference point everything is measured against | done |
| 502 | Pre-norm residual stream | Move the norm inside the branch so depth has an identity path | done |
| 503 | RMSNorm and bias removal | Drop mean-centering and every bias; same loss, fewer parameters | done |
| 504 | Rotary position embeddings | Rotate q and k so attention scores depend on distance, not index | done |
| 505 | Grouped-query attention | Share KV heads across query heads to shrink the KV cache | done |
| 506 | SwiGLU feed-forward | Multiplicative gating in the MLP, at matched parameters | done |
| 507 | QK-Norm and gated attention | Normalize q/k before the score; gate the attention output | done |
| 508 | Layerwise hybrid attention | Three sliding-window layers per one global NoPE layer | done |
| 509 | Multi-head latent attention | Compress KV to a latent on the global layers, with decoupled RoPE | done |
| 510 | Sparse mixture-of-experts | Fine-grained routed experts plus a shared expert, matched active cost | done |
| 511 | Real sparsity and loss-free balancing | Make routing actually sparse, then balance it with a selection-only bias | done |
| 512 | Bounded feed-forward activations | Cap both GLU branches so outliers cannot form | done |
| 513 | Attention sinks | Let a head attend to nothing via a learnable logit in the denominator | next |
| 514 | Multi-token prediction | A sequential auxiliary head that predicts token `t+2` | planned |
| 515 | Gated linear attention with a delta rule | Replace sliding-window layers with a KDA-style recurrence | planned |
| 516 | Residual-stream upgrade | Let each layer attend over the outputs of all preceding layers | planned |
| 517 | Muon optimizer | Orthogonalized updates on 2D matrices, AdamW on everything else | planned |
| 518 | Modern reference model | One integrated model plus the full ablation table | planned |

## Current Status

- Milestones 501 through 510 are complete and recorded as `P5-001` through `P5-010`.
- All runs live on a Kaggle `Tesla T4` at seed `1337`, which reproduces bit-exactly. Local `mps` is
  not reproducible, two identical runs diverging by up to `0.053` validation loss, so a results
  table must live on one platform.

| Milestone | Val loss | Delta | Parameters | Seconds |
| --- | ---: | ---: | ---: | ---: |
| 501 vanilla post-norm | `3.0556` | — | `6408704` | `661.1` |
| 502 pre-norm | `2.2576` | `-0.7980` | `6409216` | `678.6` |
| 503 RMSNorm, no biases | `2.1625` | `-0.0951` | `6386432` | `650.1` |
| 504 RoPE | `0.9334` | `-1.2291` | `6320896` | `714.0` |
| 505 grouped-query attention | `0.9052` | `-0.0282` | `5796608` | `659.7` |
| 506 SwiGLU | `0.9763` | `+0.0711` | `5792512` | `672.0` |
| 507 QK-Norm and gated attention | `0.7904` | `-0.1859` | `6317312` | `770.7` |
| 508 layerwise hybrid attention | `0.7887` | `-0.0017` | `6317312` | `733.8` |
| 509 latent attention, global layers | `0.8135` | `+0.0248` | `6358336` | `737.3` |
| 510 sparse mixture-of-experts | `0.8002` | `-0.0133` | `8899392` | `956.5` |
| 511 real sparsity, quantile balancing | `0.7988` | `-0.0014` | `14504768` | `2008.9` |
| 512 bounded activations | `0.7973` | `-0.0015` | `14504768` | `2493.3` |

- **Two mechanisms account for almost everything.** Pre-norm is worth `-0.7980` and RoPE `-1.2291`,
  together `-2.03` of the total `-2.26`. Everything after them moves the loss by under `0.19`.
- **Milestones 502 and 503 end stalled**, around `2.2`, and the reason is not the normalizer. Both
  were blocked on the learned absolute position table, which `M-504` removed.
- **The baseline never learns at the control learning rate.** It reaches character-unigram loss by
  step `250` and stays flat. That is post-norm's missing identity path, not a bug.
- **SwiGLU made the model worse**, `+0.0711` at matched parameters. A real negative result for a
  mechanism every frontier model uses.
- **QK-Norm with gated attention is the second-largest gain**, `-0.1859`, but it adds `9%`
  parameters, so that number confounds mechanism with capacity.
- **Sparsity bought nothing measurable**, `-0.0133` for `40%` more total parameters and `30%` more
  wall-clock, and `M-511` did not change that: `-0.0014` for another `63%` parameters at `2.1x`
  wall-clock.
- **Load balancing does work, decisively.** At `4` of `64` the unbalanced load is `24.5x` skewed;
  Quantile Balancing holds it at `1.13x` with no expert ever unused. The mechanism is real even
  though the loss is indifferent to it at this scale.
- Every run except the collapsed baseline was still improving at step `3000`. The ladder compares
  architectures at a fixed budget, not at convergence.

### Platform Notes

- Runs execute on a Kaggle `T4`, which reproduces itself bit-exactly at default settings. `P100` is not an option: it fails with `no kernel image is available for execution on the device`, since Kaggle's PyTorch no longer ships `sm_60` kernels.
- Kaggle permits two concurrent GPU sessions, and reports the limit as ordinary output rather than a failing exit code.
- Results are not comparable across devices. The same script at the same seed lands on a different loss on `mps` than on `T4`, so a results table must live entirely on one platform.
- Single deterministic runs are reproducible but not precise. Differences under roughly `0.02` cannot be claimed without repeated seeds; three seeds per milestone is about two hours of quota for the whole ladder.

## Why This Phase Is Separate

Phase 4 is the framework-to-kernel path: PyTorch baseline, profiling, Triton, CUDA.
Its first two milestones are complete; `401` built the tiny PyTorch decoder and `402` added RoPE, GQA, SwiGLU, and RMSNorm.

That leaves phase 4 blocked on a question it cannot answer itself: **profile what?**

`phase4/006_char_decoder_rope_gqa_swiglu.py` is a 2023-era architecture.
Profiling it, then writing Triton kernels for it, would teach the workflow against a workload nobody runs anymore.
The hot paths in a 2026 model are not the hot paths in a dense multi-head decoder: they are sparse expert dispatch, compressed or selected KV attention, and chunked linear-attention recurrences.

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
5. numerical-stability modernization,
6. objective, depth, and optimizer modernization,
7. one integrated modern reference model with an ablation table.

Every milestone must be explainable from mechanism.
If a change cannot be explained beyond "the frontier labs do it," it does not belong in this phase yet.

## The Evidence Base: Three Frontier Models, August 2026

This section was rebuilt from the primary technical reports rather than from recollection, and it is deliberately narrow: only models that were state of the art within three months of `2026-08-23`.

Sources read directly:

- **Kimi K3**, [arXiv:2607.24653](https://arxiv.org/abs/2607.24653), full architecture section (§2.1 through §2.5).
- **DeepSeek-V4**, [arXiv:2606.19348](https://arxiv.org/abs/2606.19348), architecture (§2) and model setups (§4.2.1).
- **GLM-5.2 / GLM-5.3**, architecture teardown and release notes. GLM-5.3 shipped `2026-08-14` on the **same architecture as GLM-5.2**; its entire gain came from scaled post-training. So for architectural purposes GLM-5.3 *is* GLM-5.2, and that itself is a finding: one of the three frontier labs shipped a major release with zero architecture change.

### Side-By-Side

| | Kimi K3 | DeepSeek-V4-Pro | GLM-5.2 / 5.3 |
| --- | --- | --- | --- |
| Total / active | `2.78T` / `104.2B` | `1.6T` / `49B` | `744B` / `~40B` |
| Layers | `93` | `61` | `78` |
| Hidden dim | `7168` | `7168` | `6144` |
| Attention | hybrid `3` KDA : `1` gated MLA | interleaved CSA / HCA | MLA + DeepSeek Sparse Attention |
| Sequence trick | linear-attention recurrence | KV compressed along *sequence* | top-`2048` token selection per query |
| Position | **NoPE everywhere** | partial RoPE, last `64` dims | partial RoPE, `64` of `256` |
| Routed experts | `896`, top-`16`, `2` shared | `384`, top-`6`, `1` shared | `256`, top-`8`, `1` shared |
| Expert hidden | `3072` (in a `3584` latent) | `3072` | `2048` |
| Sparsity | `1.8%` | `1.6%` | `3.1%` |
| Router affinity | `Sigmoid` | `Sqrt(Softplus)` | `Sigmoid` |
| Balancing | Quantile Balancing | loss-free bias + small sequence-wise loss | loss-free bias |
| Dense blocks | `1` | `0`, first `3` are hash-routed MoE | `3` |
| FFN activation | **SiTU-GLU**, softcapped | **SwiGLU clamped** to `[-10, 10]` | SwiGLU |
| Attention normalization | `L2Norm` on q/k, head RMSNorm | RMSNorm on q and KV entries | MLA-internal |
| "Attend to nothing" | full-rank sigmoid output gate | learnable sink logit in denominator | — |
| Residual stream | **Block Attention Residuals** | **mHC**, width `4`, Sinkhorn `20` | plain |
| MTP | `1` layer | depth `1` | `1` layer |
| Optimizer | Per-Head Muon | Muon + AdamW split | not documented |

### What Is Actually Unanimous

These are the rows where all three agree. They are the only ones that deserve to be called settled.

| Mechanism | Why it is unanimous |
| --- | --- |
| Pre-norm, RMSNorm, no biases | Nobody has shipped a post-norm or LayerNorm frontier model in years |
| Fine-grained MoE with shared experts | All three; the ratios differ, the shape does not |
| Sparsity near `2%`, not `50%` | `1.6%`, `1.8%`, `3.1%` — the ladder's `50%` is not sparse |
| Auxiliary-loss-free balancing | A bias on *selection only*, never on the mixture weights |
| Sub-quadratic attention | Unanimous in intent, three different mechanisms |
| Query/key normalization before the score | Unanimous, and it is why DeepSeek-V4 could drop QK-Clip entirely |
| Reduced or removed RoPE | `NoPE` (K3) or partial RoPE on `64` dims (V4, GLM) — nobody rotates the full head |
| Multi-token prediction at depth `1` | All three; all three reuse it for speculative decoding |

### Two Mechanisms Converging Right Now

Two mechanisms went from "one lab does it" to "two of three independently shipped it" in this window. Both are now milestones.

**1. Bounded feed-forward activations.** Two labs hit the same wall and patched it two different ways in the same quarter:

- Kimi K3 introduced **SiTU-GLU**, applying a smooth cap `softcap(x, β) = β·tanh(x/β)` to *both* branches, with `β₁ = 4` on the gate and `β₂ = 25` on the up branch, bounding the product at `100`.
- DeepSeek-V4 **clamped SwiGLU** to `[-10, 10]` on the linear branch and capped the gate at `10`, reporting that it "effectively eliminates outliers" and was one of only two techniques that fixed their loss spikes.

The shared diagnosis is stated plainly in both reports: SwiGLU multiplies two unbounded factors, so coincident large coordinates produce activation outliers that break low-precision arithmetic. That is a real, explainable defect in a mechanism this ladder already built at `M-506`.

**2. The residual stream is a bottleneck.** Also two of three, also independently:

- Kimi K3 ships **Attention Residuals**: each layer attends over the outputs of all preceding layers instead of reading one accumulated sum.
- DeepSeek-V4 ships **mHC**: the residual stream is widened `4x` and mixed by a matrix constrained to the doubly-stochastic Birkhoff polytope, which bounds its spectral norm at `1`.

Two frontier models shipping this in one quarter, by different routes, is what makes it a real mechanism rather than a speculative one.

### Where They Genuinely Disagree

Honesty requires listing these separately, because the roadmap must not present a contested choice as settled.

- **How to make attention sub-quadratic.** K3 uses a linear-attention recurrence, V4 compresses along the sequence axis, GLM selects top-`k` real tokens. These are three different bets. MiniMax is a fourth: after publicly reporting that sliding-window and hybrid variants were *worse* on retrieval and multi-hop reasoning, they shipped M3 in June 2026 with block-level selection over uncompressed KV — closest to GLM's approach, and a pointed rejection of compression.
- **Router affinity.** Two use `Sigmoid`; DeepSeek-V4 switched to `Sqrt(Softplus)`. One lab changing its mind once is not a trend.
- **Whether early blocks stay dense.** GLM keeps `3` dense, K3 keeps `1`, V4 keeps **none** and hash-routes the first `3` instead. Complete disagreement on a detail this ladder has already made a decision about.
- **MLA's future.** GLM-5.2 still uses it. DeepSeek invented it and has now moved past it, on the grounds that at a million tokens the sequence length dominates memory, not the head count.

### The One Result That Should Temper Everything Below

MiniMax trained hundreds of billions to trillions of tokens on sliding-window and hybrid attention and found them **worse**, with the gap widening above `32K` context. They shipped sparse attention only once their own variant was production-ready, and it is the least aggressive of the four.

That matters here. It means the attention-layout milestones are **measurements, not foregone conclusions**, and the honest result at this scale may well be that full attention wins.

## Target Architecture

This is the frozen end state for milestone 518.
Every milestone before it is a step along one of these rows.

| Component | Vanilla start (M-501) | Phase-5 end state (M-518) |
| --- | --- | --- |
| Normalization | LayerNorm, post-norm, biases everywhere | RMSNorm, pre-norm plus final norm, no biases anywhere |
| Positions | learned absolute position embedding | RoPE on local layers, NoPE on global layers |
| Attention layout | dense multi-head attention in every layer | `3` local to `1` global, last layer always global |
| Local mixer | — | gated linear attention with a delta-rule update |
| Global mixer | — | gated MLA, QK-Norm, output gate, learnable attention sink |
| Feed-forward | GELU MLP at `4x` width | block `0` dense; the rest fine-grained MoE, `~3%` sparsity, shared expert, sigmoid router, loss-free balancing bias |
| Activation bounds | none | both GLU branches softcapped |
| Residual stream | plain residual | block attention residuals |
| Objective | next-token cross-entropy | plus a depth-`1` multi-token-prediction loss |
| Embeddings | tied input and output | untied |
| Optimizer | AdamW | Muon on 2D matrices, AdamW on embeddings, norms, and the head |

## The Frozen Control

Everything in this list is identical across all eighteen milestones.
A milestone that changes the control is not a valid milestone.

Data:

- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Representation: character-level vocabulary built from both splits
- No tokenizer, no shards, no data pipeline work

Trainer:

- Device: Kaggle `Tesla T4`
- Seed: `1337`
- Sequence length: `256`
- Batch size: `32`
- Train steps: `3000`
- Optimizer: `AdamW`, learning rate `3e-3`, until milestone 517 changes it deliberately
- Gradient clipping: global norm `1.0`
- Eval interval: `250` steps
- Eval batches: `32`

Initialization, identical in every milestone:

- Every `nn.Linear` weight and every embedding table: `normal(0, 0.02)`.
- Every `nn.Linear` bias, where biases exist at all: zeros.
- Every norm gain: ones.
- This is applied with one `self.apply(init_weights)` at the end of `LanguageModel.__init__`, so a
  new module can never silently miss it.

The check that this is right is that a fresh model's loss equals `ln(vocab)`. At a `98`-character
vocabulary that is `4.585`, and every milestone starts within `0.07` of it. PyTorch's default for
`nn.Linear` is `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, which at `d_model = 128` had standard deviation
`0.051` — `2.5x` too wide — and inflates the pre-norm residual stream by `115x` over `8` blocks
against `45x` for the standard init. Nothing about this is exotic; it is what every reference
implementation does, and the ladder simply did not have it until it was audited.

Model size envelope:

- Embedding dim: `256`
- Attention heads: `8` query, `4` key/value, head dim `32`
- Decoder blocks: `8`, divisible by four so the 3:1 hybrid pattern is exact
- Dense feed-forward hidden dim: `4 * D_MODEL`, narrowing to `8 * D_MODEL // 3` once gated
- Rope dims per score head: `D_HEAD // 2`
- Latent dim on global layers: `64`
- Experts: `128` wide routed, `128` wide shared
- Roughly `6.4M` parameters at the vanilla starting point, `8.9M` once sparse

**A constant is written as a formula only when the formula is the real convention.** `4 * D_MODEL`
is the 2017 transformer's feed-forward ratio. `8 * D_MODEL // 3` is `2/3` of that, which is the
gated width Llama uses and the reason its configs read `11008`. `D_HEAD // 2` puts a third of each
score head on the rope path, matching DeepSeek V3's `128` content plus `64` rope split.

Everything else is a plain number. In particular, no constant is contorted to hold a parameter
count fixed across milestones. That kind of formula buys experimental tidiness at the cost of
readability, and this ladder is a way to see mechanisms work rather than a controlled study. Where
a milestone's number is therefore confounded with a size change, the learning log says so and moves
on.

The size was chosen by measurement on the `T4`, and the reason is not raw speed. A sweep over
`d_model` in `{128, 192, 256, 384}` and batch in `{32, 64, 128, 256}` showed that batch `32`
reaches only `75.1%` of the achievable throughput at `d_model 128`, but `90.9%` at `d_model 256`.
So the small model was wasting a quarter of the GPU, and the fix is a wider model rather than a
larger batch: widening buys capacity *and* utilization, while a larger batch would change the
optimization and force the learning rate to be re-probed.

Head count is `8` rather than `4` because the attention-shape mechanisms need something to work
with. GQA, hybrid attention, and latent attention all trade head structure for memory, and a
four-head model has almost nothing to trade: sharing `4` heads down to `2` is barely a ratio, and
there is little redundancy across four heads for a latent to compress. Head dim is `32`, so each
head behaves the same as it would at any other count.

Cost at this size: one `3000`-step run is roughly `12` to `17` minutes on a `T4`, and the full
`18`-milestone ladder at three seeds is about `11` to `15` hours, against a weekly GPU quota near
`30` hours. Two runs execute concurrently, so wall-clock is about half that.

The learning rate deserves a specific note, because it is the one control setting that is unfair to milestone 501 on purpose.
A short probe at `400` steps gave:

| Configuration | Validation loss at step 400 |
| --- | ---: |
| post-norm, learning rate `3e-3` | about `3.08`, stalled |
| post-norm, learning rate `1e-3` | `2.303` |
| post-norm, learning rate `3e-4` | `2.332` |
| pre-norm, learning rate `3e-3` | `1.713` |

Post-norm at eight blocks cannot use the learning rate that the rest of the ladder wants, and lowering it to rescue the baseline makes every later milestone worse.
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
- Keep the parameter budget in roughly the same neighborhood across milestones, so a loss difference is not obviously just a size difference. Aim for round, readable widths rather than exact parity: when a mechanism changes the natural width, narrow it sensibly and record the resulting parameter delta instead of contorting the constants to cancel it.
- Prefer explicit tensor math over fused framework calls while the mechanism is the lesson. `scaled_dot_product_attention` and other fused paths belong to phase 4 profiling work, not here.
- Report wall-clock honestly. At this scale most modern efficiency mechanisms are slower, because their value is asymptotic and their fast implementations are kernels this repo has not written yet.
- A modernization that does not improve loss at this scale is still kept if it is a genuine 2026 standard, but the learning log must say plainly that it did not pay for itself here and why.
- **A mechanism whose failure mode does not occur at this scale must say so rather than perform the fix.** `M-510` established this the hard way: routing never collapsed, so balancing it would have been a ritual.
- Do not build a framework. Standalone numbered scripts, module-level configuration constants, shared code only for artifacts.
- Keep the learning log tied to completed milestones.

## Milestones

### Milestone 501: Vanilla Decoder-Only Transformer
Track: Baseline

**Goal.** Establish the pre-modern reference point that every later milestone is measured against.

**The problem.** There isn't one yet — this milestone *creates* the problems the rest of the ladder solves. Its job is to be a faithful 2017/GPT-2 decoder so that later deltas are attributable. The one thing it must not be is a strawman: every component here was state of the art for years.

**What the field tried.** The 2017 transformer replaced recurrence with attention because recurrence forced sequential computation and a fixed-size state between timesteps. The decoder-only variant then dropped the encoder and cross-attention once it became clear that a causal LM objective on one stack scaled better than encoder-decoder pretraining for generative use.

**The solution that won.** The block that every model still uses: a causal attention sublayer and a position-wise feed-forward sublayer, each wrapped in a residual connection and a normalization, stacked `N` times. Nothing below replaces this skeleton — every later milestone changes what goes *inside* one of those two sublayers, or how the normalization and residual are wired around them.

**Implementation decisions.**
- Learned absolute position embeddings, added to the token embedding.
- Biases on every linear layer, including q/k/v/o — this is what 503 removes.
- Post-norm: `x = norm(x + sublayer(x))`.
- GELU feed-forward at `4x` width.
- Tied input and output embeddings, since at a `98`-character vocabulary untying is pointless.
- Scale attention scores by `sqrt(d_head)`, never `sqrt(d_model)`. The scale is per-head.

Status: complete via [`phase5/001_vanilla_decoder.py`](../../phase5/001_vanilla_decoder.py), recorded as `P5-001`.

Main lesson:
- At the control learning rate the baseline never learns. It reaches character-unigram loss (`3.055`) by step `250` and stays flat for the remaining `2750` steps, moving by `0.003` in either direction.
- This is the intended result rather than a bug. Post-norm puts the normalization on the residual stream itself, so there is no identity path from the loss back to the embedding, and no single learning rate serves the whole stack.
- It is worth seeing a real training collapse once, and worth knowing that it looks like a flat line rather than a divergence.

### Milestone 502: Pre-Norm Residual Stream
Track: Normalization

**Goal.** Move the normalization inside the residual branch and add a final norm before the output projection.

**The problem.** In post-norm, the residual stream passes *through* the normalization: `x = norm(x + f(x))`. That means there is no path from the loss back to the embedding that is the identity. Every layer's normalization rescales the gradient on its way down, and those rescalings compound multiplicatively over depth. At `8` blocks the result is measurable, and the failure is not divergence but the opposite — the gradients arriving at early layers are too small and too badly conditioned for one global learning rate to serve the whole stack.

**What the field tried.** The first fix was **learning-rate warmup**, which works and is why every 2017-era recipe has it: spend a few thousand steps at a tiny learning rate while the layers co-adapt, then ramp. The second was careful **initialization scaling** — schemes that shrink the residual branch by a factor depending on depth so the sum stays well-conditioned. Both are real fixes, and both are workarounds: they manage a badly-shaped optimization problem instead of reshaping it.

**The solution that won.** Move the norm inside the branch:

```
x = x + attn(norm(x))
x = x + ffn(norm(x))
```

Now the residual stream itself is never normalized. There is a clean additive path from the embedding to the logits — the gradient reaches layer `0` with an unmodified term, plus whatever the branches contribute. Depth stops multiplying and starts adding. The cost is that the stream's magnitude now grows with depth, since nothing rescales it, which is why pre-norm requires **one final norm** before the LM head. That final norm is not decoration; without it the logits inherit the stream's accumulated scale.

**Implementation decisions.**
- Two norms per block, both inside the branch, plus one `out_norm` at the end of the stack.
- Keep the same learning rate as 501, so the comparison is a single variable.
- Do not add warmup. The whole point is that pre-norm does not need it.

Status: complete via [`phase5/002_pre_norm.py`](../../phase5/002_pre_norm.py), recorded as `P5-002`.

Main lesson:
- Moving the norm inside the branch is worth `-0.7980` for four lines of code and `512` parameters. That is the single largest structural gain available.
- But it does not finish the job. The curve reaches `2.34` by step `250` and then crawls, ending at `2.2576` after `2750` more steps. The model is learning, barely.
- So pre-norm removes the hard blocker and exposes a second one. The next two milestones show that the remaining blocker is the position representation, not the normalization.

### Milestone 503: RMSNorm And Bias Removal
Track: Normalization

**Goal.** Replace LayerNorm with RMSNorm and remove biases from every linear layer.

**The problem.** LayerNorm does two things — re-center and re-scale — and carries two parameter vectors, gain and shift. Every linear layer additionally carries a bias. All of it costs parameters, memory traffic, and in LayerNorm's case a second reduction pass over the feature axis. The question is whether any of it is load-bearing.

**What the field tried.** The honest answer is that the field mostly *assumed* mean-centering mattered, because LayerNorm inherited the framing from BatchNorm, where centering genuinely does address covariate shift across a batch. The RMSNorm paper tested the assumption directly by ablating re-centering and finding it contributed essentially nothing to quality while costing a reduction. Bias removal followed a similar path: at scale, biases are a vanishing fraction of parameters but a real fraction of the optimizer state and the outlier surface, and removing them turned out to be free.

**The solution that won.** Normalize by the root mean square instead of the standard deviation, and keep only a gain:

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g
$$

versus LayerNorm's

$$
\mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot g + b
$$

RMSNorm fixes the vector's *length* and leaves its direction alone. LayerNorm additionally projects out the all-ones component first. All three frontier models use RMSNorm exclusively, and Kimi K3 goes further, stripping biases even from its vision encoder's projections and reporting that this "further stabilizes the from-scratch optimization."

**Implementation decisions.**
- `bias=False` on every `nn.Linear` in the model.
- `eps = 1e-5`, applied inside the square root.
- Use `x.pow(2).mean(-1)`, **not** `x.var(-1, correction=0)`. These agree only when the per-token mean is zero, and pre-norm is exactly the regime where the residual stream drifts off zero. This is a real trap that cost a debugging cycle.
- Use `torch.rsqrt`, which is one op instead of a divide plus a sqrt.

Status: complete via [`phase5/003_rms_norm.py`](../../phase5/003_rms_norm.py), recorded as `P5-003`.

Main lesson:
- Dropping mean-centering and every bias costs nothing and saves `22784` parameters, which is the claim RMSNorm was introduced to make.
- The `-0.0951` improvement should not be read as a win for RMSNorm. The run is still in the stalled regime that `M-002` left it in, where the loss drifts slowly, and a drift of that size over `3000` steps is not attributable to the normalizer.
- Implementation trap worth keeping: normalizing by `x.var(correction=0)` instead of the mean square agrees with RMSNorm only when the per-token mean is zero, and pre-norm is exactly where the residual stream drifts off zero.

### Milestone 504: Rotary Position Embeddings
Track: Positions

**Goal.** Replace learned absolute positions with RoPE applied to queries and keys.

**The problem.** A learned position table gives every absolute index its own free vector. Three things follow, all bad. First, position information is *added* to the token embedding at layer `0` and must survive eight layers of mixing to still be usable. Second, the model must *learn* that only differences matter — nothing in the parameterization says position `100` and `102` relate the way `3` and `5` do. Third, and fatally, index `4096` has no vector if you only trained to `2048`; the table cannot extrapolate at all.

**What the field tried.** **Sinusoidal** encodings (2017) fixed extrapolation by making the table a fixed function of the index, but kept the additive-at-layer-0 problem, and in practice models did not extrapolate well anyway. **Learned absolute** (GPT-2) traded extrapolation for fit. **T5 relative bias** added a learned scalar per bucketed distance directly to the attention logits — genuinely relative, but a per-layer additive bias with limited expressivity. **ALiBi** simplified that to a fixed linear penalty on distance, which extrapolates beautifully but hard-codes a recency prior. Each fixed one thing.

**The solution that won.** Do not add position to the content; **rotate** the query and key by an angle proportional to their position. Split each head's `d_head` features into `d_head/2` pairs, and rotate pair `i` at position `m` by angle `mθ_i`, where `θ_i = base^(-2i/d_head)`.

The whole trick is one property of rotation matrices:

$$
R(\alpha)^\top R(\beta) = R(\beta - \alpha)
$$

So when a query at position `n` meets a key at position `m`:

$$
\langle R(n\theta) q,\; R(m\theta) k \rangle = q^\top R(n\theta)^\top R(m\theta) k = q^\top R\big((m-n)\theta\big) k
$$

The absolute positions cancel and only `m - n` survives. Relative position is not learned or approximated; it is algebraically guaranteed. And because rotations are length-preserving, position can be injected at every layer without disturbing activation scale — which an additive table cannot do.

The frequency spread is what makes it work across scales: high-`θ` pairs spin fast and resolve adjacent tokens, low-`θ` pairs spin slowly and encode coarse long-range position.

**Implementation decisions.**
- Apply to `q` and `k` only, never `v`. Position belongs in *which* token to attend to, not in *what* is retrieved.
- Apply **after** the head split, per head.
- Use the **split-half** convention (`rotate_half` pairs feature `i` with `i + d_head/2`), matching HuggingFace, rather than the interleaved convention in the original paper. They are equivalent up to a permutation of the feature axis, but mixing them silently breaks weight compatibility.
- `ROPE_BASE = 10000.0`, precomputed `cos`/`sin` buffers of shape `[T, Dh]`.
- Frontier note: nobody rotates the full head any more. DeepSeek-V4 and GLM-5.2 both apply RoPE to only the **last `64` dimensions**, and Kimi K3 uses none at all. Full-head RoPE is the pedagogically correct starting point and is already legacy.

Status: complete via [`phase5/004_rope.py`](../../phase5/004_rope.py), recorded as `P5-004`.

Main lesson:
- This is the milestone that makes the model work. Validation loss falls from `2.1625` to `0.9334`, a gain of `-1.2291`, while the model *loses* `65536` parameters.
- Nothing else in the ladder comes close, and the reason is that the previous three milestones were all stalled on the same thing. A learned absolute position table gives every index its own free vector and forces the model to discover that only differences matter. RoPE makes the score between positions `m` and `n` depend on `n - m` by construction, so that discovery is not needed.
- The rotation is also length-preserving, so position is injected without disturbing activation scale. That is the mechanical reason it can be applied at every layer while an additive table cannot.
- The honest reading of milestones `002` through `004` together: pre-norm was necessary but not sufficient, and the learned position table was the binding constraint all along.

### Milestone 505: Grouped-Query Attention
Track: Attention

**Goal.** Share key and value heads across query heads.

**The problem.** This one is not about quality at all, and pretending otherwise wastes the milestone. During autoregressive generation every past token's keys and values must be kept in memory. That cache is `2 · L · H · d_head` numbers per token, and at long context it dominates everything — it is larger than the weights, and every decode step must re-read all of it. Decoding is therefore **memory-bandwidth bound**, not compute bound. Halving the cache nearly doubles decode throughput.

**What the field tried.** **Multi-query attention** (MQA) went straight to the extreme: one shared K/V head for all query heads, an `H`-fold cache reduction. It works, but quality degrades measurably and training becomes less stable. The lesson was that the endpoints are wrong.

**The solution that won.** Interpolate. Partition `H_q` query heads into `H_kv` groups; every head in a group shares one K/V head. `H_kv = H_q` is plain MHA, `H_kv = 1` is MQA, and everything between is GQA. Cache shrinks by `H_q / H_kv` while each query head keeps its own query projection and thus its own view. Empirically quality is close to MHA at `H_kv` around `8`, which is why essentially every open model from 2023 onward ships GQA as the floor.

**Implementation decisions.**
- `NUM_Q_HEADS = 4`, `NUM_KV_HEADS = 2`, group size `2`.
- Expand with `repeat_interleave`, **not** `repeat`. `repeat_interleave` produces group-contiguous pairing (`[k0, k0, k1, k1]`), matching HuggingFace; `repeat` produces `[k0, k1, k0, k1]` and pairs the wrong heads. Note that `x.repeat(n, dim=2)` is not even a valid call — `Tensor.repeat` does not take a `dim`.
- Expect **no quality signal** at this scale. Training has no KV cache, so the mechanism's entire benefit is invisible here. Report the cache arithmetic explicitly instead; that is the actual deliverable.

Status: complete via [`phase5/005_gqa.py`](../../phase5/005_gqa.py), recorded as `P5-005`.

Main lesson:
- Sharing key and value heads improved the loss by `-0.0282` while removing `524288` parameters, roughly `9%` of the model.
- The loss change is small enough that it should be read as "no cost" rather than as a gain; a single seed cannot resolve `0.03` reliably. The parameter saving is the real result.
- The mechanism's actual payoff is invisible here by construction. GQA exists to shrink the KV cache during autoregressive decoding, and training never builds one.

### Milestone 506: SwiGLU Feed-Forward
Track: Feed-forward

**Goal.** Replace the GELU MLP with a gated SwiGLU feed-forward block at matched parameter count.

**The problem.** `down(act(up(x)))` can only ever *add* contributions. Each output feature is a weighted sum of nonlinear functions of the input, and the nonlinearity is fixed and elementwise. There is no way for one part of the representation to *modulate* another — no multiplicative interaction, no data-dependent routing of information through the block.

**What the field tried.** The GLU family was explored systematically: swap the activation for a product of two projections, one passed through a nonlinearity acting as a gate. The original GLU used a sigmoid gate. Variants tried ReLU (`ReGLU`), GELU (`GEGLU`), and Swish (`SwiGLU`). The comparison paper that settled it tested them head-to-head and famously declined to explain the result, attributing the win to "divine benevolence." That honesty is worth preserving: **there is still no accepted theory for why SwiGLU wins**, and Kimi K3's 2026 report says the same, that "a complete account of its empirical effectiveness remains open."

**The solution that won.** Three matrices instead of two:

$$
\mathrm{SwiGLU}(x) = W_{\text{down}}\big(\mathrm{Swish}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x\big), \qquad \mathrm{Swish}(z) = z\,\sigma(z)
$$

The gate branch scales the up branch elementwise, per token and per feature — a genuine multiplicative interaction. Swish specifically matters because it dips **below zero** near the origin, so the gate can *flip a feature's sign* rather than only attenuate it. A sigmoid gate bounded in `[0, 1]` cannot do that, which is a large part of why SwiGLU beat plain GLU.

**Implementation decisions.**
- **Narrow the block when you add the gate.** A gated block has three matrices where the dense block had two, so `2/3` of the old width keeps it in the same neighborhood: `D_FFN = 8 * D_MODEL // 3`. This is a real convention, not bookkeeping — it is where Llama's odd `11008` comes from (`8 * 4096 // 3` rounded up to a multiple of `256`). We skip the rounding, so `1024` becomes `682`.
- `F.silu` is Swish; they are the same function.
- Keep this milestone's activation **unbounded**. The cap is milestone 512's job, and merging them would confound gating with bounding.

Status: complete via [`phase5/006_swiglu.py`](../../phase5/006_swiglu.py), recorded as `P5-006`.

Main lesson:
- At matched parameters SwiGLU made the model **worse**, by `+0.0711`. This is the clearest negative result in the ladder and it is large enough to be real rather than noise.
- The comparison is clean, which is what makes it interesting. A gated block has three matrices where the dense block had two, so `D_FFN` narrowed from `1024` to `682` and the parameter count barely moved, from `5796608` to `5792512`.
- So the gate bought a multiplicative interaction and paid for it with `33%` less width, and at this scale the width was worth more. The field's preference for SwiGLU is established at far larger widths, where the trade goes the other way.
- Worth stating plainly: this ladder now contains a mechanism that every frontier model uses and that measurably hurt at `6M` parameters.

### Milestone 507: QK-Norm And Gated Attention
Track: Attention stability

**Goal.** Add RMSNorm to queries and keys, and a data-dependent sigmoid gate on the attention output.

**The problem.** Two distinct failure modes on the attention path, both about magnitudes.

First, **attention logits explode.** The score `q·k/√d` has no bound. As `q` and `k` projections grow during training, some logits grow with them, softmax saturates to one-hot, gradients through that head vanish, and the head effectively dies. In `bf16` the logit can also simply overflow. This is a leading cause of loss spikes at scale.

Second, **softmax cannot output nothing.** Attention weights are forced to sum to `1`, so a head that has found nothing relevant must still return a weighted average of something. Models solve this by dumping attention onto an arbitrary token — usually the first — producing the **attention sink**. The sink's value vector then flows into the residual stream as a large, meaningless activation, and those *massive activations* are precisely the outliers that break low-precision inference and quantization.

**What the field tried.** For exploding logits: **logit soft-capping** (`tanh`-based, used in Gemma 2) bounds the score directly but is a nonlinearity in the hot path and interferes with fused attention kernels. **QK-Clip**, from the Muon line of work, clips the q/k *weights* post-update whenever a logit exceeds a threshold — effective but a post-hoc optimizer intervention. For sinks: **StreamingLLM** established the diagnosis and worked around it by just always keeping the first few tokens in the cache, which manages the symptom rather than removing it.

**The solution that won.** Two independent mechanisms, both now near-universal.

**QK-Norm** applies RMSNorm to `q` and `k` per head, right before the score:

$$
\text{score} = \frac{\mathrm{RMSNorm}(q) \cdot \mathrm{RMSNorm}(k)}{\sqrt{d_{\text{head}}}}
$$

Once both vectors have fixed length, the dot product is bounded by `d_head` regardless of how large the projections grow. The logit magnitude is now structurally controlled rather than monitored. DeepSeek-V4 states this outright: because they RMSNorm the queries and KV entries, they "do not employ the QK-Clip technique" at all. The architectural fix retired the optimizer patch.

**Gated attention** multiplies the attention output by a data-dependent sigmoid gate before the output projection:

$$
y = W_o\big(\sigma(W_g x) \odot \mathrm{attn\_out}\big)
$$

This is the direct fix for the sink. The head is no longer forced to emit something: if the gate closes, the output is near zero regardless of what softmax was forced to do. The paper that introduced it ablated roughly thirty variants at `15B` MoE scale and found it adds non-linearity, introduces input-dependent sparsity, and eliminates the sink — reporting reduced massive activations and better `bf16` stability as a consequence. Kimi K3 uses exactly this on **both** its KDA and MLA layers.

**Implementation decisions.**
- QK-Norm is **per head**, over the `d_head` axis, with its own learned gain — not one norm over the flattened `d_model`.
- Normalize **after** the head split and **before** RoPE is not required either way here, but keep the ordering fixed once chosen; `M-508` regressed once by letting `repeat_kv_heads` drift above the RoPE block.
- The gate is **elementwise** (`d_model` values), not per-head (`H` values). This was decided on evidence: the source paper's default is `elementwise_attn_output_gate: true, headwise_attn_output_gate: false`, and Kimi K3 explicitly upgraded its gate from low-rank to **full rank** in K3. The granularity is part of the mechanism, not a cost knob.
- The gate reads `x`, the block input — not the attention output. It is data-dependent on the token, not on what attention retrieved.
- Accept that this adds `~9%` parameters and record that the loss gain is therefore confounded with capacity.

Status: complete via [`phase5/007_qk_norm_gated_attention.py`](../../phase5/007_qk_norm_gated_attention.py), recorded as `P5-007`.

Main lesson:
- The second-largest gain of the ladder, `-0.1859`, taking validation loss below `0.80` for the first time.
- It is also the least attributable. The elementwise gate is a full `D x D` matrix, so the model grew by `524800` parameters, about `9%`. Some of the improvement is the mechanism and some is the capacity, and this run does not separate them.
- The cheap control, if it is ever wanted, is a QK-Norm-only variant: the two norms cost `64` parameters between them.

### Milestone 508: Layerwise Hybrid Attention
Track: Attention layout

**Goal.** Interleave three sliding-window local attention layers with one full global attention layer, and remove RoPE from the global layers.

**The problem.** Attention is `O(T²)` in both time and memory. At `1M` tokens that is `10¹²` pairs per head per layer — not a constant-factor problem, an asymptotic one. But the deeper observation is that full attention in *every* layer is redundant: most heads in most layers attend locally anyway, so the quadratic cost buys global reach that most layers never use.

**What the field tried.** **Sparse/strided patterns** (Sparse Transformer) fixed the pattern in advance — cheap, but the pattern is a guess. **Longformer/BigBird** combined sliding windows with a few designated global tokens, which works but requires choosing those tokens. **Sliding window everywhere** (Mistral) is simple and relies on receptive field growing with depth — `L` layers of window `W` reach `L·W` — but that reach is indirect, mediated through intermediate representations, and it measurably hurts retrieval. **Full attention everywhere** remains the quality ceiling and the cost floor.

**The solution that won.** Put the two in the *same stack* at a fixed ratio. Most layers are cheap and local; a periodic minority are global and carry long-range information. All three frontier models do a version of this, and the ratio has converged on roughly `3:1`:

- **Kimi K3**: `3` KDA layers to `1` gated MLA layer, repeated, **plus one extra global layer at the very end** so the final layer always sees everything. `69` KDA + `24` MLA = `93`.
- **DeepSeek-V4**: CSA and HCA interleaved, plus a sliding-window branch inside *both*.
- **GLM-5.2**: one full indexer every `4` layers, the other three reusing its selection (IndexShare).

The second half of this milestone is **NoPE on the global layers**. If the local layers already carry RoPE, position is in the residual stream by the time a global layer reads it; the global layer's job is unrestricted content matching, and forcing a distance-dependent phase onto it only limits its reach. Kimi K3 took this to its conclusion and removed positional encoding **entirely**, letting the KDA recurrence carry position implicitly — which, they note, also means no RoPE base to retune when extending context.

**Implementation decisions.**
- Ratio `3:1`, global at indices `3` and `7` with `NUM_BLOCKS = 8`, so the **last layer is always global**. This matches K3's explicit choice.
- Window `64` at context `256`.
- The local mask is `triu(1) | tril(-W)` — block the future *and* the distant past. Getting only the first half gives plain causal attention and silently invalidates the milestone.
- Masks are buffers, so this milestone changes **zero parameters**. That makes it the cleanest comparison in the ladder.
- **Treat the result as a measurement, not a confirmation.** MiniMax trained this exact family of variants at scale and found them worse on retrieval and multi-hop reasoning. A negative result here is a real result.

Status: complete via [`phase5/008_hybrid_attention.py`](../../phase5/008_hybrid_attention.py), recorded as `P5-008`.

Main lesson:
- Six of eight layers lost `75%` of their receptive field and the loss did not move: `-0.0017` at **identical** parameter count.
- This is the cleanest comparison in the ladder, since masks are buffers and NoPE removes rather than adds. Nothing but the receptive field changed.
- At context `256` with a `64`-token window that is the expected result. The mechanism's value is asymptotic, and `256` tokens is not long. It is evidence that the layout is free here, not evidence that it is free.

### Milestone 509: Multi-Head Latent Attention
Track: Attention

**Goal.** Replace the global layers' key/value projections with a compressed latent representation that is up-projected during attention.

**The problem.** GQA shrinks the KV cache by sharing heads, but it does so by *removing* heads — every query head in a group sees literally the same keys. That is a real capacity loss, and it is a coarse knob: the cache only shrinks in integer divisors of the head count.

**What the field tried.** MQA and GQA, covered at `M-505`, both trade heads for memory. The question MLA asks is whether the cache can shrink *without* any head sharing at all.

**The solution that won.** Cache a **compressed latent** instead of the keys and values. Project each token down to `c_t = W_c x_t` of dimension `d_latent`, cache only `c_t`, and reconstruct per-head keys and values with learned up-projections during attention. Every query head gets its own distinct keys and values again; the compression is along the *feature* axis rather than the head axis.

The reason it is efficient is the **absorption trick**. The score is

$$
q^\top (W_{uk}\, c) = (W_{uk}^\top q)^\top c
$$

so `W_uk` can be folded into the query projection once, and attention runs directly against the cached latent — the keys are never materialized at all.

**And that is exactly what breaks RoPE.** Absorption requires the matrix sitting between query and latent to be *constant*. RoPE inserts `R(m-n)`, which depends on the key's position, so a folded query would serve exactly one key. The fix is **decoupled RoPE**: split the head into content dims (compressed, absorbed, no rotation) and a few rope dims (uncompressed, rotated, shared across heads). Scores are computed on the concatenation, which is exact because a dot product over a concatenation is the sum of the dot products over the pieces. Values stay content-width only — rope dims decide *how much* to attend and carry nothing to retrieve.

**Implementation decisions.**
- Apply MLA to **global layers only**; keep GQA on local layers. This matches every frontier model — nobody pays MLA's cost on a local mixer.
- `D_LATENT = 64` and `D_ROPE = D_HEAD // 2`. The rope fraction is real: a third of each score head carries position, as in DeepSeek V3. The latent is a plain number, because no published rule transfers — DeepSeek's `4 * d_head` gives `128` here, which is half the model dim and barely compresses. At `64` the cache holds `64 + 16 = 80` numbers against `512` for full multi-head KV, a `6.4x` squeeze. Real MLA models reach far more: GLM-5.2 caches `576` against `32768`, or `57x`. **DeepSeek's own sizing rule does not transfer**: they use `4 · d_head`, which here is `4 * 32 = 128`, exactly half the model dim and only a `2x` compression. Their `28x` saving comes from having `128` heads of `128` dims to squeeze into `512`; an `8`-head model has far less redundancy to exploit, so the latent is set as a fraction of the model dim instead.
- Implement the **decoupled-RoPE form even though `M-508` made these layers NoPE**, because that conflict is the entire reason MLA has its shape. Then compare against the NoPE form the ladder can actually use.
- **Normalize the latent, not the reconstructed key.** This is the decision that is easiest to get
  backwards, and getting it backwards silently destroys the mechanism. A norm applied to
  `W_uk c` is a per-key rescale sitting between the latent and the query, so `W_uk` can no longer
  be folded into the query and the absorption trick is gone — measured error `4.54` on random
  inputs against `7e-7` when the norm sits on the latent. DeepSeek's own implementation calls this
  `kv_a_layernorm` and applies it to the compressed vector before `kv_b_proj`. So the global layer
  carries `q_norm` at `D_HEAD` on the query content path and `kv_norm` at `D_LATENT` on the latent,
  and it has no key norm at all. The local GQA layers keep their ordinary `k_norm`, since they have
  no latent to normalize.
- Record the frontier context honestly: **DeepSeek-V4 has moved past MLA**, to shared-KV MQA plus compression along the *sequence* axis, on the grounds that at `1M` tokens sequence length dominates memory, not head count. GLM-5.2 still uses MLA. This milestone builds a mechanism that is simultaneously current and being superseded.

Status: complete via [`phase5/009_mla.py`](../../phase5/009_mla.py), recorded as `P5-009`.

Main lesson:
- Latent attention cost `+0.0248` for `41024` more parameters. A small regression, near the resolution of a single seed.
- The cache arithmetic is the actual deliverable. A global layer now caches `64 + 16 = 80` numbers per token against `512` for full multi-head keys and values, a `6.4x` reduction, and every query head gets its own keys and values back rather than sharing.
- DeepSeek's own sizing rule does not transfer. They set the latent to `4 * d_head`, which here is `128`, exactly half the model dim and barely a compression. Their `28x` saving comes from having `128` heads of `128` dims to squeeze into `512`; an eight-head model has far less redundancy to exploit.
- The decoupled rope path was implemented even though these layers are NoPE, because that conflict is the entire reason MLA has its shape. Absorption requires the map from latent to key to be linear and constant, which is why the norm sits on the latent and there is no key norm at all.

### Milestone 510: Sparse Mixture-Of-Experts Feed-Forward
Track: Sparsity

**Goal.** Replace the dense feed-forward block with fine-grained routed experts plus a shared expert, keeping block `0` dense.

**The problem.** Capacity and compute are welded together. Every parameter in a dense feed-forward block runs for every token, so the only way to make the model know more is to make every forward pass more expensive. But most tokens do not need most of the network — the knowledge required to continue `" the cat sat on the "` and to continue a line of Rust are almost disjoint.

**What the field tried.** **Sparsely-gated MoE** (Shazeer, 2017) established the shape: `N` expert FFNs, a router picking a few per token. It worked but was fragile. **Switch Transformer** simplified to top-`1` and scaled it, exposing the real problems — load imbalance, token dropping, and instability. The `2020-2022` generation used few, wide experts (`8` or `16` experts at full FFN width), and found that experts had to specialize in coarse, overlapping ways because there were so few of them.

**The solution that won: DeepSeekMoE's two changes.**

**Fine-grained experts.** Instead of `8` experts at width `d_ff`, use `N·8` experts at width `d_ff/N`, and activate `N·k` of them. Active parameters are identical, but the number of *combinations* the router can express explodes — with `256` experts choosing `8`, there are `~10¹⁴` possible routings versus `28` for `8` choose `2`. Specialization becomes finer because each expert is smaller and can afford to be narrow. The frontier has pushed hard in this direction: `256` experts (GLM-5.2), `384` (DeepSeek-V4-Pro), `896` (Kimi K3).

**Shared experts.** Reserve one or two experts that **every** token uses, unrouted. Without them every routed expert must independently learn the common transformations that all tokens need — grammar, basic syntax — wasting capacity on redundancy. The shared expert absorbs the common case so routed experts can afford to be genuinely specialized. All three frontier models have them: `1` (V4, GLM), `2` (K3).

**Implementation decisions.**
- **Keep active parameters per token near the dense block.** `4` routed experts at `128` plus one shared at `128` gives an active width of `640` against the dense `768`, so the mixture is in the same neighborhood rather than exactly matched. Exact matching would force ugly widths; near enough keeps the comparison meaningful without that.
- Use an `Expert` class in an `nn.ModuleList` with a loop, not a batched 3D-tensor formulation. The loop is `O(N)` Python and slow, but it is the readable form and this phase optimizes for mechanism. Production speed comes from fused grouped-GEMM kernels, which is a phase-4 concern.
- Dispatch via `(chosen == i).nonzero(as_tuple=True)` to get `(token_index, slot)`, then `index_add_` the weighted expert output. This is the standard scatter/gather shape without obscure ops.
- Sigmoid router, top-`k`, then renormalize the chosen weights to sum to `1`. **The renormalization is for output scale, not competition** — it is not a softmax, there is no exponential. Sigmoid gives each expert an independent, bounded score, which matters at `M-511` because a fixed bias step then means the same thing for every expert.
- Track `expert_load` in a **non-persistent buffer** via `torch.bincount(chosen.flatten(), minlength=N)`. `minlength` is essential: without it a dead expert vanishes from the statistic entirely instead of showing as zero.
- Keep block `0` dense. Note that the frontier disagrees here — GLM keeps `3` dense, K3 keeps `1`, and **DeepSeek-V4 keeps none**, using hash routing on the first `3` MoE layers instead. The stated reason for dense-early is that the router needs decent features to route on, and layer-`0` hidden states are not good enough yet.

Status: complete via [`phase5/010_moe.py`](../../phase5/010_moe.py), recorded as `P5-010`.

Main lesson:
- Total and active parameters diverge for the first time: total rises `40%` to `8899392` while the active feed-forward width per token is `4 * 128 + 128 = 640` against the dense `682`, so each token costs slightly less than before.
- Validation loss improved `-0.0133`, which is inside what a single seed can resolve. The honest statement is that `1.5x` the feed-forward capacity at slightly lower active cost bought nothing measurable.
- Wall-clock is the real cost: `956.5s` against `737.3s`, `30%` slower. Seven blocks now run eight small matrix multiplications where they ran one large one, and the shapes are too small to use the GPU well. Production speed comes from fused grouped-GEMM kernels.
- - **Routing is measurably imbalanced.** Tracked on training dispatch, the busiest expert takes `0.177` of the tokens and the quietest `0.038`, against an ideal `0.125` — a `4.6x` spread that appears within `250` steps and then holds steady. No expert dies, because `4` of `8` is barely sparse enough for the winner-take-all dynamic to run away. That gap is what `M-511` acts on.

### Milestone 511: Real Sparsity, Then Auxiliary-Loss-Free Load Balancing
Track: Sparsity

**Goal.** Make the routing genuinely sparse, then balance it with a per-expert bias applied to selection only.

**What `M-510` measured.** Routing is imbalanced, and the gap is large enough to act on. Tracked
on training dispatch, the busiest expert takes `0.177` of the tokens and the quietest `0.038`,
against an ideal `0.125`:

| Step | min share | max share | unused |
| ---: | ---: | ---: | ---: |
| 1 | `0.094` | `0.148` | `0` |
| 250 | `0.032` | `0.177` | `0` |
| 3000 | `0.038` | `0.177` | `0` |

The spread appears within `250` steps and then holds rather than widening. Nothing dies, and the
reason is that `4` of `8` is barely sparse by frontier standards:

| Model | Active of total | Ratio |
| --- | --- | ---: |
| DeepSeek-V4-Pro | `6` of `384` | `1.6%` |
| Kimi K3 | `16` of `896` | `1.8%` |
| GLM-5.2 | `8` of `256` | `3.1%` |
| this ladder | `4` of `8` | `50%` |

Every token uses half the experts, so even an unpopular one keeps receiving gradient and the
winner-take-all dynamic stabilizes instead of running away. That makes this milestone tractable in
two stages: balance the load that exists, and separately ask what happens when the routing is made
genuinely sparse.

**The problem.** Routing has positive feedback built in. An expert that receives slightly more tokens early gets more gradient, becomes better at what it sees, so the router prefers it more, so it receives more tokens. Rich-get-richer, ending with a handful of experts doing everything and the rest dead — you paid for `N` experts and got `k`. There is a second, independent problem at scale: experts are sharded across devices, and throughput is set by the *most loaded* expert, so imbalance directly burns money even when quality is fine.

**What the field tried.**

1. **Capacity factor with token dropping** (GShard, Switch Transformer). Give each expert a fixed buffer of `(tokens/experts) · capacity_factor`; tokens arriving at a full expert are **dropped** and pass through on the residual alone. This solves the systems problem by construction — fixed buffers, static shapes, no stragglers — but it is brutal: some tokens are never processed, which token gets dropped depends on arbitrary ordering, and inference does not drop, so train and inference disagree.

2. **An auxiliary load-balancing loss** (Switch Transformer, the standard for years). Add a penalty to the objective:

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i P_i
$$

with `f_i` the fraction of tokens routed to expert `i` and `P_i` its mean router probability; the product is minimized at uniform. The flaw is structural — **it is a second objective fighting the first.** Its gradients flow into the router and pull routing away from what the LM loss wants. Too small an `α` and it still collapses; too large and quality suffers. There is no principled value and it needs retuning per model. DeepSeek names exactly this as the motivation for replacing it.

**The solution that won.** The insight is sharp: **you do not need a gradient to fix load — you need to change which experts get selected.** Those are different things, and the auxiliary loss conflated them.

Add a per-expert bias to the score used for **selection only**:

$$
\mathcal{T}_i = \operatorname{argtopk}\big(s_i + b\big), \qquad p_{i,j} = \frac{s_{i,j}}{\sum_{r \in \mathcal{T}_i} s_{i,r}}, \quad j \in \mathcal{T}_i
$$

The bias appears in the `topk` and **nowhere in the weights**. Kimi K3 states the consequence precisely: because `b` is omitted from `p`, "it regulates dispatch without altering the mixture weights or the gradient-based optimization of the router." Update it outside backprop with a fixed step:

$$
b_j \leftarrow b_j + \gamma \cdot \operatorname{sign}\big(\bar{\ell} - \ell_j\big)
$$

Overloaded experts get nudged down, starved ones up. No new loss term, no coefficient trading against the LM objective.

**The 2026 refinement — Quantile Balancing (Kimi K3).** The fixed-step rule has a real weakness: `γ` trades slow adaptation against oscillation, and at `896` experts it is too crude. QB **solves** for the bias instead of nudging it. Route with top-`(k+1)` instead of top-`k`; the `(k+1)`-th entry is the cutoff `α_i` that an expert must beat to enter token `i`'s top-`k`. Then set each expert's bias to the quantile of its margins that yields exactly its target load `q = mk/n`:

$$
\hat{b}_j^{(t+1)} \leftarrow -\operatorname{quantile}_{1-k/n}\big(s_{:,j} - \alpha^{(t)}\big), \qquad b^{(t+1)} \leftarrow \hat{b}^{(t+1)} - \operatorname{mean}\big(\hat{b}^{(t+1)}\big)\mathbf{1}
$$

One pass, no step size. The mean-subtraction removes a common offset that would not change top-`k` anyway. At scale the exact quantile is unaffordable, so they read it from a histogram of margins with a single all-reduce of bin counts.

**The arc worth remembering: drop tokens → penalize with a loss → bias the selection → solve for the bias exactly.**

**Implementation decisions.**
- **Change the sparsity ratio first**, as a separate measured step, and report load statistics before adding any balancer. Target roughly `3%` — the GLM ratio, the mildest of the three. Holding active hidden units at `344`, that means on the order of `32` to `64` experts, which multiplies total feed-forward parameters by roughly `7x` and lengthens the Python expert loop proportionally. **State the cost plainly**: sparsity is only economical when the total budget is large, which is exactly why every model using fine-grained routing is enormous.
- Register `expert_bias` as a **buffer, not a parameter** — it must never receive a gradient.
- The `topk` no longer returns the weights. Select on `scores + bias`, then `gather` the weights from raw `scores`. Conflating these silently reintroduces the distortion the whole design exists to avoid, and it is the single most likely bug in this milestone.
- Update the bias under `no_grad` and **only when `self.training`**. The update uses the current batch's load and applies to the *next* batch — a batch must never be routed with a bias derived from itself.
- `γ = 0.001`, matching DeepSeek-V3.
- **Quantile Balancing is worth implementing over the fixed-step rule.** The expectation was that QB only pays off at hundreds of experts with distributed histogram estimation, and that the sign rule would do at this size. That was wrong in a useful way: QB needs no step size, converges in one pass rather than adapting over many, and its per-expert bias is directly readable as how much the router wants to be imbalanced. At `64` experts on one GPU the exact quantile is a single `sort` per layer, so the histogram estimator is unnecessary and the simple form is not simpler.
- Keep the **sigmoid** router. DeepSeek-V4 switched to `Sqrt(Softplus)`, but two of three frontier models still use sigmoid, and one lab changing its mind once is not a trend. Sigmoid's bounded `(0,1)` range is also what makes a fixed `γ` mean the same thing for every expert.
- **Skip DeepSeek's sequence-wise balance loss.** It is a safety net against extreme within-sequence imbalance at trillion scale, and adding it would reintroduce exactly the auxiliary-loss coupling this milestone is about removing.
- Track per-expert load, min/max share, and dead-expert count at every eval, both before and after the balancer.

Status: complete via [`phase5/011_load_balancing.py`](../../phase5/011_load_balancing.py), recorded as `P5-011`.

Main lesson:
- **The balancer works and the margin is large.** Ideal share at `64` experts is `0.0156`. With a zero bias the load spans `0.002` to `0.049`, a `24.5x` skew. By step `250` it is `1.38x` and it ends at `1.13x`, with no expert ever unused.
- Against `M-010`, which had no balancer, the comparison is direct: `4.66x` final spread there against `1.13x` here, while being eight times sparser — the regime where imbalance should be worse.
- **Quality is indifferent.** `0.7988` against `0.8002` is `-0.0014`, far inside single-seed resolution. Perfect balance, `63%` more parameters, and eight times the expert pool bought nothing measurable.
- **`bias_span` is the interesting signal, not the loss.** The widest bias gap rises from `0.284` at initialization to `0.972` near step `500`, then falls steadily to `0.484`. It measures how hard the balancer must fight, so it reads directly as how much the router *wants* to be imbalanced — a want that peaks early and then decays as experts specialize and demand spreads out on its own.
- **Cost is `2.1x` wall-clock**, `2008.9s` against `956.5s`, because the Python expert loop now runs `64` narrow iterations per layer. This is the ladder's strongest argument for fused grouped-GEMM dispatch.
- The missing control is `4` of `64` *without* the balancer over a full run. Step `1` shows the unbalanced starting point but not what it would decay to, so the defensible claim is that the balancer holds a badly-skewed regime flat, not a number for the damage avoided.


### Milestone 512: Bounded Feed-Forward Activations
Track: Numerical stability

**Goal.** Cap both branches of the gated feed-forward block so activation outliers cannot form.

**Why this milestone exists.** Two of the three frontier models independently added activation bounding in the same quarter, each having hit the same wall and patched it differently. That is the strongest possible signal that the defect is real and belongs to a mechanism this ladder already built.

**The problem.** Look again at what `M-506` shipped:

$$
\mathrm{SwiGLU}(x) = W_d\big(\mathrm{Swish}(W_g x) \odot W_u x\big)
$$

**Both factors of that product are unbounded.** `Swish(z) → z` for large positive `z`, and `W_u x` is a plain linear map. So the block's output grows *quadratically* in the input scale, and a single token where both branches happen to be large in the same coordinate produces an activation orders of magnitude above the typical value. Those outliers are the "massive activations" that break `bf16` and `fp8` arithmetic, blow up quantization ranges, and — per DeepSeek's own account — are what their loss spikes were consistently tied to.

This compounds with `M-510`. Kimi K3 reports that in their MoE the routed path chains `W↓`, a gated expert FFN, and `W↑` into nearly four consecutive matrix multiplications, and that this ill-conditioned chain "produces exploding internal activations."

**What the field tried.** **Loss-spike rollbacks** — restore an earlier checkpoint and skip the offending batch — are the traditional response and DeepSeek says plainly they were "inadequate as a long-term solution because they do not prevent the recurrence." **Gradient clipping** bounds the update, not the activation, so it does not touch this. **Attention logit soft-capping** (Gemma 2) applied the right idea in the wrong place: it bounds attention scores, not FFN outputs. **QK-Norm** (`M-507`) fixed the attention half of the outlier problem structurally; nothing had fixed the FFN half.

**The solution that won.** Bound the branches. Two shipped forms, converging on the same idea:

**DeepSeek-V4 — hard clamping.** Clamp the linear component of SwiGLU to `[-10, 10]` and cap the gate's upper bound at `10`. They report it "effectively eliminates outliers and substantially aids in stabilizing the training process, without compromising performance," and it was one of only two techniques that fixed their instability.

**Kimi K3 — SiTU-GLU, a smooth cap.** Apply `softcap(x, β) = β \tanh(x/β)` to *both* branches independently:

$$
\mathrm{SiTU\text{-}GLU}(x) = \left[\beta_1 \tanh\!\left(\frac{W_g x}{\beta_1}\right) \odot \sigma(W_g x)\right] \odot \beta_2 \tanh\!\left(\frac{W_u x}{\beta_2}\right)
$$

with `β₁ = 4` on the gate and `β₂ = 25` on the up branch, so the product is bounded by `β₁β₂ = 100`.

The design is careful. `β·tanh(x/β)` is approximately linear near the origin — it is `x - x³/(3β²) + …` — so SiTU-GLU tracks SwiGLU almost exactly in the normal operating range and only bends where SwiGLU would have produced an outlier. Note also that the gate keeps its **own sigmoid** factor: K3 caps the *linear* part of Swish and leaves `σ(W_g x)` intact, so the sign-flipping behaviour that makes SwiGLU beat plain GLU is preserved.

**Implementation decisions.**
- **Implement SiTU-GLU, not hard clamping.** `tanh` is smooth and has non-zero gradient everywhere; a hard clamp has exactly zero gradient outside its range, so a coordinate that saturates stops learning entirely. The smooth form is also the newer of the two.
- `β₁ = 4`, `β₂ = 25`, taken directly from K3 rather than tuned. This ladder does not have the budget to tune them, and inventing values would make the comparison meaningless.
- Apply to **every** gated block: the dense block `0`, the routed experts, and the shared expert.
- Run the hard-clamp variant as the A/B if time permits, since it is a two-line change and the smooth-vs-hard question is the interesting one.
- **Measure whether the failure mode exists here before claiming the fix works.** Instrument the maximum absolute pre-`down_proj` activation across training in `M-506`'s configuration. At `d_model = 256` in `fp32` on a `T4`, outliers may simply not form — in which case the honest result is "no outliers to cap," exactly as `M-510` reported "no collapse to balance." **This is the expected outcome and it is a real finding, not a failed milestone.**

Status: complete via [`phase5/012_bounded_activations.py`](../../phase5/012_bounded_activations.py), recorded as `P5-012`.

Main lesson:
- **Quality is unchanged**, `0.7973` against `0.7988` at identical parameter count. The expected result for a stability mechanism in a run that was never unstable.
- **Cost is `1.24x` wall-clock.** Two `tanh` calls per gated block across `64` experts in seven layers is not free.
- **The smooth cap earns its choice over DeepSeek's hard clamp, measurably.** At `x = 30` a `clamp(-10, 10)` has derivative exactly `0.000` while SiTU-GLU passes `1.220`. A coordinate that saturates a clamp stops learning; one that saturates a `tanh` does not.
- **The failure mode is real here.** An instrumented rerun shows both caps engaging hard: the gate branch peaks at `28.29` against a cap of `4` and the up branch at `47.22` against `25`. Uncapped, the worst coordinate would emit `1336` where a typical one emits `0.731` — the cap removes a `14x` outlier. So the mechanism is not decorative at this scale, even though `fp32` absorbs the outlier without any quality cost. Its value is realized in `bf16`, in quantization, and in loss spikes over long runs, none of which a `3000`-step `fp32` ladder can see.
- **`expert_unused` is a lower bound, not a count.** `expert_load_share` sums across all seven mixture layers before checking for zeros, so per-layer deaths are invisible. A per-layer counter shows `2` dead experts in the worst layer at step `1` while the summed statistic reported `0`. The balancer revived both by step `250`.
- **Unexpected second-order effect:** capping made the load balancer work harder. `bias_span` ends `42%` higher than in `M-011` (`0.687` against `0.484`) and decays far more slowly. Load stays balanced, but the correction needed to hold it there is persistently larger. One seed, so a hypothesis rather than a result.

### Milestone 513: Attention Sinks
Track: Attention stability

**Goal.** Add a learnable per-head sink logit to the softmax denominator so a head can attend to nothing.

**The problem.** Softmax normalizes over the keys, so attention weights are **forced to sum to `1`**. A head that has found nothing relevant in the context has no way to say so — it must return a convex combination of value vectors regardless. What models actually learn to do is dump the mass onto a semantically empty token, usually position `0`, whose value vector then enters the residual stream as a large meaningless activation. This is the attention sink, and it is a direct cause of the massive activations that wreck quantization.

**What the field tried.** **StreamingLLM** diagnosed the sink and worked around it: always retain the first few tokens in the KV cache, because evicting them destroys the model. Correct, and purely a workaround — it preserves the sink rather than removing the need for one. **Registers / prepended learnable tokens** (from the ViT literature) give the model a dedicated dumping ground, which is cleaner but costs real sequence positions and cache. **Gated attention** (`M-507`) attacks it from the output side: if the gate closes, whatever softmax was forced to emit is zeroed. That works, and Kimi K3 relies on it exclusively.

**The solution that won.** Change the denominator. Add a learnable per-head scalar `z'_h` to the normalization *without* a corresponding value:

$$
s_{h,i,j} = \frac{\exp(z_{h,i,j})}{\sum_k \exp(z_{h,i,k}) + \exp(z'_h)}
$$

Now the attention weights sum to something **less than `1`**, and the head can drive the total near zero by learning a large `z'_h`. There is no phantom token, no wasted position, and no value vector attached to the sink — the missing mass simply goes nowhere. DeepSeek-V4 ships exactly this in both CSA and HCA, describing it as allowing "each query head to adjust its total attention scores to be not equal to 1, and even to be near 0."

**Implementation decisions.**
- One learnable scalar per head, shape `[H]`, initialized to `0` (equivalent to one extra key with logit `0`).
- Implement by concatenating the sink logit as an extra column before `softmax`, then dropping that column from the weights — this is numerically identical to modifying the denominator and reuses the stable `softmax`, rather than hand-rolling an `exp`/normalize that can overflow.
- Apply to **all** attention layers, local and global.
- **Expect overlap with `M-507`'s output gate, and say so.** These two mechanisms solve the same problem from opposite ends, and the frontier is split: K3 uses the gate alone, DeepSeek-V4 uses the sink alone. Having both is defensible but redundant, so this milestone's real question is whether the sink adds anything *given* a gate is already present. A null result would be informative and should be reported as such.
- Instrument the learned sink logits at the end of training. If they stay near `0`, no head wanted the escape hatch at this scale.

### Milestone 514: Multi-Token Prediction
Track: Objective

**Goal.** Add an auxiliary module that predicts the token after next, and blend its loss into the objective.

**The problem.** Next-token cross-entropy is a *local* objective. The representation at position `t` only ever needs to be good enough to pick token `t+1`, so there is no pressure to encode anything about `t+2` onward. This rewards a model that latches onto short-range surface statistics, and it means the training signal per token is thin — one forward pass produces exactly one supervised prediction per position.

**What the field tried.** **Meta's multi-token prediction** (Gloeckle et al., 2024) put `n` independent output heads on a shared trunk, all reading the representation at position `t`: head `1` predicts `t+1`, head `2` predicts `t+2`, and so on, **in parallel**. It works and improved code generation, but the heads are independent, so nothing enforces that the `t+2` prediction is consistent with the `t+1` prediction — the model is not asked to reason about the sequence, only to make `n` simultaneous guesses.

**The solution that won.** DeepSeek's **sequential** MTP. Instead of parallel heads, chain small modules that each predict one step further, preserving the causal relationship:

1. Take the main model's hidden state `h_t`.
2. Concatenate it with the **embedding of the actual next token** `t+1`.
3. RMSNorm, then a linear projection back to `d_model`.
4. Run one transformer block.
5. Predict token `t+2` through the **shared** output head.

The key difference from Meta's version is step `2`: the module is conditioned on the true `t+1`, so it is genuinely predicting "given what actually comes next, what comes after," preserving the causal chain rather than guessing two independent things.

Two properties made this the standard. The embedding table and output head are **shared** with the main model, so the added parameters are one projection and one block per depth. And the module is **discardable** — drop it and ordinary inference is unchanged, or keep it and use it as a draft model for speculative decoding. Kimi K3 does exactly that, fine-tuning its pretrained MTP layer into an EAGLE-3-style drafter, noting the structures match.

All three frontier models ship this, and **all three use depth `1`**. Deeper MTP was tried and the returns did not justify it.

**Implementation decisions.**
- Depth `1` — predict `t+2` only. Unanimous at the frontier.
- Share the embedding table and the output projection with the main model. Only the concat-projection and one decoder block are new.
- Loss is `L_main + λ · L_mtp`. Start at `λ = 0.3` (DeepSeek's value) and check at least one other value, since the whole milestone is about whether the auxiliary signal helps or distracts.
- **Report the main-task loss separately.** The blended number is not comparable to any previous milestone; only `L_main` is. Getting this wrong would silently break the entire ladder's comparability.
- Targets shift by two, so the batch sampler needs `tokens[positions + 2]`. The last two positions have no `t+2` target and must be masked out of the auxiliary loss.
- Do not use the MTP head at eval. Evaluation runs the main path only.

### Milestone 515: Gated Linear Attention With A Delta Rule
Track: Attention

**Goal.** Replace the sliding-window local layers with a gated linear-attention recurrence using a delta-rule state update, keeping the `3:1` pattern against the global MLA layers.

**The problem.** Sliding-window attention (`M-508`) bounds cost per token, but it does so by **throwing information away**: a token more than `W` positions back is simply invisible to that layer. Reach is recovered only indirectly, through depth. And the KV cache still grows with the window, so state is `O(W)` per layer rather than genuinely constant.

**What the field tried.** **Linear attention** replaces `softmax(qk)v` with a kernel feature map so associativity can be exploited: `(φ(q)φ(k)ᵀ)v` becomes `φ(q)(φ(k)ᵀv)`, turning attention into a recurrent state `S_t = S_{t-1} + k_t v_tᵀ` of fixed size. Constant memory, linear time. But the naive form **only ever adds** to the state — it can never remove or overwrite anything, so `S` saturates and old information is never displaced. Quality was well below softmax attention.

**Gated variants** (GLA, Mamba-2, RetNet) added a decay term, `S_t = γ_t S_{t-1} + k_t v_tᵀ`, so old information fades. Better, but decay is indiscriminate forgetting — everything fades at the same rate regardless of relevance.

**The delta rule** (DeltaNet) changed *how* the write happens. Instead of blindly adding, first read what is currently stored at `k_t`, and write the **difference** between what is stored and what should be:

$$
S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t k_t v_t^\top
$$

That is one step of online gradient descent on `‖S k_t − v_t‖²`. The `(I − βkkᵀ)` term erases the existing association at `k_t` before writing the new one, so the state *overwrites* rather than accumulates. This is what fixed the associative-recall failures.

**The solution that won: Kimi Delta Attention.** KDA composes both — a **channel-wise** forget gate applied before a delta-rule update:

$$
S_t = \big(I - \beta_t k_t k_t^\top\big)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top, \qquad \tilde{o}_t = S_t^\top q_t
$$

`α_t ∈ (0,1)^{d_k}` is a **per-channel** retention factor, so different feature channels can forget at different rates — a strict generalization of the scalar decay in GLA. `β_t ∈ (0,1)` controls write strength.

K3's specific contribution is the **lower-bounded decay**, and it is a systems fix with an architectural shape. The chunked parallel form rescales keys by `1/Γ`, the reciprocal cumulative decay. Since `Γ` is a product of numbers in `(0,1)`, that reciprocal can grow without bound and overflow. Kimi Linear handled it with log-space arithmetic and special-cased diagonal tiles. K3 instead bounds the log-decay by construction:

$$
g_t = g_{\min}\,\sigma(e^{A} z_t) \in (g_{\min}, 0), \qquad \alpha_t = \exp(g_t), \qquad g_{\min} = -5
$$

With `g_min = -5` every retention factor exceeds `e⁻⁵ ≈ 6.7×10⁻³`, the cumulative log-decay over a `16`-token tile stays in `(-80, 0)`, and the rescaling factor stays inside `bf16` range. **Because the range is now finite, every tile can use dense tensor-core matmuls** and the special-cased diagonal path disappears entirely. A numerical bound bought a kernel simplification.

**Implementation decisions.**
- Implement **both the recurrent form and the chunked parallel form, and assert they agree numerically.** This is the milestone's real exit criterion. The recurrent form is the definition; the chunked form is what runs. If they disagree, the chunked derivation is wrong, and that bug is invisible in a loss curve.
- Use the **bounded** decay parameterization with `g_min = -5`. The unbounded negative-softplus form is the older Kimi Linear/GDN mapping and is strictly harder to make numerically safe.
- Keep the full KDA input pipeline: `q, k = L2Norm(Swish(ShortConv(Wx)))`, `v = Swish(ShortConv(W_v x))`, `β = σ(W_β x)`, and a **low-rank** projection for the decay logits `z`. The `L2Norm` on `q`/`k` is KDA's analogue of QK-Norm.
- Output path is `y = W_o[σ(W_g x) ⊙ RMSNorm(õ_t)]` — head-wise RMSNorm on the recurrent output, then the same full-rank output gate from `M-507`. The norm matters: the recurrent state has no softmax to bound its scale.
- **Remove RoPE from these layers.** Position now comes from the recurrence's decay ordering. This is what lets K3 run NoPE across the entire model.
- Expect this to be **slow** — a Python-level chunked scan against fused kernels. Report it honestly; this is the single best phase-4 Triton target the ladder will produce.

### Milestone 516: Residual-Stream Upgrade
Track: Depth

**Goal.** Replace the plain residual connection with one that lets each layer read from all preceding layers selectively.

**Why this milestone matters.** Two of the three frontier models ship a residual-stream replacement, independently and by different routes. At `8` layers it is a plausible loser, which is exactly why it is worth measuring rather than assuming.

**The problem.** Kimi K3 states it in one sentence: standard residual connections "compress all prior information into a single state `h_l` over depth — a bottleneck reminiscent of RNNs over time." Layer `l` sees exactly one vector, the running sum of everything before it. It cannot ask for layer `3`'s output specifically; that output has already been added into an undifferentiated total, possibly swamped by later contributions.

The parallel is precise and worth sitting with. The transformer's founding insight was that compressing a sequence into one recurrent state is lossy, and attention fixed it by letting each position address all previous positions with data-dependent weights. **The residual stream is that same lossy compression, along depth, and it was never fixed.**

**What the field tried.** **DenseNet-style concatenation** gives every layer access to all previous outputs, but the width grows with depth and the parameter cost is quadratic. **Highway networks and gated residuals** add a learned scalar or vector gate on the residual branch, which controls *how much* to add but not *what to read*. **LayerScale** shrinks each branch by a learned per-channel factor at initialization, which helps trainability without changing what information is reachable.

**The solution that won — two of them, actually.**

**Kimi K3's Attention Residuals.** Apply attention to depth. Give each layer `l` a learnable pseudo-query `q_l = w_l`, treat every preceding layer's output as a key and value (with the token embedding as index `0`), and attend:

$$
\phi(q,k) = \exp\!\big(q^\top \mathrm{RMSNorm}(k)\big), \qquad \alpha_{i \to l} = \frac{\phi(q_l, k_i)}{\sum_{j=0}^{l-1}\phi(q_l, k_j)}, \qquad h_l = \sum_{i=0}^{l-1} \alpha_{i\to l}\, v_i
$$

The `RMSNorm` inside the kernel is load-bearing: it prevents layers with large-magnitude outputs from dominating the weights purely by scale. Note the query is a plain learnable vector, not a function of the token — this is per-layer, not per-token, addressing.

The full form costs `O(L²d)` arithmetic, which K3 calls affordable below `100` layers, and `O(Ld)` memory. To cut that they use **Block AttnRes**: partition `L` layers into `N` blocks, sum within a block, and attend only over the `N` block-level representations. `N ≈ 8` recovers most of the benefit; K3 uses `8` blocks of `12` layers.

**DeepSeek-V4's mHC.** Widen the residual stream from `ℝ^d` to `ℝ^{n_hc × d}` and mix the channels with a learned matrix:

$$
X_{l+1} = B_l X_l + C_l\,\mathcal{F}_l(A_l X_l)
$$

Plain hyper-connections do this and are numerically unstable when stacked. mHC's contribution is constraining `B_l` to the **Birkhoff polytope** — doubly stochastic, every row and column summing to `1`, all entries non-negative — via `20` Sinkhorn-Knopp iterations. A doubly stochastic matrix has spectral norm bounded by `1`, so the residual transform is non-expansive and cannot amplify signal across depth. The set is also closed under multiplication, so the guarantee survives arbitrarily deep stacks. `A_l` and `C_l` are sigmoid-bounded to prevent signal cancellation. DeepSeek uses `n_hc = 4`.

**Implementation decisions.**
- **Implement Attention Residuals, not mHC.** Three reasons. AttnRes is one mechanism with a clean statement — attention over depth — and it teaches something transferable; mHC needs dynamic parameter generation, Sinkhorn projection, and a `4x`-wide residual state, which is a lot of machinery for one measurement. AttnRes's justification is architectural (the depth bottleneck) while mHC's is primarily numerical (stability at `1.6T` parameters), and the numerical problem does not exist at `6M`. And AttnRes changes the residual *routing*, which is the interesting question; mHC changes its *width*.
- Use the **full** form, not Block AttnRes. At `L = 8` the block partition would be near-vacuous, and `O(L²d)` at `L = 8` is `64` inner products per token — free.
- Include the **token embedding as index `0`**, per K3. Every layer keeps direct access to the raw input.
- Keep the `RMSNorm` inside the attention kernel. Removing it is the obvious "simplification" and it is the thing that makes the weights scale-invariant.
- Parameter cost is one `d`-vector per layer — `8 × 128 = 1024` parameters. This is nearly free, which makes it an unusually clean comparison.
- Memory cost is keeping all `L` layer outputs alive, so activation memory rises. Measure it.

### Milestone 517: Muon Optimizer And Untied Embeddings
Track: Optimization

**Goal.** Move from AdamW everywhere to Muon on 2D matrices with AdamW on embeddings, norms, and the head, and untie the input and output embeddings.

**Why it comes last among the mechanisms.** An optimizer change shifts every earlier comparison simultaneously. Doing it before the architecture is frozen would invalidate the whole ladder.

**The problem.** Adam normalizes each parameter *independently* by its own gradient history. For a weight **matrix**, that is a strange thing to do — it ignores the matrix structure entirely. In practice gradient matrices for linear layers are dominated by a few large singular directions, so an Adam update moves the weights mostly along one or two directions while the remaining directions are barely touched. The update is ill-conditioned, and its effect on the function the layer computes is uneven.

**What the field tried.** **Second-order and quasi-Newton methods** (K-FAC, Shampoo) explicitly model curvature and condition the update, and they work — but they need to maintain and invert preconditioner matrices, which is expensive in both memory and time. **LAMB and LARS** used layer-wise update-norm rescaling, which fixes the scale across layers but not the conditioning within a matrix.

**The solution that won.** **Orthogonalize the update.** Take the momentum matrix `M`, compute its SVD `M = UΣVᵀ`, and use `UVᵀ` — all singular values set to `1`. Every direction gets an equal-magnitude update, so no single direction dominates. This is Shampoo's conditioning benefit without maintaining a preconditioner.

An actual SVD every step would be far too slow, so Muon approximates `UVᵀ` with **Newton-Schulz iterations**: normalize `M₀ = M/‖M‖_F`, then repeat

$$
M_k = a M_{k-1} + b\,(M_{k-1}M_{k-1}^\top)M_{k-1} + c\,(M_{k-1}M_{k-1}^\top)^2 M_{k-1}
$$

This is matrix multiplication only — GPU-friendly, no decomposition. DeepSeek-V4 uses a **hybrid schedule** of `10` iterations: `8` steps at `(a,b,c) = (3.4445, -4.7750, 2.0315)` to drive the singular values rapidly toward `1`, then `2` steps at `(2, -1.5, 0.5)` to settle them precisely there. The aggressive coefficients converge fast but overshoot; the conservative ones stabilize.

Muon applies **only to 2D matrices**. Embeddings, norm gains, and the output head stay on AdamW. Both frontier reports that document their optimizer agree exactly on this split, and DeepSeek additionally keeps mHC's static biases and gating factors on AdamW. The reason is that orthogonalization is a statement about a linear map's singular values; a `1D` gain vector and a lookup table are not linear maps in that sense, and an embedding row's update should stay sparse rather than being spread across directions.

Kimi K3 refines further with **Per-Head Muon**: for attention projections, partition the momentum matrix along the head axis and orthogonalize each head's block separately. Full-matrix orthogonalization treats all heads as one coupled block, so heads with larger gradient scale dominate the shared update direction. Per-head equalizes update scale across heads, and is cheaper besides — Newton-Schulz on tall thin blocks beats one big matrix.

**Untying** belongs in this milestone rather than its own, for a mechanistic reason: a tied embedding matrix must serve as both a lookup table and an output linear map, so it would have to be on AdamW and Muon simultaneously. The optimizer split forces the question.

**Implementation decisions.**
- **Write Newton-Schulz explicitly**, not imported. It is five lines and it is the entire mechanism.
- Use the hybrid coefficient schedule: `8 + 2` iterations as above.
- Rescale the update RMS so AdamW's learning rate stays reusable — DeepSeek rescales to `0.18`; the alternative published form is `√max(n,m) · γ`. Pick one and record which.
- Parameter groups: **Muon** for all `2D` weights in attention, feed-forward, experts, and the router. **AdamW** for the embedding table, the output head, every RMSNorm gain, the attention sink logits, and the MoE bias buffer's neighbours. Justify each group in the log.
- Use Nesterov momentum and apply weight decay to Muon parameters, per both reports.
- **Skip Per-Head Muon initially.** With `4` heads its benefit is small; implement it as the A/B if the main comparison is clean.
- Untie the embeddings and expect this to **hurt or do nothing**. At a `98`-character vocabulary the output head is `98 × 128 = 12544` parameters and the tying constraint is nearly harmless; untying matters at a `160K` vocabulary where the two roles genuinely conflict. Predict the null result in advance and check it.
- Do **not** adopt QK-Clip. `M-507` already installed QK-Norm, and DeepSeek-V4 states explicitly that RMSNorm on queries and KV entries is why they dropped QK-Clip from their Muon implementation.

### Milestone 518: Modern Reference Model
Track: Integration

**Goal.** Produce one integrated model containing every mechanism that earned its place, plus the ablation table for the whole phase.

**The problem.** Seventeen incremental scripts is a ladder, not an architecture. Each milestone was measured against its immediate predecessor, which means the errors compound and no single run demonstrates the whole thing. There is also no artifact for phase 4 to profile.

**What the field does.** Every technical report read for this roadmap ends the same way: one frozen configuration table, one ablation study, and an honest section on what did not work. DeepSeek-V4's "Mitigating Training Instability" section — publishing two fixes while admitting "a comprehensive theoretical understanding of their underlying mechanisms remains an open question" — is the model to imitate.

**The solution.** One file, one config, one table.

**Deliverables.**
- One frozen model definition, self-contained.
- One frozen training recipe.
- One table of every milestone: loss, total parameters, active parameters per token, wall-clock.
- One explicit list of mechanisms that were implemented and then dropped, with reasons.
- One list of mechanisms whose failure mode **never appeared at this scale** — routing collapse, activation outliers, attention sinks, logit explosion — separated from those that were measured and rejected. These are different categories and conflating them would be dishonest.

**Implementation decisions.**
- Rerun the final model at **three seeds**. The entire ladder is single-seed and the noise floor is roughly `0.02`; the final number at least should be defensible.
- Keep every mechanism that is a genuine 2026 standard even where it did not pay for itself here, and say so in the table. The phase-4 profiling target must be representative, not locally optimal.
- Also settle the two debts outstanding from the start of the phase: the `post-norm @ lr 3e-4` control for `M-501`/`M-502` attribution, and a parameter-matched control for `M-507`.

**Questions to answer.**
- Which mechanisms were quality changes and which were purely systems changes?
- Which ones only pay off at scales this repo has not reached?
- What is the shortest defensible description of a 2026 architecture?

## Recommended Order

The intended order is exactly `501` through `518`.

The order is not arbitrary:

- **normalization first**, because it makes everything else trainable,
- **positions next**, because they are independent of the attention layout,
- **attention layout before attention internals**, because the layout decides what the internals operate on,
- **sparsity after the dense feed-forward is understood**, because MoE is a modification of a block that must already work,
- **stability after the mechanisms that create the instability** — `M-512` caps SwiGLU, which only exists from `M-506`, and only becomes urgent once `M-511` chains it inside a routed path; `M-513` fixes a softmax pathology that `M-507`'s gate has already half-addressed,
- **the objective after the model**, because MTP adds a second loss and the main loss must stay comparable,
- **the local mixer late**, because gated linear attention is the least settled and hardest mechanism, and it needs the `M-508` hybrid layout to drop into,
- **depth after width**, because AttnRes changes what every layer reads and would otherwise confound every earlier measurement,
- **the optimizer last**, because it changes the meaning of every earlier comparison,
- **integration only once each mechanism has a recorded, individual result.**

The two new milestones, `512` and `513`, are deliberately cheap. Both are small code changes with a strong chance of a null result at this scale, and both are worth running precisely *because* a null result is informative — it distinguishes "this mechanism does nothing" from "this mechanism fixes a problem that does not exist here."

## Non-Goals

Phase 5 should not become:

- a tokenizer or data-pipeline project,
- a scaling run,
- a distributed-training project,
- a kernel-optimization project, which is phase 4's job with phase 5's output,
- a race to reimplement every named mechanism from every technical report,
- a model that is modern but that the author cannot explain.

Specific mechanisms deliberately **not** implemented, with reasons:

| Mechanism | Model | Why it is skipped |
| --- | --- | --- |
| LatentMoE | Kimi K3 | Routed experts operate in a `0.5×d` latent so expert traffic does not scale with routing multiplicity. The benefit is *communication* under expert parallelism, which does not exist on one GPU. |
| Quantile Balancing | Kimi K3 | Solves fixed-step bias adaptation at `896` experts with distributed histogram estimation. At `32` experts on one GPU the simple sign rule is sufficient. |
| mHC | DeepSeek-V4 | A numerical-stability fix for `1.6T` parameters. `M-516` implements AttnRes instead, which addresses the architectural question rather than the numerical one. |
| CSA / HCA | DeepSeek-V4 | Compression along the *sequence* axis, which only matters when sequence length dominates memory. At context `256` it cannot. |
| DeepSeek Sparse Attention | GLM-5.2 | A learned indexer selecting top-`2048` of a `1M`-token prefix. At context `256`, top-`2048` is the whole sequence. |
| Sequence-wise balance loss | DeepSeek-V4 | Reintroduces the auxiliary-loss coupling that `M-511` exists to remove. |
| Hash routing in early blocks | DeepSeek-V4 | Contradicted by the other two models; not a converged choice. |
| Native multimodality | Kimi K3 | Out of scope; this phase is character-level text. |

Each of these should still be **explained** in the learning log where it is relevant. Knowing why a mechanism does not apply at this scale is the same kind of understanding as knowing why one does.

## Success Condition

Phase 5 succeeds if, by the end, the following are all true:

- I can build a current-generation language model architecture from scratch, without copying a reference implementation.
- I can explain every mechanism in it from mechanism, including what it costs.
- I know which of those mechanisms help at small scale, which are purely about inference economics, and which I only believe because a technical report said so.
- I can name, for each mechanism, the failure mode it fixes and whether that failure mode actually occurs at `6M` parameters.
- I have one frozen modern workload that the phase-4 profiling, Triton, and CUDA path can attack next.

## Sources

Primary technical reports, read directly:

- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653) — architecture §2.1–2.5, config table §3.2
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) — architecture §2, model setups §4.2.1, instability §4.2.3

Mechanism papers:

- [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) — NeurIPS 2025 Oral, the gated-attention ablation behind `M-507`
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — auxiliary-loss-free balancing and sequential MTP

Secondary coverage used for GLM-5.2/5.3, which has no public technical report:

- [The Architecture Of A Frontier Model: GLM-5.2](https://paulsbrookes.github.io/2026/06/30/glm-5-2-attention.html) — layer-by-layer teardown with dimensions
- [GLM 5.2 Architecture Deep Dive: Index Share, Sparse Attention, and Multi-Token Prediction](https://www.mindstudio.ai/blog/glm-5-2-architecture-index-share-sparse-attention)
- [GLM-5 repository](https://github.com/zai-org/GLM-5)

Counter-signal:

- [The MiniMax-M2 Series](https://arxiv.org/abs/2605.26494) — the negative result on sliding-window and hybrid attention
- [MiniMax Sparse Attention](https://huggingface.co/papers/2606.13392) — what they shipped in M3 instead
