# Phase 5 Learning Log

This log contains the completed runs from the phase-5 architecture-modernization path.
The roadmap and the frozen control are in [roadmap.md](./roadmap.md).

Every run below is one seed (`1337`) on a Kaggle `Tesla T4`, `3000` steps, batch `32`, context `256`.

## Summary

| Run | Script | Val loss | Delta | Parameters | Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| P5-001 | [`phase5/001_vanilla_decoder.py`](../../phase5/001_vanilla_decoder.py) | `3.0556` | — | `6408704` | `661.1` |
| P5-002 | [`phase5/002_pre_norm.py`](../../phase5/002_pre_norm.py) | `2.2576` | `-0.7980` | `6409216` | `678.6` |
| P5-003 | [`phase5/003_rms_norm.py`](../../phase5/003_rms_norm.py) | `2.1625` | `-0.0951` | `6386432` | `650.1` |
| P5-004 | [`phase5/004_rope.py`](../../phase5/004_rope.py) | `0.9334` | `-1.2291` | `6320896` | `714.0` |
| P5-005 | [`phase5/005_gqa.py`](../../phase5/005_gqa.py) | `0.9052` | `-0.0282` | `5796608` | `659.7` |
| P5-006 | [`phase5/006_swiglu.py`](../../phase5/006_swiglu.py) | `0.9763` | `+0.0711` | `5792512` | `672.0` |
| P5-007 | [`phase5/007_qk_norm_gated_attention.py`](../../phase5/007_qk_norm_gated_attention.py) | `0.7904` | `-0.1859` | `6317312` | `770.7` |
| P5-008 | [`phase5/008_hybrid_attention.py`](../../phase5/008_hybrid_attention.py) | `0.7887` | `-0.0017` | `6317312` | `733.8` |
| P5-009 | [`phase5/009_mla.py`](../../phase5/009_mla.py) | `0.8135` | `+0.0248` | `6358336` | `737.3` |
| P5-010 | [`phase5/010_moe.py`](../../phase5/010_moe.py) | `0.8002` | `-0.0133` | `8899392` | `956.5` |
| P5-011 | [`phase5/011_load_balancing.py`](../../phase5/011_load_balancing.py) | `0.7988` | `-0.0014` | `14504768` | `2008.9` |
| P5-012 | [`phase5/012_bounded_activations.py`](../../phase5/012_bounded_activations.py) | `0.7973` | `-0.0015` | `14504768` | `2493.3` |
| P5-013 | [`phase5/013_attention_sinks.py`](../../phase5/013_attention_sinks.py) | `0.7984` | `+0.0011` | `14504832` | `2758.2` |

## Reading The Ladder

Two mechanisms account for almost everything. Pre-norm is worth `-0.7980` and RoPE is worth
`-1.2291`; together they are `-2.03` of the total `-2.26` from baseline to the final model.
Everything after them moves the loss by less than `0.19`, and most of it by less than `0.03`.

The ordering hid something. Milestones `002` and `003` both end in a stalled regime around `2.2`,
and it is tempting to read pre-norm as a partial fix that RMSNorm nudged along. It is not. Both
runs were blocked on the learned absolute position table, and `M-004` unblocked them at once. The
`-0.0951` credited to RMSNorm is drift inside a stall, not a normalizer effect.

Three results are worth arguing with:

- **SwiGLU made it worse**, `+0.0711` at matched parameters. Not noise, and not explained away.
- **QK-Norm with gated attention is the second-largest gain**, `-0.1859`, but it adds `9%`
  parameters, so mechanism and capacity are confounded in that one number.
- **Sparsity bought nothing measurable**, `-0.0133` for `40%` more total parameters, and cost
  `30%` wall-clock.

Two caveats that apply to every row:

- **One seed.** The `T4` is bit-deterministic at a fixed seed, so these numbers reproduce exactly,
  but that is repeatability, not precision. Nothing under roughly `0.03` should be called.
- **`3000` steps is not convergence.** Every run except the collapsed baseline was still improving
  at the final checkpoint, by between `0.019` and `0.138` over the last `500` steps. The ladder
  compares architectures at a fixed budget, not at their best.

## P5-001 Milestone 501 Vanilla Decoder Baseline

- Script: [`phase5/001_vanilla_decoder.py`](../../phase5/001_vanilla_decoder.py)
- Parameters: `6408704`
- Final train loss: `3.0669`
- Final validation loss: `3.0556`
- Wall-clock time: `661.1s` on a Kaggle `T4`

What it contains:

- nothing yet; this is the starting architecture,
- learned absolute position embeddings added to the token embedding,
- post-norm LayerNorm around both sublayers, biases on every projection,
- a GELU feed-forward block at `4x` width.

Logged checkpoints:

```text
step=1 train_loss=3.4361 val_loss=3.4255 seconds=5.5
step=250 train_loss=3.0725 val_loss=3.0554 seconds=60.5
step=500 train_loss=3.0679 val_loss=3.0570 seconds=114.8
step=750 train_loss=3.0707 val_loss=3.0536 seconds=169.5
step=1000 train_loss=3.0656 val_loss=3.0516 seconds=224.3
step=1250 train_loss=3.0705 val_loss=3.0567 seconds=278.9
step=1500 train_loss=3.0676 val_loss=3.0558 seconds=333.6
step=1750 train_loss=3.0701 val_loss=3.0537 seconds=388.1
step=2000 train_loss=3.0691 val_loss=3.0569 seconds=442.9
step=2250 train_loss=3.0684 val_loss=3.0527 seconds=497.5
step=2500 train_loss=3.0686 val_loss=3.0524 seconds=552.2
step=2750 train_loss=3.0686 val_loss=3.0526 seconds=606.6
step=3000 train_loss=3.0669 val_loss=3.0556 seconds=661.1
```

Main lesson:

- At the control learning rate the baseline never learns. It reaches character-unigram loss (`3.055`) by step `250` and stays flat for the remaining `2750` steps, moving by `0.003` in either direction.
- This is the intended result rather than a bug. Post-norm puts the normalization on the residual stream itself, so there is no identity path from the loss back to the embedding, and no single learning rate serves the whole stack.
- It is worth seeing a real training collapse once, and worth knowing that it looks like a flat line rather than a divergence.

## P5-002 Milestone 502 Pre-Norm Residual Stream

- Script: [`phase5/002_pre_norm.py`](../../phase5/002_pre_norm.py)
- Parameters: `6409216`
- Final train loss: `2.2557`
- Final validation loss: `2.2576`
- Wall-clock time: `678.6s` on a Kaggle `T4`

What changed from `M-001`:

- the normalization moved inside the residual branch, `x + attn(norm(x))`,
- one final norm added before the output projection.

Logged checkpoints:

```text
step=1 train_loss=3.9137 val_loss=3.9092 seconds=5.8
step=250 train_loss=2.3395 val_loss=2.3414 seconds=57.4
step=500 train_loss=2.3284 val_loss=2.3327 seconds=112.6
step=750 train_loss=2.3574 val_loss=2.3562 seconds=170.3
step=1000 train_loss=2.3316 val_loss=2.3347 seconds=227.3
step=1250 train_loss=2.2861 val_loss=2.2920 seconds=284.2
step=1500 train_loss=2.3586 val_loss=2.3532 seconds=341.0
step=1750 train_loss=2.4075 val_loss=2.4009 seconds=397.4
step=2000 train_loss=2.2866 val_loss=2.2857 seconds=453.7
step=2250 train_loss=2.2617 val_loss=2.2541 seconds=509.9
step=2500 train_loss=2.3112 val_loss=2.3077 seconds=566.1
step=2750 train_loss=2.3541 val_loss=2.3402 seconds=622.4
step=3000 train_loss=2.2557 val_loss=2.2576 seconds=678.6
```

Main lesson:

- Moving the norm inside the branch is worth `-0.7980` for four lines of code and `512` parameters. That is the single largest structural gain available.
- But it does not finish the job. The curve reaches `2.34` by step `250` and then crawls, ending at `2.2576` after `2750` more steps. The model is learning, barely.
- So pre-norm removes the hard blocker and exposes a second one. The next two milestones show that the remaining blocker is the position representation, not the normalization.

## P5-003 Milestone 503 RMSNorm And Bias Removal

- Script: [`phase5/003_rms_norm.py`](../../phase5/003_rms_norm.py)
- Parameters: `6386432`
- Final train loss: `2.1587`
- Final validation loss: `2.1625`
- Wall-clock time: `650.1s` on a Kaggle `T4`

What changed from `M-002`:

- LayerNorm became RMSNorm, dropping the mean subtraction and the shift,
- `bias=False` on every linear layer.

Logged checkpoints:

```text
step=1 train_loss=3.8677 val_loss=3.8625 seconds=5.3
step=250 train_loss=2.3571 val_loss=2.3562 seconds=53.1
step=500 train_loss=2.3332 val_loss=2.3366 seconds=104.9
step=750 train_loss=2.3539 val_loss=2.3522 seconds=161.3
step=1000 train_loss=2.3234 val_loss=2.3271 seconds=215.6
step=1250 train_loss=2.3879 val_loss=2.3948 seconds=270.5
step=1500 train_loss=2.3955 val_loss=2.3894 seconds=325.2
step=1750 train_loss=2.3552 val_loss=2.3503 seconds=379.4
step=2000 train_loss=2.2999 val_loss=2.2988 seconds=434.1
step=2250 train_loss=2.2997 val_loss=2.2942 seconds=488.4
step=2500 train_loss=2.3053 val_loss=2.3005 seconds=542.3
step=2750 train_loss=2.2284 val_loss=2.2146 seconds=596.1
step=3000 train_loss=2.1587 val_loss=2.1625 seconds=650.1
```

Main lesson:

- Dropping mean-centering and every bias costs nothing and saves `22784` parameters, which is the claim RMSNorm was introduced to make.
- The `-0.0951` improvement should not be read as a win for RMSNorm. The run is still in the stalled regime that `M-002` left it in, where the loss drifts slowly, and a drift of that size over `3000` steps is not attributable to the normalizer.
- Implementation trap worth keeping: normalizing by `x.var(correction=0)` instead of the mean square agrees with RMSNorm only when the per-token mean is zero, and pre-norm is exactly where the residual stream drifts off zero.

## P5-004 Milestone 504 Rotary Position Embeddings

- Script: [`phase5/004_rope.py`](../../phase5/004_rope.py)
- Parameters: `6320896`
- Final train loss: `0.9204`
- Final validation loss: `0.9334`
- Wall-clock time: `714.0s` on a Kaggle `T4`

What changed from `M-003`:

- the learned absolute position table is gone, and with it `65536` parameters,
- queries and keys are rotated by a position-dependent angle after the head split,
- values are not rotated, because position belongs in the comparison, not the retrieved content.

Logged checkpoints:

```text
step=1 train_loss=3.8378 val_loss=3.8335 seconds=5.6
step=250 train_loss=1.7891 val_loss=1.7935 seconds=56.7
step=500 train_loss=1.4177 val_loss=1.4125 seconds=114.0
step=750 train_loss=1.2671 val_loss=1.2640 seconds=174.9
step=1000 train_loss=1.1506 val_loss=1.1512 seconds=235.3
step=1250 train_loss=1.0876 val_loss=1.0978 seconds=295.4
step=1500 train_loss=1.0672 val_loss=1.0559 seconds=355.3
step=1750 train_loss=1.0254 val_loss=1.0185 seconds=415.1
step=2000 train_loss=0.9961 val_loss=1.0030 seconds=474.9
step=2250 train_loss=0.9827 val_loss=0.9708 seconds=534.6
step=2500 train_loss=0.9661 val_loss=0.9675 seconds=594.3
step=2750 train_loss=0.9432 val_loss=0.9433 seconds=654.0
step=3000 train_loss=0.9204 val_loss=0.9334 seconds=714.0
```

Main lesson:

- This is the milestone that makes the model work. Validation loss falls from `2.1625` to `0.9334`, a gain of `-1.2291`, while the model *loses* `65536` parameters.
- Nothing else in the ladder comes close, and the reason is that the previous three milestones were all stalled on the same thing. A learned absolute position table gives every index its own free vector and forces the model to discover that only differences matter. RoPE makes the score between positions `m` and `n` depend on `n - m` by construction, so that discovery is not needed.
- The rotation is also length-preserving, so position is injected without disturbing activation scale. That is the mechanical reason it can be applied at every layer while an additive table cannot.
- The honest reading of milestones `002` through `004` together: pre-norm was necessary but not sufficient, and the learned position table was the binding constraint all along.

## P5-005 Milestone 505 Grouped-Query Attention

- Script: [`phase5/005_gqa.py`](../../phase5/005_gqa.py)
- Parameters: `5796608`
- Final train loss: `0.8933`
- Final validation loss: `0.9052`
- Wall-clock time: `659.7s` on a Kaggle `T4`

What changed from `M-004`:

- key and value heads dropped from `8` to `4`, shared across query-head groups of two,
- expansion uses `repeat_interleave`, which pairs heads group-contiguously.

Logged checkpoints:

```text
step=1 train_loss=4.0761 val_loss=4.0679 seconds=5.2
step=250 train_loss=1.8144 val_loss=1.8163 seconds=54.2
step=500 train_loss=1.3971 val_loss=1.3937 seconds=106.4
step=750 train_loss=1.2251 val_loss=1.2211 seconds=162.8
step=1000 train_loss=1.1392 val_loss=1.1421 seconds=217.5
step=1250 train_loss=1.0831 val_loss=1.0928 seconds=272.5
step=1500 train_loss=1.0415 val_loss=1.0352 seconds=327.7
step=1750 train_loss=1.0030 val_loss=0.9987 seconds=383.2
step=2000 train_loss=0.9710 val_loss=0.9790 seconds=438.6
step=2250 train_loss=0.9541 val_loss=0.9441 seconds=493.8
step=2500 train_loss=0.9390 val_loss=0.9429 seconds=548.8
step=2750 train_loss=0.9161 val_loss=0.9222 seconds=604.2
step=3000 train_loss=0.8933 val_loss=0.9052 seconds=659.7
```

Main lesson:

- Sharing key and value heads improved the loss by `-0.0282` while removing `524288` parameters, roughly `9%` of the model.
- The loss change is small enough that it should be read as "no cost" rather than as a gain; a single seed cannot resolve `0.03` reliably. The parameter saving is the real result.
- The mechanism's actual payoff is invisible here by construction. GQA exists to shrink the KV cache during autoregressive decoding, and training never builds one.

## P5-006 Milestone 506 SwiGLU Feed-Forward

- Script: [`phase5/006_swiglu.py`](../../phase5/006_swiglu.py)
- Parameters: `5792512`
- Final train loss: `0.9674`
- Final validation loss: `0.9763`
- Wall-clock time: `672.0s` on a Kaggle `T4`

What changed from `M-005`:

- the two-matrix GELU MLP became a three-matrix gated block, `down(silu(gate(x)) * up(x))`,
- `D_FFN` narrowed from `1024` to `682`, so the gated block matches the dense one it replaces.

Logged checkpoints:

```text
step=1 train_loss=3.8200 val_loss=3.8126 seconds=5.3
step=250 train_loss=1.6215 val_loss=1.6293 seconds=56.7
step=500 train_loss=1.3732 val_loss=1.3682 seconds=110.9
step=750 train_loss=1.2054 val_loss=1.2022 seconds=168.0
step=1000 train_loss=1.1843 val_loss=1.1872 seconds=224.1
step=1250 train_loss=1.0774 val_loss=1.0882 seconds=280.2
step=1500 train_loss=1.0642 val_loss=1.0559 seconds=336.3
step=1750 train_loss=1.0437 val_loss=1.0388 seconds=392.2
step=2000 train_loss=1.0414 val_loss=1.0515 seconds=448.1
step=2250 train_loss=1.0040 val_loss=0.9915 seconds=504.0
step=2500 train_loss=1.0194 val_loss=1.0190 seconds=560.1
step=2750 train_loss=1.0021 val_loss=1.0051 seconds=616.0
step=3000 train_loss=0.9674 val_loss=0.9763 seconds=672.0
```

Main lesson:

- At matched parameters SwiGLU made the model **worse**, by `+0.0711`. This is the clearest negative result in the ladder and it is large enough to be real rather than noise.
- The comparison is clean, which is what makes it interesting. A gated block has three matrices where the dense block had two, so `D_FFN` narrowed from `1024` to `682` and the parameter count barely moved, from `5796608` to `5792512`.
- So the gate bought a multiplicative interaction and paid for it with `33%` less width, and at this scale the width was worth more. The field's preference for SwiGLU is established at far larger widths, where the trade goes the other way.
- Worth stating plainly: this ladder now contains a mechanism that every frontier model uses and that measurably hurt at `6M` parameters.

## P5-007 Milestone 507 QK-Norm And Gated Attention

- Script: [`phase5/007_qk_norm_gated_attention.py`](../../phase5/007_qk_norm_gated_attention.py)
- Parameters: `6317312`
- Final train loss: `0.7681`
- Final validation loss: `0.7904`
- Wall-clock time: `770.7s` on a Kaggle `T4`

What changed from `M-006`:

- RMSNorm applied per head to queries and keys, before the score,
- a data-dependent elementwise sigmoid gate on the attention output.

Logged checkpoints:

```text
step=1 train_loss=4.1095 val_loss=4.1036 seconds=6.3
step=250 train_loss=1.3681 val_loss=1.3728 seconds=64.5
step=500 train_loss=1.0998 val_loss=1.0978 seconds=124.9
step=750 train_loss=0.9959 val_loss=0.9965 seconds=187.0
step=1000 train_loss=0.9242 val_loss=0.9300 seconds=250.1
step=1250 train_loss=0.8874 val_loss=0.9018 seconds=315.3
step=1500 train_loss=0.8699 val_loss=0.8691 seconds=380.3
step=1750 train_loss=0.8474 val_loss=0.8489 seconds=445.3
step=2000 train_loss=0.8252 val_loss=0.8372 seconds=510.2
step=2250 train_loss=0.8121 val_loss=0.8135 seconds=575.0
step=2500 train_loss=0.7992 val_loss=0.8098 seconds=640.0
step=2750 train_loss=0.7837 val_loss=0.7957 seconds=704.9
step=3000 train_loss=0.7681 val_loss=0.7904 seconds=770.7
```

Main lesson:

- The second-largest gain of the ladder, `-0.1859`, taking validation loss below `0.80` for the first time.
- It is also the least attributable. The elementwise gate is a full `D x D` matrix, so the model grew by `524800` parameters, about `9%`. Some of the improvement is the mechanism and some is the capacity, and this run does not separate them.
- The cheap control, if it is ever wanted, is a QK-Norm-only variant: the two norms cost `64` parameters between them.

## P5-008 Milestone 508 Layerwise Hybrid Attention

- Script: [`phase5/008_hybrid_attention.py`](../../phase5/008_hybrid_attention.py)
- Parameters: `6317312`
- Final train loss: `0.7668`
- Final validation loss: `0.7887`
- Wall-clock time: `733.8s` on a Kaggle `T4`

What changed from `M-007`:

- six of eight layers attend only to the last `64` tokens, masking the future and the distant past,
- layers `3` and `7` attend over the whole sequence and carry no positional encoding,
- no parameters change, since masks are buffers and RoPE has no weights.

Logged checkpoints:

```text
step=1 train_loss=4.1362 val_loss=4.1309 seconds=6.0
step=250 train_loss=1.3447 val_loss=1.3487 seconds=63.0
step=500 train_loss=1.0883 val_loss=1.0867 seconds=123.3
step=750 train_loss=0.9954 val_loss=0.9965 seconds=184.2
step=1000 train_loss=0.9235 val_loss=0.9317 seconds=245.3
step=1250 train_loss=0.8864 val_loss=0.9001 seconds=306.4
step=1500 train_loss=0.8725 val_loss=0.8711 seconds=367.4
step=1750 train_loss=0.8500 val_loss=0.8503 seconds=428.5
step=2000 train_loss=0.8274 val_loss=0.8404 seconds=489.5
step=2250 train_loss=0.8138 val_loss=0.8145 seconds=550.6
step=2500 train_loss=0.8022 val_loss=0.8127 seconds=611.7
step=2750 train_loss=0.7855 val_loss=0.8008 seconds=672.7
step=3000 train_loss=0.7668 val_loss=0.7887 seconds=733.8
```

Main lesson:

- Six of eight layers lost `75%` of their receptive field and the loss did not move: `-0.0017` at **identical** parameter count.
- This is the cleanest comparison in the ladder, since masks are buffers and NoPE removes rather than adds. Nothing but the receptive field changed.
- At context `256` with a `64`-token window that is the expected result. The mechanism's value is asymptotic, and `256` tokens is not long. It is evidence that the layout is free here, not evidence that it is free.

## P5-009 Milestone 509 Multi-Head Latent Attention

- Script: [`phase5/009_mla.py`](../../phase5/009_mla.py)
- Parameters: `6358336`
- Final train loss: `0.7902`
- Final validation loss: `0.8135`
- Wall-clock time: `737.3s` on a Kaggle `T4`

What changed from `M-008`:

- global layers compress keys and values into a `64`-wide latent, up-projected during attention,
- a decoupled rope path of `16` dims per head carries position, shared across heads,
- the norm sits on the latent, so the up-projection stays foldable into the query.

Logged checkpoints:

```text
step=1 train_loss=3.9939 val_loss=3.9845 seconds=5.8
step=250 train_loss=1.3658 val_loss=1.3680 seconds=63.1
step=500 train_loss=1.1078 val_loss=1.1062 seconds=123.1
step=750 train_loss=1.0152 val_loss=1.0188 seconds=184.6
step=1000 train_loss=0.9508 val_loss=0.9598 seconds=246.0
step=1250 train_loss=0.9152 val_loss=0.9313 seconds=307.4
step=1500 train_loss=0.9008 val_loss=0.8999 seconds=368.8
step=1750 train_loss=0.8740 val_loss=0.8735 seconds=430.3
step=2000 train_loss=0.8481 val_loss=0.8597 seconds=491.8
step=2250 train_loss=0.8393 val_loss=0.8375 seconds=553.2
step=2500 train_loss=0.8245 val_loss=0.8343 seconds=614.5
step=2750 train_loss=0.8107 val_loss=0.8215 seconds=675.8
step=3000 train_loss=0.7902 val_loss=0.8135 seconds=737.3
```

Main lesson:

- Latent attention cost `+0.0248` for `41024` more parameters. A small regression, near the resolution of a single seed.
- The cache arithmetic is the actual deliverable. A global layer now caches `64 + 16 = 80` numbers per token against `512` for full multi-head keys and values, a `6.4x` reduction, and every query head gets its own keys and values back rather than sharing.
- DeepSeek's own sizing rule does not transfer. They set the latent to `4 * d_head`, which here is `128`, exactly half the model dim and barely a compression. Their `28x` saving comes from having `128` heads of `128` dims to squeeze into `512`; an eight-head model has far less redundancy to exploit.
- The decoupled rope path was implemented even though these layers are NoPE, because that conflict is the entire reason MLA has its shape. Absorption requires the map from latent to key to be linear and constant, which is why the norm sits on the latent and there is no key norm at all.

## P5-010 Milestone 510 Sparse Mixture-Of-Experts Feed-Forward

- Script: [`phase5/010_moe.py`](../../phase5/010_moe.py)
- Parameters: `8899392`
- Final train loss: `0.7790`
- Final validation loss: `0.8002`
- Wall-clock time: `956.5s` on a Kaggle `T4`

What changed from `M-009`:

- blocks `1` through `7` route each token to `4` of `8` narrow experts plus one shared expert,
- block `0` stays dense,
- per-expert token counts are tracked during training and reported at every evaluation.

Logged checkpoints:

```text
step=1 train_loss=3.7895 val_loss=3.7831 seconds=7.7 expert_min=0.094 expert_max=0.148 expert_unused=0
step=250 train_loss=1.3333 val_loss=1.3410 seconds=82.8 expert_min=0.032 expert_max=0.177 expert_unused=0
step=500 train_loss=1.0930 val_loss=1.0928 seconds=162.2 expert_min=0.037 expert_max=0.176 expert_unused=0
step=750 train_loss=1.0004 val_loss=1.0009 seconds=241.5 expert_min=0.037 expert_max=0.178 expert_unused=0
step=1000 train_loss=0.9347 val_loss=0.9406 seconds=320.9 expert_min=0.041 expert_max=0.176 expert_unused=0
step=1250 train_loss=0.8977 val_loss=0.9134 seconds=400.2 expert_min=0.041 expert_max=0.176 expert_unused=0
step=1500 train_loss=0.8840 val_loss=0.8833 seconds=479.5 expert_min=0.041 expert_max=0.174 expert_unused=0
step=1750 train_loss=0.8624 val_loss=0.8612 seconds=559.0 expert_min=0.041 expert_max=0.173 expert_unused=0
step=2000 train_loss=0.8361 val_loss=0.8491 seconds=638.6 expert_min=0.039 expert_max=0.174 expert_unused=0
step=2250 train_loss=0.8270 val_loss=0.8254 seconds=718.1 expert_min=0.038 expert_max=0.176 expert_unused=0
step=2500 train_loss=0.8103 val_loss=0.8224 seconds=797.6 expert_min=0.038 expert_max=0.177 expert_unused=0
step=2750 train_loss=0.7955 val_loss=0.8082 seconds=877.1 expert_min=0.040 expert_max=0.176 expert_unused=0
step=3000 train_loss=0.7790 val_loss=0.8002 seconds=956.5 expert_min=0.038 expert_max=0.177 expert_unused=0
```

Main lesson:

- Total and active parameters diverge for the first time: total rises `40%` to `8899392` while the active feed-forward width per token is `4 * 128 + 128 = 640` against the dense `682`, so each token costs slightly less than before.
- Validation loss improved `-0.0133`, which is inside what a single seed can resolve. The honest statement is that `1.5x` the feed-forward capacity at slightly lower active cost bought nothing measurable.
- Wall-clock is the real cost: `956.5s` against `737.3s`, `30%` slower. Seven blocks now run eight small matrix multiplications where they ran one large one, and the shapes are too small to use the GPU well. Production speed comes from fused grouped-GEMM kernels.

### Routing Is Measurably Imbalanced

Per-expert load is tracked on **training** dispatch, and the picture is not uniform:

| Step | min share | max share | unused |
| ---: | ---: | ---: | ---: |
| 1 | `0.094` | `0.148` | `0` |
| 250 | `0.032` | `0.177` | `0` |
| 1500 | `0.041` | `0.174` | `0` |
| 3000 | `0.038` | `0.177` | `0` |

Ideal share is `0.125`. At initialization the load is nearly uniform, as it must be. Within `250`
steps the busiest expert is taking `0.177` and the quietest `0.032`, a **`4.6x` spread**, and that
gap then holds steady for the rest of the run rather than widening further.

So the rich-get-richer dynamic is real and visible, but it stabilizes instead of collapsing. No
expert dies. The reason it does not run away is that `4` of `8` is barely sparse: every token uses
half the experts, so even an unpopular one keeps receiving gradient.

This is what milestone `511` acts on. There is a genuine imbalance for a selection-only bias to
correct, and a quietest expert at `30%` of its fair share is the number to beat.

## P5-011 Milestone 511 Real Sparsity And Quantile Balancing

- Script: [`phase5/011_load_balancing.py`](../../phase5/011_load_balancing.py)
- Parameters: `14504768`
- Final train loss: `0.7778`
- Final validation loss: `0.7988`
- Wall-clock time: `2008.9s` on a Kaggle `T4`
- Experts: `64` routed at hidden `32`, top-`4` per token, plus one shared expert at hidden `128`
- Sparsity: `4` of `64`, `6.2%`

What changed from `M-010`:

- the routed pool grew from `8` experts to `64` and each expert narrowed from `128` to `32`, taking
  sparsity from `50%` to `6.2%`,
- a per-expert `router_bias` buffer is added to the score used for top-`k` **selection only**, while
  the mixture weights are gathered from the raw sigmoid scores,
- routing takes top-`(k+1)` instead of top-`k`, so the `(k+1)`-th entry gives the cutoff each token
  imposes,
- after every training step the bias is recomputed by Quantile Balancing, and the widest bias gap in
  any layer is reported as `bias_span`.

Logged checkpoints:

```text
step=1 train_loss=3.7451 val_loss=3.7426 seconds=12.7 expert_min=0.002 expert_max=0.049 expert_unused=0 bias_span=0.284
step=250 train_loss=1.3758 val_loss=1.3804 seconds=178.7 expert_min=0.013 expert_max=0.018 expert_unused=0 bias_span=0.943
step=500 train_loss=1.1127 val_loss=1.1115 seconds=345.4 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.972
step=750 train_loss=1.0095 val_loss=1.0123 seconds=511.7 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.936
step=1000 train_loss=0.9388 val_loss=0.9460 seconds=677.7 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.702
step=1250 train_loss=0.9035 val_loss=0.9173 seconds=845.1 expert_min=0.015 expert_max=0.017 expert_unused=0 bias_span=0.668
step=1500 train_loss=0.8837 val_loss=0.8803 seconds=1011.4 expert_min=0.014 expert_max=0.017 expert_unused=0 bias_span=0.678
step=1750 train_loss=0.8587 val_loss=0.8601 seconds=1178.7 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.590
step=2000 train_loss=0.8354 val_loss=0.8504 seconds=1344.9 expert_min=0.014 expert_max=0.017 expert_unused=0 bias_span=0.557
step=2250 train_loss=0.8263 val_loss=0.8251 seconds=1510.5 expert_min=0.014 expert_max=0.017 expert_unused=0 bias_span=0.466
step=2500 train_loss=0.8071 val_loss=0.8187 seconds=1676.8 expert_min=0.014 expert_max=0.017 expert_unused=0 bias_span=0.481
step=2750 train_loss=0.7968 val_loss=0.8123 seconds=1843.3 expert_min=0.015 expert_max=0.017 expert_unused=0 bias_span=0.509
step=3000 train_loss=0.7778 val_loss=0.7988 seconds=2008.9 expert_min=0.015 expert_max=0.017 expert_unused=0 bias_span=0.484
```

Main lesson:

- **The balancer works, and the margin is not subtle.** Ideal share at `64` experts is `0.0156`. Step
  `1` reports the load produced with a zero bias, and it is already `24.5x` imbalanced, from `0.002`
  to `0.049`. By step `250` the spread is `1.38x`, and it ends at `1.13x` with no expert ever unused.
- Against `M-010`, which had no balancer, the contrast is direct: `4.66x` final spread there against
  `1.13x` here, and that is while being eight times sparser, which is the regime where imbalance is
  supposed to get *worse*.
- **Quality did not move at all**: `0.7988` against `0.8002`, a delta of `-0.0014` that is far inside
  what one seed can resolve. Perfectly balanced routing, `63%` more parameters, and eight times the
  expert pool bought nothing measurable at this scale.
- **The cost is wall-clock: `2.1x`**, `2008.9s` against `956.5s`. The Python expert loop now runs
  `64` iterations per mixture layer instead of `8`, and each expert is too narrow to use the GPU.
  This is the single strongest argument in the ladder for fused grouped-GEMM dispatch.

### What The Bias Span Reveals

`bias_span` is the widest gap between any two expert biases in a layer, so it measures how hard the
balancer had to work to keep the load flat. It is not monotone:

| Step | 1 | 500 | 1000 | 2000 | 3000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bias_span` | `0.284` | `0.972` | `0.702` | `0.557` | `0.484` |

At initialization the router is random, so its preferences are weak and almost no correction is
needed. As the router starts learning it develops strong preferences, and the bias has to fight
hardest around step `500`. From there the span falls steadily for the rest of training.

The most useful reading is that the span is a direct measure of **how much the router wants to be
imbalanced**, and that this want peaks early and then decays. The experts specialize, demand
spreads out on its own, and the balancer's job gets easier. That is a mechanism you cannot see from
the loss curve at all.

### The Control This Run Does Not Have

The clean ablation would be `4` of `64` **without** the balancer, run to `3000` steps. This run does
not provide it. Step `1` shows what unbalanced routing looks like at initialization, `24.5x`, but
not what it would decay or collapse to over a full run.

So the defensible claim is that the balancer holds load near uniform from the first few hundred
steps onward in a regime that starts badly skewed. The claim it does *not* support is a specific
number for how much damage the absence of a balancer would have done.

## P5-012 Milestone 512 Bounded Feed-Forward Activations

- Script: [`phase5/012_bounded_activations.py`](../../phase5/012_bounded_activations.py)
- Parameters: `14504768`, identical to `M-011`
- Final train loss: `0.7756`
- Final validation loss: `0.7973`
- Wall-clock time: `2493.3s` on a Kaggle `T4`
- Caps: `GATE_CAP = 4.0`, `UP_CAP = 25.0`

What changed from `M-011`:

- both branches of every gated block are soft-capped with `softcap(x, b) = b * tanh(x / b)`,
- the cap replaces the *linear* factor of Swish only, so the gate keeps its own sigmoid and the
  sign-flipping behaviour that distinguishes SwiGLU from plain GLU,
- no parameters are added, since a `tanh` has no weights.

Logged checkpoints:

```text
step=1 train_loss=3.7449 val_loss=3.7424 seconds=14.8 expert_min=0.002 expert_max=0.049 expert_unused=0 bias_span=0.284
step=250 train_loss=1.3243 val_loss=1.3347 seconds=217.1 expert_min=0.013 expert_max=0.022 expert_unused=0 bias_span=0.975
step=500 train_loss=1.0955 val_loss=1.0924 seconds=420.7 expert_min=0.014 expert_max=0.019 expert_unused=0 bias_span=0.911
step=750 train_loss=0.9893 val_loss=0.9942 seconds=624.0 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.893
step=1000 train_loss=0.9236 val_loss=0.9321 seconds=827.2 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.884
step=1250 train_loss=0.8893 val_loss=0.9047 seconds=1034.1 expert_min=0.014 expert_max=0.019 expert_unused=0 bias_span=0.894
step=1500 train_loss=0.8749 val_loss=0.8744 seconds=1235.7 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.862
step=1750 train_loss=0.8535 val_loss=0.8535 seconds=1442.7 expert_min=0.015 expert_max=0.017 expert_unused=0 bias_span=0.766
step=2000 train_loss=0.8281 val_loss=0.8420 seconds=1646.1 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.789
step=2250 train_loss=0.8194 val_loss=0.8190 seconds=1857.9 expert_min=0.014 expert_max=0.019 expert_unused=0 bias_span=0.724
step=2500 train_loss=0.8062 val_loss=0.8175 seconds=2072.1 expert_min=0.014 expert_max=0.017 expert_unused=0 bias_span=0.712
step=2750 train_loss=0.7937 val_loss=0.8066 seconds=2282.8 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.689
step=3000 train_loss=0.7756 val_loss=0.7973 seconds=2493.3 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.687
```

Main lesson:

- **Quality is unchanged**: `0.7973` against `0.7988`, a delta of `-0.0015` at *identical* parameter
  count. Well inside single-seed resolution. This is the expected result for a stability mechanism
  in a run that was never unstable.
- **Cost is `1.24x` wall-clock**, `2493.3s` against `2008.9s`. Two `tanh` calls per gated block, over
  `64` experts in seven layers, is not free at this scale.
- The implementation is faithful to Kimi K3: `softcap` applied to the linear factor of the Swish
  gate and independently to the up branch, with K3's own `b1 = 4` and `b2 = 25`. Verified offline
  that the product is bounded at exactly `b1 * b2 = 100` while plain SwiGLU reaches `250000` over
  the same input range, and that SiTU-GLU tracks SwiGLU to within `2.1%` at `x = 1`.
- **Why the smooth cap rather than DeepSeek's hard clamp**, measured rather than asserted: at
  `x = 30` a `clamp(-10, 10)` has a derivative of exactly `0.000`, while SiTU-GLU still passes
  `1.220`. A coordinate that saturates a hard clamp stops learning; one that saturates a `tanh` does
  not. That is the whole argument for the smooth form.

### An Unexpected Interaction With The Router

Capping the activations made the load balancer work noticeably harder. `bias_span` is higher at
almost every checkpoint, and the gap widens as training goes on:

| Step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M-011` span | `0.972` | `0.702` | `0.678` | `0.557` | `0.481` | `0.484` |
| `M-012` span | `0.911` | `0.884` | `0.862` | `0.789` | `0.712` | `0.687` |

In `M-011` the span decays steadily as the router's preferences spread out. In `M-012` it decays far
more slowly and ends `42%` higher. Load stays balanced either way, since that is the balancer's job,
but it needs a persistently larger correction to get there.

The plausible reading is that bounding each expert's output makes experts harder to tell apart, so
the router's raw preferences stay concentrated for longer and the bias has to keep pushing against
them. This is a guess from one seed and one pair of runs, not a result. It is recorded because it is
the kind of second-order interaction that only shows up once two mechanisms are instrumented at the
same time.

### The Failure Mode Is Real Here

The milestone's own exit criterion is whether outliers form at all at this scale, since a cap that
never engages makes the loss delta meaningless. An instrumented rerun recorded the largest pre-cap
magnitude either branch reached:

| Step | 1 | 250 | 1000 | 2000 | 3000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gate_max` (cap `4`) | `1.69` | `19.29` | `24.27` | `27.33` | `28.29` |
| `up_max` (cap `25`) | `1.67` | `36.93` | `44.29` | `45.72` | `47.22` |

**Both caps engage, hard.** By step `250` the gate branch is already `4.8x` past its cap and the up
branch `1.5x` past its own, and both keep climbing for the rest of training. At the end the gate
reaches `7.1x` its cap.

What that does to the worst coordinate in the tensor:

- unbounded SwiGLU would emit `28.29 * 47.22 = 1336`,
- SiTU-GLU emits `4.00 * 23.88 = 95.5`,
- so the cap removes a **`14x` outlier**.

For scale, a coordinate sitting at `x = 1` produces `0.731`. The largest coordinate in the same
tensor would reach `1336` uncapped, a spread of roughly `1800x`. That is exactly the outlier
structure the two frontier reports describe, and it is present at `6M` parameters in `fp32`.

So the honest verdict is split, and both halves matter:

- **The mechanism is doing real work.** It is not decorative at this scale, which is what the null
  result was expected to show.
- **It still bought no measurable quality**, because `fp32` absorbs a `1336` activation without
  complaint. The value of removing that outlier is realized in `bf16` and `fp8`, in quantization
  ranges, and in loss spikes over far longer runs. None of those are visible in a `3000`-step `fp32`
  run, and this ladder cannot see them.

### Two Methodology Notes

**The instrumentation is provably non-invasive.** The diagnostic rerun reproduced the verbatim run's
losses exactly at every checkpoint, to four decimals, from `3.7424` at step `1` to `0.7973` at step
`3000`. Only wall-clock differs, `2614.2s` against `2493.3s`, so the extra `4.8%` is the tracking
itself. That is the check that makes the diagnostic quotable alongside the primary run.

**The summed dead-expert statistic hides per-layer deaths.** `expert_load_share` sums load across
all seven mixture layers before checking for zeros, so a layer can have dead experts while the total
reports none. Adding a per-layer counter shows exactly that at step `1`:

```text
step=1     expert_unused=0   layer_unused=2
step=250   expert_unused=0   layer_unused=0
```

Two experts in the worst layer received no tokens at all on the first step, and the summed statistic
reported zero unused. By step `250` the balancer had revived both, so this was a transient rather
than a persistent blind spot — but the metric would not have shown it either way. `expert_unused`
should be read as a lower bound on how many experts are idle.

## P5-013 Milestone 513 Attention Sinks

- Script: [`phase5/013_attention_sinks.py`](../../phase5/013_attention_sinks.py)
- Parameters: `14504832`, which is `M-012` plus `64`
- Final train loss: `0.7787`
- Final validation loss: `0.7984`
- Wall-clock time: `2758.2s` on a Kaggle `T4`

What changed from `M-012`:

- every attention layer, local and global, gains one learnable sink logit per query head,
- the sink joins the softmax as an extra column with no value behind it, so the returned weights
  sum to less than one,
- the sink column is dropped after the softmax, which is why no extra value vector is needed,
- `8` heads across `8` layers is `64` new parameters, the cheapest mechanism in the ladder.

Logged checkpoints:

```text
step=1 train_loss=3.7344 val_loss=3.7319 seconds=16.2 expert_min=0.002 expert_max=0.048 expert_unused=0 bias_span=0.279
step=250 train_loss=1.3517 val_loss=1.3550 seconds=238.2 expert_min=0.013 expert_max=0.019 expert_unused=0 bias_span=0.984
step=500 train_loss=1.1083 val_loss=1.1021 seconds=470.0 expert_min=0.014 expert_max=0.019 expert_unused=0 bias_span=0.983
step=750 train_loss=1.0071 val_loss=1.0073 seconds=699.0 expert_min=0.013 expert_max=0.019 expert_unused=0 bias_span=0.912
step=1000 train_loss=0.9361 val_loss=0.9422 seconds=924.9 expert_min=0.014 expert_max=0.019 expert_unused=0 bias_span=0.892
step=1250 train_loss=0.8961 val_loss=0.9098 seconds=1152.0 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.886
step=1500 train_loss=0.8812 val_loss=0.8816 seconds=1378.9 expert_min=0.014 expert_max=0.019 expert_unused=0 bias_span=0.877
step=1750 train_loss=0.8577 val_loss=0.8587 seconds=1606.7 expert_min=0.014 expert_max=0.019 expert_unused=0 bias_span=0.937
step=2000 train_loss=0.8351 val_loss=0.8484 seconds=1835.9 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.904
step=2250 train_loss=0.8251 val_loss=0.8234 seconds=2065.8 expert_min=0.014 expert_max=0.017 expert_unused=0 bias_span=0.768
step=2500 train_loss=0.8089 val_loss=0.8217 seconds=2296.1 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.764
step=2750 train_loss=0.7950 val_loss=0.8083 seconds=2526.5 expert_min=0.014 expert_max=0.017 expert_unused=0 bias_span=0.805
step=3000 train_loss=0.7787 val_loss=0.7984 seconds=2758.2 expert_min=0.014 expert_max=0.018 expert_unused=0 bias_span=0.701
```

Main lesson:

- **No measurable effect**: `0.7984` against `0.7973`, a delta of `+0.0011` for `64` parameters.
  Inside single-seed resolution in the unhelpful direction.
- **Cost is `1.11x` wall-clock.** The concatenation builds a fresh `[B, Hq, T, T+1]` tensor on every
  attention call, and at eight layers that memory traffic is not free for a mechanism whose entire
  parameter budget is `64` floats.
- The implementation is exactly DeepSeek-V4's formula. Computing
  `exp(z) / (sum_k exp(z_k) + exp(z'))` directly and comparing against the concat-softmax-slice form
  agrees to `5.96e-08`, which is `fp32` rounding.

### The Heads Did Want The Escape Hatch, A Little

The sink logits do not stay at their zero initialization. They spread steadily and mostly upward:

| Step | 1 | 500 | 1000 | 2000 | 3000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| min | `-0.00` | `-0.45` | `-0.64` | `-0.62` | `-0.58` |
| mean | `-0.00` | `0.06` | `0.14` | `0.24` | `0.34` |
| max | `0.00` | `0.66` | `0.97` | `1.17` | `1.38` |

So heads differentiate, and the average head moves *toward* the sink rather than away from it. But
the magnitudes are small. What a logit `z` buys depends on how peaked the row is:

| `z` | share of a flat `256`-key row | share if the row is effectively `8` keys |
| ---: | ---: | ---: |
| `-0.58` | `0.22%` | `6.5%` |
| `0.34` | `0.55%` | `14.9%` |
| `1.38` | `1.53%` | `33.2%` |
| `5.00` | `36.7%` | `94.9%` |

A head that genuinely wanted to attend to nothing would drive `z` toward `5`. The most extreme head
here reached `1.38`. The reading is that the escape hatch gets used, moderately, by heads whose
attention is already concentrated, and that no head needed to shut off entirely.

### The Real Question Was Whether It Adds Anything To The Gate

Milestone `507` already installed a data-dependent sigmoid gate on the attention output, which
attacks the same problem from the opposite end: the sink lets a head *retrieve* less, the gate lets
it *emit* less. The frontier is split and neither side ships both. Kimi K3 uses the full-rank output
gate alone. DeepSeek-V4 uses the learnable sink alone.

This run is the direct test of stacking them, and the answer is that the second mechanism adds
nothing measurable once the first is present. That is the expected outcome if both are routes to the
same fix, and it is the most useful thing this milestone produced.

It is one seed and `3000` steps of `fp32`, so this is not evidence that sinks are worthless. It is
evidence that **given a gate, a sink is redundant at this scale**, which is a narrower and more
defensible claim, and it is consistent with nobody at the frontier shipping both.

### Methodology Note

The instrumented rerun reproduced the verbatim run's validation loss exactly at every checkpoint, so
the sink-logit tracking is provably non-invasive and the two runs can be quoted together.

One caveat on the diagnostic's own output: the field printed as `sink_mass` is `sigmoid(-z)`, which
is the weight a single competing key at logit `0` would retain. It is a monotone index of the logit,
not the true attention mass, which depends on the whole row. The raw `min`, `mean`, and `max` logits
above are the numbers to read.

