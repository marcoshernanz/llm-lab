# Phase 5 Learning Log

Runs recorded through 2026-08-16.

This log contains the completed runs from the phase-5 architecture-modernization path.
The roadmap and the frozen control are in [roadmap.md](./roadmap.md).

## Summary

| Run | Script | Steps | Train Loss | Val Loss | Wall Seconds | Parameters |
| --- | ------ | ----: | ---------: | -------: | -----------: | ---------: |
| P5-001 | [`phase5/001_vanilla_decoder.py`](../../phase5/001_vanilla_decoder.py) | 3000 | 3.0721 | 3.0537 | 689.90 | 1631488 |
| P5-002 | [`phase5/002_pre_norm.py`](../../phase5/002_pre_norm.py) | 3000 | 0.9948 | 1.0161 | 663.90 | 1631744 |
| P5-003 | [`phase5/003_rms_norm.py`](../../phase5/003_rms_norm.py) | 3000 | 0.9358 | 0.9563 | 601.10 | 1620352 |
| P5-004 | [`phase5/004_rope.py`](../../phase5/004_rope.py) | 3000 | 0.8548 | 0.8760 | 762.50 | 1587584 |
| P5-005 | [`phase5/005_gqa.py`](../../phase5/005_gqa.py) | 3000 | 0.8488 | 0.8712 | 692.80 | 1456512 |
| P5-006 | [`phase5/006_swiglu.py`](../../phase5/006_swiglu.py) | 3000 | 0.8359 | 0.8584 | 724.60 | 1464704 |

## P5-001 Milestone 501 Vanilla Decoder Baseline

- Script: [`phase5/001_vanilla_decoder.py`](../../phase5/001_vanilla_decoder.py)
- Date: `2026-08-15`
- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Text representation: character-level vocabulary built from the loaded train and validation text
- Vocabulary size: `98`
- Device: `mps`
- Seed: `1337`
- Context length: `256`
- Model dim: `128`
- Heads: `4`, head dim `32`
- Feed-forward dim: `512`
- Decoder blocks: `8`
- Parameters: `1631488`
- Batch size: `32`
- Learning rate: `3e-3`
- Gradient clip norm: `1.0`
- Train steps: `3000`
- Eval interval: `250`
- Eval batches: `32`
- Final train loss: `3.0721`
- Final validation loss: `3.0537`
- Wall-clock time: `689.90s`

Architecture, as the deliberately pre-modern reference point:

- post-norm LayerNorm, `LayerNorm(x + Sublayer(x))`,
- learned absolute position embeddings,
- dense multi-head causal self-attention with biased q/k/v/o projections,
- GELU feed-forward at `4x` width, with biases,
- tied input and output embeddings, both initialized at `std=0.02`,
- AdamW at a constant learning rate, no warmup, no schedule.

Logged checkpoints:

```text
step=1 train_loss=4.0680 val_loss=4.0634 seconds=5.0
step=250 train_loss=3.0746 val_loss=3.0579 seconds=57.3
step=500 train_loss=3.0685 val_loss=3.0586 seconds=109.5
step=750 train_loss=3.0686 val_loss=3.0567 seconds=161.4
step=1000 train_loss=3.0708 val_loss=3.0525 seconds=213.2
step=1250 train_loss=3.0695 val_loss=3.0563 seconds=266.2
step=1500 train_loss=3.0659 val_loss=3.0549 seconds=322.7
step=1750 train_loss=3.0684 val_loss=3.0525 seconds=385.0
step=2000 train_loss=3.0684 val_loss=3.0502 seconds=449.6
step=2250 train_loss=3.0694 val_loss=3.0540 seconds=513.5
step=2500 train_loss=3.0695 val_loss=3.0584 seconds=572.7
step=2750 train_loss=3.0730 val_loss=3.0570 seconds=631.2
step=3000 train_loss=3.0721 val_loss=3.0537 seconds=689.9
```

### The Baseline Does Not Learn At The Control Learning Rate

The run reaches `3.07` by step `250` and then stays there for `2750` more steps, moving less than `0.01` in either direction.
That value is approximately the entropy of the character unigram distribution, so the model learned character frequencies and nothing else.

This means the milestone-501 exit criterion "loss decreases smoothly" is **not** met at the frozen control learning rate.
That was an accepted risk when the control was chosen, but the collapse is more complete than the earlier `400`-step probe suggested.

### Diagnosis

A control sweep on the same script, `400` steps each, separates the possible causes:

| Configuration | Loss @ 400 | Mean grad norm | Steps clipped |
| --- | ---: | ---: | ---: |
| post-norm, lr `3e-3` (control) | `3.0516` | `0.22` | `1%` |
| post-norm, lr `1e-3` | `2.9394` | `0.56` | `10%` |
| post-norm, lr `3e-4` | `2.1515` | `0.81` | `19%` |
| pre-norm, lr `3e-3` | `1.5069` | `1.82` | `50%` |

Conclusions:

- The architecture code is correct. The same script at lr `3e-4` reaches `2.15`, so nothing about the attention, mask, norm, or head is broken.
- The failure is optimization, not divergence. The stalled run has the **smallest** gradients of the four, and gradient clipping fires on only `1%` of steps. The model settled into a flat region where the residual branches contribute nothing, rather than blowing up.
- Healthy learning here correlates with **larger** gradient norms. The pre-norm run has the largest gradients, clips half the time, and gets by far the best loss.
- Lower learning rates monotonically improve the post-norm result over this horizon, which is the classic post-norm signature: it needs warmup or a small learning rate because the normalization sits directly on the residual stream.

### Main Lesson

Post-norm at `8` blocks cannot use the learning rate the rest of the phase-5 ladder is built around.
This is the mechanism behind the historical shift to pre-norm, observed directly rather than read about:

- in post-norm, every layer normalizes the residual stream itself, so there is no identity path from the embedding to the output and no learning rate that suits both shallow and deep behavior,
- in pre-norm, the normalization moves inside the residual branch, the skip path stays pure addition, and the same recipe that collapses post-norm reaches `1.51` in `400` steps.

### Open Question For Milestone 502

The `3.0537` control result is a degenerate baseline, so a raw `501` to `502` comparison will overstate the gain from norm placement alone.
Two honest options:

1. keep `3.0537` as the headline `M-501` number and report the pre-norm gain against it, noting the collapse every time it is cited,
2. or add a supplementary `post-norm, lr 3e-4` run at the full `3000` steps as a "post-norm best effort" reference, and compare `M-502` against both.

Option 2 costs one extra run and makes the rest of the ladder more defensible.
This should be decided before `M-502` is recorded.

## P5-002 Milestone 502 Pre-Norm Residual Stream

- Script: [`phase5/002_pre_norm.py`](../../phase5/002_pre_norm.py)
- Date: `2026-08-16`
- Parameters: `1631744`
- Final train loss: `0.9948`
- Final validation loss: `1.0161`
- Wall-clock time: `663.90s`

What changed from `M-501`, and nothing else:

- the normalization moved inside the residual branch, from `LayerNorm(x + Sublayer(x))` to `x + Sublayer(LayerNorm(x))`,
- one final `LayerNorm` was added after the block stack, because pre-norm leaves the residual stream unnormalized all the way to the LM head.

That is four lines of code and `+256` parameters.

Logged checkpoints:

```text
step=1 train_loss=4.1776 val_loss=4.1743 seconds=5.2
step=250 train_loss=2.3459 val_loss=2.3460 seconds=58.4
step=500 train_loss=2.2191 val_loss=2.2263 seconds=111.7
step=750 train_loss=1.6943 val_loss=1.6976 seconds=165.0
step=1000 train_loss=1.3785 val_loss=1.3745 seconds=218.7
step=1250 train_loss=1.2522 val_loss=1.2495 seconds=272.8
step=1500 train_loss=1.1776 val_loss=1.1775 seconds=326.0
step=1750 train_loss=1.1166 val_loss=1.1027 seconds=379.6
step=2000 train_loss=1.0752 val_loss=1.0835 seconds=435.3
step=2250 train_loss=1.0594 val_loss=1.0612 seconds=492.1
step=2500 train_loss=1.0338 val_loss=1.0315 seconds=548.6
step=2750 train_loss=1.0074 val_loss=1.0186 seconds=605.3
step=3000 train_loss=0.9948 val_loss=1.0161 seconds=663.9
```

Main lesson:

- Norm placement is the difference between a model that does not learn and a model that learns well. Validation loss goes from `3.0537` to `1.0161` with no change in size, data, seed, or learning rate.
- The mechanism is visible outside of training: feeding activations scaled by `50x` into one block, the post-norm block returns output with standard deviation `1.000`, while the pre-norm block returns `50.265`. Pre-norm leaves the residual stream untouched, so there is an identity path from the embedding to the output and gradients never traverse a norm on the way back.
- Pre-norm also trains with much larger gradients. In the `400`-step control sweep the pre-norm run clipped on `50%` of steps against `1%` for the collapsed post-norm run, so aggressive clipping here is a sign of health, not instability.

Caveat that must be repeated whenever this comparison is cited:

- `M-501` collapsed at the control learning rate, so the `3.0537` to `1.0161` gap credits norm placement with a gap that is partly a learning-rate mismatch.
- The `400`-step sweep gives the fairer reference: post-norm reaches `2.15` at lr `3e-4` where pre-norm reaches `1.51` at lr `3e-3`.
- A full `3000`-step post-norm run at lr `3e-4` is still the honest supplementary baseline and has not been run.

## P5-003 Milestone 503 RMSNorm And Bias Removal

- Script: [`phase5/003_rms_norm.py`](../../phase5/003_rms_norm.py)
- Date: `2026-08-16`
- Parameters: `1620352`
- Final train loss: `0.9358`
- Final validation loss: `0.9563`
- Wall-clock time: `601.10s`

What changed from `M-502`:

- `LayerNorm` became `RMSNorm`: no mean subtraction, no shift parameter, normalizing by the root mean square instead of the centered standard deviation,
- every `nn.Linear` in the model now uses `bias=False`.

Logged checkpoints:

```text
step=1 train_loss=4.1971 val_loss=4.1946 seconds=4.5
step=250 train_loss=2.3275 val_loss=2.3284 seconds=50.9
step=500 train_loss=1.9631 val_loss=1.9771 seconds=98.6
step=750 train_loss=1.4360 val_loss=1.4377 seconds=150.1
step=1000 train_loss=1.2298 val_loss=1.2242 seconds=200.1
step=1250 train_loss=1.1462 val_loss=1.1439 seconds=249.2
step=1500 train_loss=1.0952 val_loss=1.0949 seconds=298.6
step=1750 train_loss=1.0424 val_loss=1.0312 seconds=350.4
step=2000 train_loss=1.0073 val_loss=1.0155 seconds=400.8
step=2250 train_loss=0.9951 val_loss=0.9929 seconds=450.5
step=2500 train_loss=0.9805 val_loss=0.9792 seconds=500.2
step=2750 train_loss=0.9528 val_loss=0.9642 seconds=550.7
step=3000 train_loss=0.9358 val_loss=0.9563 seconds=601.1
```

Main lesson:

- Removing parameters and arithmetic improved every axis at once: validation loss falls from `1.0161` to `0.9563`, parameters fall by `11392` (`9216` from six linear biases across eight blocks, `2176` from seventeen norm shifts), and wall-clock falls from `663.90s` to `601.10s`, a `9.5%` speedup.
- This is the cheapest win in the ladder so far, and it is a pure simplification. The literature claim that re-centering contributes little is reproduced here: dropping it did not cost quality.
- The speedup is real but should not be over-read at this scale. RMSNorm removes one reduction pass and one subtraction per norm, and bias removal deletes an add per projection; both matter more as models get larger and more bandwidth-bound.
- One implementation trap worth recording: the first draft normalized by `x.var(correction=0)` rather than the mean square. Those agree only when the per-token mean is zero. On input shifted by a constant of `3.0` it produced output RMS `3.164` instead of `1.0`, and pre-norm is exactly the setting where the residual stream is free to drift away from zero mean.


## P5-004 Milestone 504 Rotary Position Embeddings

- Script: [`phase5/004_rope.py`](../../phase5/004_rope.py)
- Date: `2026-08-16`
- Parameters: `1587584`
- Final train loss: `0.8548`
- Final validation loss: `0.8760`
- Wall-clock time: `762.50s`

What changed from `M-503`:

- the learned absolute position table is gone, and with it `32768` parameters,
- queries and keys are rotated by a position-dependent angle after the head split, using the split-half convention,
- values are not rotated, because position belongs in the comparison rather than in the retrieved content,
- `rope_cos` and `rope_sin` are precomputed buffers of shape `[T, Dh]`, built once per attention layer.

Logged checkpoints:

```text
step=1 train_loss=4.2291 val_loss=4.2293 seconds=5.9
step=250 train_loss=1.5966 val_loss=1.5898 seconds=62.8
step=500 train_loss=1.2012 val_loss=1.2109 seconds=119.9
step=750 train_loss=1.0802 val_loss=1.0813 seconds=177.8
step=1000 train_loss=1.0094 val_loss=1.0080 seconds=235.4
step=1250 train_loss=0.9766 val_loss=0.9776 seconds=296.6
step=1500 train_loss=0.9513 val_loss=0.9553 seconds=362.7
step=1750 train_loss=0.9246 val_loss=0.9167 seconds=427.0
step=2000 train_loss=0.9037 val_loss=0.9162 seconds=492.1
step=2250 train_loss=0.8917 val_loss=0.8954 seconds=556.7
step=2500 train_loss=0.8845 val_loss=0.8889 seconds=626.5
step=2750 train_loss=0.8676 val_loss=0.8792 seconds=694.0
step=3000 train_loss=0.8548 val_loss=0.8760 seconds=762.5
```

Main lesson:

- RoPE is the largest quality win of the ladder so far and it is also a simplification: validation loss falls from `0.9563` to `0.8760` while the model loses `32768` parameters.
- Relative position is what the model wanted. The learned table gave every absolute position its own vector and forced the model to infer that only differences matter; RoPE makes the attention score between positions `m` and `n` depend on `n - m` by construction. Verified directly: the score for a gap of `2` is identical at positions `(3, 5)`, `(10, 12)`, and `(100, 102)`.
- The rotation is length-preserving, so position is injected without disturbing activation scale. This is the mechanical reason it can be applied at every layer while an additive table cannot.
- Cost is real but modest: `762.50s` against `601.10s`, about `27%` slower. A direct measurement puts RoPE at `204.5ms` per step against `165.8ms` with the rotation replaced by the identity, roughly `23%`, so the run-level number and the microbenchmark agree.
- `rotate_half` is the expensive half of the operation, `0.410ms` of `1.067ms` per call, because `chunk` plus `cat` allocates a new tensor instead of reading in place. That is the piece a fused kernel would remove, and it is a good candidate for the phase-4 Triton milestone.

### Two Methodology Findings

The first run of this milestone was performed with macOS Low Power Mode enabled and took `1390.6s`, roughly double the clean rerun at `762.5s`, with a steady slowdown from the first checkpoint to the last.

- Wall-clock is only comparable across milestones if the power mode is fixed. Low Power Mode must be off for every phase-5 run.
- The wall-clock times recorded for `P5-001` through `P5-003` were measured before this was noticed and may or may not be affected. They should be treated as provisional until those runs are repeated under known conditions.

The two `M-504` runs also did not reproduce each other exactly, despite identical code, seed, and data:

| Step | Clean rerun | Low Power run |
| ---: | ---: | ---: |
| 250 | `1.5898` | `1.5933` |
| 1500 | `0.9553` | `0.9518` |
| 3000 | `0.8760` | `0.8727` |

- Step `1` agrees to four decimals and the runs drift apart after that, which points at floating-point non-determinism in MPS kernels rather than at a data or seeding difference.
- This gives an accidental estimate of the run-to-run noise floor: about `0.003` in final validation loss.
- Differences smaller than roughly `0.01` therefore cannot be called from a single run. The `0.08` gap from `M-003` to `M-004` is far above the floor, but milestones expected to be near-neutral, such as grouped-query attention, will need repeated seeds before any claim is made.

## P5-005 Milestone 505 Grouped-Query Attention

- Script: [`phase5/005_gqa.py`](../../phase5/005_gqa.py)
- Date: `2026-08-16`
- Parameters: `1456512`
- Final train loss: `0.8488`
- Final validation loss: `0.8712`
- Wall-clock time: `692.80s`
- Query heads: `4`, key and value heads: `2`, group size `2`

What changed from `M-504`:

- `k_proj` and `v_proj` now project to `NUM_KV_HEADS * D_HEAD` instead of `D_MODEL`, which is where the saving comes from,
- keys and values are rotated as `Hkv` heads and only then repeated to `Hq` with `repeat_interleave`, so the rotation runs on half as many heads,
- `split_heads` takes the head count as an argument, since queries and keys no longer agree on it.

Logged checkpoints:

```text
step=1 train_loss=4.1674 val_loss=4.1652 seconds=5.5
step=250 train_loss=1.6551 val_loss=1.6515 seconds=59.5
step=500 train_loss=1.2323 val_loss=1.2411 seconds=113.0
step=750 train_loss=1.1044 val_loss=1.1019 seconds=166.5
step=1000 train_loss=1.0175 val_loss=1.0153 seconds=220.0
step=1250 train_loss=0.9753 val_loss=0.9782 seconds=275.0
step=1500 train_loss=0.9503 val_loss=0.9568 seconds=333.5
step=1750 train_loss=0.9155 val_loss=0.9125 seconds=392.9
step=2000 train_loss=0.8945 val_loss=0.9098 seconds=452.1
step=2250 train_loss=0.8890 val_loss=0.8925 seconds=511.7
step=2500 train_loss=0.8769 val_loss=0.8823 seconds=571.9
step=2750 train_loss=0.8588 val_loss=0.8744 seconds=632.7
step=3000 train_loss=0.8488 val_loss=0.8712 seconds=692.8
```

Main lesson:

- Grouped-query attention is close to free at this scale. Parameters fall by `131072` (`8.3%`), wall-clock falls from `762.50s` to `692.80s` (`9.1%`), and validation loss does not get worse.
- The quality result must be stated carefully. `M-505` reaches `0.8712` against `0.8760` for `M-504`, but the two `M-504` runs of identical code spanned `0.8727` to `0.8760`. A gap of `0.0048` is at the edge of that spread, so the defensible claim is that halving the key and value heads costs nothing measurable here, not that it helps.
- The real payoff is invisible in this experiment. The KV cache at this configuration drops from `2.00MB` to `1.00MB` at `T=256`, and there is no cache during training, so nothing in the loss curve or wall-clock reflects the mechanism's actual purpose. This is a systems change measured in a setting that cannot show its benefit.
- The wall-clock gain that does appear comes from the smaller key and value projections, partly offset by the `repeat_interleave` that materializes the shared heads. An `expand` plus `reshape` would avoid that copy, at the cost of readability.
- Implementation notes worth keeping: `Tensor.repeat` has no `dim` argument, so `x.repeat(n, dim=2)` raises `TypeError`; the correct call is `repeat_interleave`, which also gives the group-contiguous pairing that HuggingFace uses. Verified directly that query heads `0` and `1` share key/value head `0` while heads `1` and `2` do not share.

## P5-006 Milestone 506 SwiGLU Feed-Forward

- Script: [`phase5/006_swiglu.py`](../../phase5/006_swiglu.py)
- Date: `2026-08-16`
- Parameters: `1464704`
- Final train loss: `0.8359`
- Final validation loss: `0.8584`
- Wall-clock time: `724.60s`
- Feed-forward dim: `344`, down from `512`

What changed from `M-505`:

- the two-matrix GELU MLP became a three-matrix gated MLP, `down(silu(gate(x)) * up(x))`,
- `D_FFN` fell from `512` to `344`, two thirds of `4 * D_MODEL` rounded to a multiple of eight, so the gated block has the same parameter count as the dense one it replaces.

Logged checkpoints:

```text
step=1 train_loss=4.2226 val_loss=4.2222 seconds=6.1
step=250 train_loss=1.5637 val_loss=1.5618 seconds=61.6
step=500 train_loss=1.1870 val_loss=1.1924 seconds=117.6
step=750 train_loss=1.0589 val_loss=1.0598 seconds=173.8
step=1000 train_loss=0.9858 val_loss=0.9867 seconds=229.8
step=1250 train_loss=0.9520 val_loss=0.9552 seconds=286.1
step=1500 train_loss=0.9275 val_loss=0.9321 seconds=345.5
step=1750 train_loss=0.8997 val_loss=0.8969 seconds=408.0
step=2000 train_loss=0.8831 val_loss=0.8970 seconds=471.2
step=2250 train_loss=0.8775 val_loss=0.8805 seconds=532.7
step=2500 train_loss=0.8663 val_loss=0.8720 seconds=597.1
step=2750 train_loss=0.8499 val_loss=0.8664 seconds=660.5
step=3000 train_loss=0.8359 val_loss=0.8584 seconds=724.6
```

Main lesson:

- Gating is a genuine quality mechanism, not an artifact of extra capacity. Validation loss falls from `0.8712` to `0.8584` for `0.56%` more parameters, and the gap of `0.0128` is roughly four times the measured noise floor.
- The parameter matching is what makes that claim possible. A gated block has three matrices where the dense block has two, so keeping `D_FFN = 4 * D_MODEL` would have added `524288` parameters, a `36%` larger model, and any improvement would have been unattributable. The `2/3` rule brings the feed-forward block to `132096` parameters against `131072`.
- The cost is wall-clock: `724.60s` against `692.80s`, about `4.6%` slower. Three narrower matrix multiplications are slower here than two wider ones, because each launch has fixed overhead and the narrower shapes use the hardware less efficiently.
- What the gate buys mechanically is a multiplicative interaction. A projection followed by an activation can only add contributions; gating lets one projection scale another elementwise, per token and per feature. Half the gate values are negative at initialization, because Swish dips below zero near the origin, so the gate can flip a feature's sign rather than only attenuate it. A sigmoid gate bounded in `[0, 1]` cannot do that, which is part of why SwiGLU beat the original GLU.
