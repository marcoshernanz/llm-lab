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
