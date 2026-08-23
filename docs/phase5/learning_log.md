# Phase 5 Learning Log

Runs recorded through 2026-08-16.

This log contains the completed runs from the phase-5 architecture-modernization path.
The roadmap and the frozen control are in [roadmap.md](./roadmap.md).

## Summary

| Run | Script | Steps | Train Loss | Val Loss | Wall Seconds | Parameters |
| --- | ------ | ----: | ---------: | -------: | -----------: | ---------: |
| P5-001 | [`phase5/001_vanilla_decoder.py`](../../phase5/001_vanilla_decoder.py) | 3000 | 3.0669 | 3.0557 | 267.80 | 1631488 |
| P5-002 | [`phase5/002_pre_norm.py`](../../phase5/002_pre_norm.py) | 3000 | 0.9227 | 0.9354 | 273.30 | 1631744 |
| P5-003 | [`phase5/003_rms_norm.py`](../../phase5/003_rms_norm.py) | 3000 | 0.9369 | 0.9498 | 264.70 | 1620352 |
| P5-004 | [`phase5/004_rope.py`](../../phase5/004_rope.py) | 3000 | 0.8487 | 0.8644 | 300.40 | 1587584 |
| P5-005 | [`phase5/005_gqa.py`](../../phase5/005_gqa.py) | 3000 | 0.8432 | 0.8594 | 291.60 | 1456512 |
| P5-006 | [`phase5/006_swiglu.py`](../../phase5/006_swiglu.py) | 3000 | 0.8483 | 0.8663 | 285.30 | 1464704 |
| P5-007 | [`phase5/007_qk_norm_gated_attention.py`](../../phase5/007_qk_norm_gated_attention.py) | 3000 | 0.7874 | 0.8056 | 329.20 | 1596288 |
| P5-008 | [`phase5/008_hybrid_attention.py`](../../phase5/008_hybrid_attention.py) | 3000 | 0.7814 | 0.8008 | 313.90 | 1596288 |
| P5-009 | [`phase5/009_mla.py`](../../phase5/009_mla.py) | 3000 | 0.7878 | 0.8079 | 321.70 | 1633152 |
| P5-010 | [`phase5/010_moe.py`](../../phase5/010_moe.py) | 3000 | 0.7796 | 0.7984 | 476.50 | 2328448 |

All runs above were produced on a Kaggle `Tesla T4` at seed `1337`.
They replace an earlier set measured on local `mps`, which turned out not to be reproducible.
The per-milestone sections below were written against the `mps` numbers; their mechanism analysis stands, but every loss figure quoted inside them is superseded by this table.

## Platform Migration, And What It Invalidated

The first eight milestones were measured on Apple `mps`. Three separate problems made those numbers unusable as a comparison set.

**macOS Low Power Mode halved throughput.** One `M-504` run took `1390.6s` against `762.5s` for the identical script with the setting off. Wall-clock is only comparable when the power mode is fixed.

**`mps` is not reproducible.** Two runs of identical code, identical seed, and identical data diverge:

| Milestone | Run A | Run B | Spread |
| --- | ---: | ---: | ---: |
| `M-001` | `3.0537` | `3.0537` | `0.0000` |
| `M-002` | `1.0161` | `0.9628` | `0.0533` |
| `M-003` | `0.9563` | `0.9773` | `0.0210` |
| `M-004` | `0.8760` | `0.8727` | `0.0033` |

A short probe locates the divergence precisely: two `200`-step `mps` runs agree exactly at step `1` and step `10`, differ by `0.000003` at step `50`, and by `0.002513` at step `200`. The same test on CPU is bit-identical at every checkpoint, so the cause is `mps` kernel non-determinism in the backward pass, compounding over training. `torch.use_deterministic_algorithms(True)` does not fix it and does not raise; the runs still diverge by `0.014435` at step `100`.

`M-001` is the exception that proves the mechanism: its collapsed solution is a strong enough attractor that float noise cannot move it.

**Thermal drift across sequential runs.** Rerunning `M-001` through `M-003` back to back gave `770.3s`, `850.4s`, and `724.8s` against original figures of `689.9s`, `663.9s`, and `601.1s`. Every rerun was slower, so a laptop cannot hold wall-clock steady across a batch.

### Why Kaggle T4

A benchmark of the phase-5 model shape, pushed to Kaggle as a self-contained script:

| | local `mps` | Kaggle `T4` |
| --- | ---: | ---: |
| benchmark step time | about `300ms` | `90.0ms` |
| real `M-008` run, `3000` steps | `917.7s` | `313.9s` |
| same seed twice | diverges by `0.02` to `0.05` | bit-identical |
| deterministic flags needed | ineffective | none |

The T4 reproduces itself exactly at default settings, at every checkpoint, without `use_deterministic_algorithms`. Enabling that flag costs about `6%` and changes nothing.

Two operational findings worth keeping:

- **`P100` cannot run current PyTorch.** It fails immediately with `CUDA error: no kernel image is available for execution on the device`. `P100` is compute capability `6.0` and Kaggle's PyTorch `2.10+cu128` no longer ships `sm_60` kernels. The accelerator preference is now `T4`, with `P100` removed rather than demoted.
- **Kaggle allows two concurrent GPU sessions.** Pushing more returns `Maximum batch GPU session count of 2 reached` as an ordinary output line, not a failing exit code, so a batch script that discards push output loses jobs silently. Five of the first eight pushes vanished this way.

Migration cost was one line, because the phase-5 scripts import nothing from the repo:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
```

Results are **not** comparable across devices. The same script and seed gives `0.8040` on `mps` and `0.8008` on `T4`, which is why the whole ladder was rerun rather than extended.

## Revised Conclusions On One Consistent Platform

| Step | Change | Val loss | Delta | Parameters |
| --- | --- | ---: | ---: | ---: |
| 501 | vanilla post-norm baseline | `3.0557` | — | `1631488` |
| 502 | pre-norm | `0.9354` | `-2.1203` | `1631744` |
| 503 | RMSNorm, no biases | `0.9498` | `+0.0144` | `1620352` |
| 504 | RoPE | `0.8644` | `-0.0854` | `1587584` |
| 505 | grouped-query attention | `0.8594` | `-0.0050` | `1456512` |
| 506 | SwiGLU | `0.8663` | `+0.0069` | `1464704` |
| 507 | QK-Norm and gated attention | `0.8056` | `-0.0607` | `1596288` |
| 508 | layerwise hybrid attention | `0.8008` | `-0.0048` | `1596288` |
| 509 | latent attention on the global layers | `0.8079` | `+0.0071` | `1633152` |
| 510 | sparse mixture-of-experts feed-forward | `0.7984` | `-0.0095` | `2328448` |

Three changes move the loss clearly:

- **pre-norm**, by `2.12`, which is the difference between a model that learns and one that does not,
- **RoPE**, by `0.085`, while removing `32768` parameters,
- **QK-Norm with gated attention**, by `0.061`, but with `9%` more parameters, so mechanism and capacity are confounded in that number.

Five changes do not move it: RMSNorm `+0.0144`, GQA `-0.0050`, SwiGLU `+0.0069`, hybrid attention `-0.0048`, and latent attention `+0.0071`. RMSNorm and SwiGLU actually came out marginally worse here, having looked like wins on `mps`, which is what reading a result out of noise looks like in hindsight.

That is the expected outcome rather than a disappointment:

- RMSNorm is a **simplification**. It removed `11392` parameters and one reduction pass per norm at no measurable quality cost, which is exactly the claim the literature makes.
- GQA, hybrid attention, and latent attention are **inference-economics mechanisms**. GQA halves the KV cache, hybrid attention bounds the local layers' cache at `WINDOW_SIZE` tokens, and latent attention cuts a global layer's cache from `128` numbers per token to `80`. None can pay off in a training-only benchmark, and none cost anything measurable here, which is the result that matters.
- SwiGLU at matched parameters is a modest effect that a `1.5M`-parameter character model over `3000` steps cannot resolve.

**What is still missing is seeds.** Every number here is one deterministic run at seed `1337`. Determinism means each run reproduces itself, not that the measurement is precise; a different seed changes initialization and data order. Differences under roughly `0.02` should not be called from a single seed. At `5` minutes per T4 run, three seeds for all nine milestones is about `2.5` hours of quota, and that is the step that would turn this table into a measurement.

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

## P5-007 Milestone 507 QK-Norm And Gated Attention

- Script: [`phase5/007_qk_norm_gated_attention.py`](../../phase5/007_qk_norm_gated_attention.py)
- Date: `2026-08-16`
- Parameters: `1596288`
- Final train loss: `0.7888`
- Final validation loss: `0.8147`
- Wall-clock time: `879.50s`

What changed from `M-506`:

- `RMSNorm` gained a `dim` argument so it can normalize the head axis as well as the model axis,
- queries and keys are normalized per head, after the head split and before the rotation,
- a full-rank sigmoid gate, computed from the layer input, scales the attention output before the output projection.

Logged checkpoints:

```text
step=1 train_loss=4.4165 val_loss=4.4168 seconds=6.6
step=250 train_loss=1.2731 val_loss=1.2727 seconds=75.1
step=500 train_loss=1.0464 val_loss=1.0548 seconds=143.1
step=750 train_loss=0.9654 val_loss=0.9674 seconds=210.8
step=1000 train_loss=0.9162 val_loss=0.9159 seconds=277.8
step=1250 train_loss=0.8847 val_loss=0.8893 seconds=345.6
step=1500 train_loss=0.8648 val_loss=0.8722 seconds=416.7
step=1750 train_loss=0.8420 val_loss=0.8414 seconds=489.7
step=2000 train_loss=0.8279 val_loss=0.8449 seconds=567.0
step=2250 train_loss=0.8233 val_loss=0.8261 seconds=644.3
step=2500 train_loss=0.8157 val_loss=0.8234 seconds=719.0
step=2750 train_loss=0.8005 val_loss=0.8199 seconds=793.5
step=3000 train_loss=0.7888 val_loss=0.8147 seconds=879.5
```

Main lesson:

- This is the largest single-milestone gain since pre-norm: validation loss falls from `0.8584` to `0.8147`, a gap of `0.0437`, roughly fifteen times the noise floor.
- The gain is not cleanly attributable. The gate is a full `D x D` matrix, so the model grew by `131584` parameters (`9.0%`), and this is the first milestone in the ladder that buys capacity rather than shedding it. Some of the improvement is the mechanism and some is size; this run does not separate them. The cheap control, if it is ever needed, is a QK-Norm-only variant, since the norms cost only `512` parameters.
- The mechanism QK-Norm defends against was measured directly rather than assumed. Feeding activations scaled by `1`, `10`, and `100` into the attention path gives maximum absolute logits of `1.48`, `158.78`, and `17464.14` without QK-Norm, and `3.50`, `3.50`, `3.50` with it. Softmax at a logit of `17464` is exactly one-hot, its gradient is zero, and in `bf16` the exponential overflows.
- Cost is `21%` wall-clock, `879.50s` against `724.60s`, from two extra normalizations per attention layer and one extra `D x D` projection.

### Choosing The Gate Granularity

The gate can be one scalar per head, roughly `1K` parameters, or one value per feature, `131K`. The per-head form was tempting because it would have kept the parameter comparison clean.

The elementwise form was chosen instead, on evidence:

- the paper that introduced the mechanism ablated about thirty variants and recommends elementwise, with `headwise_attn_output_gate: false` and `elementwise_attn_output_gate: true` as its default configuration, reporting that elementwise gating produces sparser and more structured attention maps,
- Kimi K3 states plainly that its gate projection is full rank,
- Qwen3-Next, Qwen3.5, and Arcee Trinity all ship gated attention in production, applied after the attention output and before the output projection, at under `2%` wall-clock overhead.

The granularity is part of the mechanism rather than a cost knob, so the frontier form was kept and the parameter caveat recorded instead.

## P5-008 Milestone 508 Layerwise Hybrid Attention

- Script: [`phase5/008_hybrid_attention.py`](../../phase5/008_hybrid_attention.py)
- Date: `2026-08-16`
- Parameters: `1596288`, identical to `M-507`
- Final train loss: `0.7814`
- Final validation loss: `0.8008`
- Wall-clock time: `313.90s` on a Kaggle `T4`
- Window size: `64`, one global layer every `4`

What changed from `M-507`:

- six of the eight layers attend only to the last `WINDOW_SIZE` tokens, using a mask that blocks both the future and the distant past,
- the remaining two layers, at indices `3` and `7`, attend over the whole sequence and carry no positional encoding at all,
- no parameters were added or removed, since masks are buffers and RoPE has no weights.

Logged checkpoints:

```text
step=1 train_loss=4.4791 val_loss=4.4798 seconds=3.2
step=250 train_loss=1.2486 val_loss=1.2554 seconds=27.8
step=500 train_loss=1.0442 val_loss=1.0431 seconds=53.1
step=750 train_loss=0.9639 val_loss=0.9651 seconds=79.4
step=1000 train_loss=0.9070 val_loss=0.9169 seconds=105.8
step=1250 train_loss=0.8747 val_loss=0.8906 seconds=131.7
step=1500 train_loss=0.8673 val_loss=0.8649 seconds=157.6
step=1750 train_loss=0.8478 val_loss=0.8481 seconds=183.7
step=2000 train_loss=0.8279 val_loss=0.8423 seconds=209.7
step=2250 train_loss=0.8183 val_loss=0.8192 seconds=235.7
step=2500 train_loss=0.8073 val_loss=0.8188 seconds=261.8
step=2750 train_loss=0.7954 val_loss=0.8080 seconds=287.8
step=3000 train_loss=0.7814 val_loss=0.8008 seconds=313.9
```

Main lesson:

- Restricting three quarters of the layers to a `64`-token window, and stripping positional encoding from the rest, costs nothing measurable: `0.8008` against `0.8056`, a difference well inside single-seed noise. This is the cleanest attribution in the ladder because the parameter count is unchanged.
- Local layers do not limit what the model can see, only what each layer sees directly. Reach compounds with depth: `64` tokens at layer `0`, `127` at layer `1`, `190` at layer `2`, then the global layer at `3` makes it exact. The global layers exist to make long range **precise**, not possible, since information arriving through stacked local layers has been averaged at every hop.
- The NoPE global layers still receive position information indirectly. With the final token held fixed and the preceding `127` shuffled, the last-position logits still move by `0.898`, so position is reaching them through what the local RoPE layers wrote into the residual stream.
- No speedup is expected or observed. The implementation computes the full `[T, T]` score matrix and then masks it, so masking makes the answer correct without making the arithmetic cheaper. At `T=256` the quadratic part is only about half of attention's cost anyway; at `T=4096` it is `94%`, and at `1M` tokens essentially all of it. This mechanism is aimed at a problem this context length does not have.
- The genuine payoff is invisible here: a local layer never needs more than `WINDOW_SIZE` keys and values in cache regardless of sequence length, which is what makes million-token context affordable.

## P5-009 Milestone 509 Multi-Head Latent Attention

- Script: [`phase5/009_mla.py`](../../phase5/009_mla.py)
- Date: `2026-08-16`
- Parameters: `1633152`
- Final train loss: `0.7878`
- Final validation loss: `0.8079`
- Wall-clock time: `321.70s` on a Kaggle `T4`
- Latent dim: `64`, rope dim `16`, applied to the two global layers only

What changed from `M-508`:

- the two global layers replace `k_proj` and `v_proj` with a shared down-projection to a `64`-dim latent plus two up-projections, so keys and values are rebuilt from one cached vector,
- those layers regain full `Hq` heads for keys and values, so `repeat_kv_heads` is gone from the global path,
- position returns to the global layers through a decoupled rope path: `32` content dims per head carry no rotation, and `16` rope dims per head are rotated, with a single shared rope key broadcast across heads,
- the six local layers keep grouped-query attention unchanged,
- `CausalSelfAttention` split into `LocalSelfAttention` and `GlobalSelfAttention`, with the shared machinery moved to module-level functions.

Logged checkpoints:

```text
step=1 train_loss=4.2231 val_loss=4.2201 seconds=3.2
step=250 train_loss=1.2182 val_loss=1.2239 seconds=28.7
step=500 train_loss=1.0310 val_loss=1.0327 seconds=54.8
step=750 train_loss=0.9563 val_loss=0.9614 seconds=81.6
step=1000 train_loss=0.9078 val_loss=0.9155 seconds=108.5
step=1250 train_loss=0.8786 val_loss=0.8940 seconds=135.0
step=1500 train_loss=0.8726 val_loss=0.8701 seconds=161.6
step=1750 train_loss=0.8530 val_loss=0.8540 seconds=188.4
step=2000 train_loss=0.8338 val_loss=0.8446 seconds=215.0
step=2250 train_loss=0.8244 val_loss=0.8243 seconds=241.7
step=2500 train_loss=0.8145 val_loss=0.8248 seconds=268.3
step=2750 train_loss=0.8023 val_loss=0.8141 seconds=295.0
step=3000 train_loss=0.7878 val_loss=0.8079 seconds=321.7
```

Main lesson:

- Latent attention costs nothing measurable and buys nothing measurable here: `0.8079` against `0.8008`, a difference inside single-seed noise, for `36864` more parameters. That is the fourth mechanism in a row whose payoff is invisible in a training-only benchmark.
- The cache arithmetic is the actual result. A global layer now caches `64 + 16 = 80` numbers per token instead of `128`, and every query head gets its own keys and values again rather than sharing. Local layers still cache `128`, but bounded to a `64`-token window.
- DeepSeek's own sizing rule does not transfer. They set the latent to `4 * d_head`, which at this scale is `4 * 32 = 128 = D_MODEL`, meaning no compression at all. Their `28x` saving comes from having `128` heads of `128` dims to compress into `512`; a four-head model has far less redundancy to exploit. `D_LATENT = 64` was chosen instead as half the model dim, which beats the GQA cache while staying a real bottleneck.
- Decoupled rope was implemented even though the NoPE variant would have worked, because that conflict is the entire reason MLA has its shape. The absorption trick, `q_content . (W_uk c) = (W_uk^T q_content) . c`, only holds when the matrix between query and latent is constant. A rotation makes it `R_(n-m)`, which depends on the key's position, so one folded query would serve exactly one key. Verified directly: folding once and reusing it gives zero error at `n = m` and growing error everywhere else.
- The split is exact rather than approximate. A dot product over concatenated vectors equals the sum of the dot products over the pieces, so scoring on `cat(content, rope)` is identical to `content . content + rope . rope`. Confirmed the rope half stays relative: the same gap gives the same score at different absolute positions, `5.483795` at `(3, 5)` against `5.483794` at `(20, 22)`.
- Values stay `Dh`-wide while scores use `Dh + Dr`. Rope dims decide how much to attend and carry nothing to retrieve, which is why `combine_heads` and `o_proj` are untouched and DeepSeek's head dims read as an odd `128 + 64`.
- Worth recording against the ladder: **DeepSeek V4 has since abandoned MLA.** It uses shared key-equals-value multi-query attention with `num_key_value_heads = 1` plus compression along the *sequence* dimension, on the grounds that at a million tokens sequence length dominates memory, not head count. So this milestone implements a mechanism the frontier is already moving past, which is worth knowing while building it.

### Structural Note

`CausalSelfAttention` was split into two classes here, with `split_heads`, `combine_heads`, `repeat_kv_heads`, `rotate_half`, `apply_rope`, `rope_tables`, and `attend` lifted to module-level functions.

The trigger was that local and global layers stopped sharing weights, not just a mask. Keeping one class would have meant branching in both `__init__` and the key/value path, and the branchy version would have been copied forward into `010`, `011`, and `012` before being torn out at `515` anyway. Module-level helpers keep duplication near zero without inheritance, and they degrade gracefully: when the local mixer becomes a linear-attention recurrence, that class simply stops calling `attend` and `apply_rope` instead of needing them carved out of a shared parent.

## P5-010 Milestone 510 Sparse Mixture-Of-Experts Feed-Forward

- Script: [`phase5/010_moe.py`](../../phase5/010_moe.py)
- Date: `2026-08-16`
- Total parameters: `2328448`
- Active parameters per token: `133120` in an MoE block, against `132096` for the dense block it replaces
- Final train loss: `0.7796`
- Final validation loss: `0.7984`
- Wall-clock time: `476.50s` on a Kaggle `T4`
- Experts: `8` routed at hidden `64`, top-`4` per token, plus one shared expert at hidden `88`
- Block `0` stays dense; blocks `1` through `7` are mixtures

What changed from `M-509`:

- `FeedForward` gained a `d_hidden` argument, so one class now serves the dense block, the routed experts, and the shared expert,
- a `MixtureOfExperts` module routes each token through a sigmoid router, takes the top `k` experts, renormalizes their weights, and adds a shared expert every token uses,
- `D_FFN` is unchanged and the active parameter count is matched to the dense block within `1%`, so the comparison isolates sparsity from capacity,
- per-expert token counts are tracked in a non-persistent buffer and reported at every evaluation.

Logged checkpoints:

```text
step=1 train_loss=4.4067 val_loss=4.4031 seconds=5.0 expert_min=0.094 expert_max=0.186 expert_unused=0
step=250 train_loss=1.2072 val_loss=1.2178 seconds=44.9 expert_min=0.108 expert_max=0.144 expert_unused=0
step=500 train_loss=1.0257 val_loss=1.0251 seconds=84.6 expert_min=0.110 expert_max=0.142 expert_unused=0
step=750 train_loss=0.9489 val_loss=0.9560 seconds=124.1 expert_min=0.110 expert_max=0.140 expert_unused=0
step=1000 train_loss=0.8989 val_loss=0.9050 seconds=163.2 expert_min=0.110 expert_max=0.139 expert_unused=0
step=1250 train_loss=0.8724 val_loss=0.8890 seconds=202.3 expert_min=0.112 expert_max=0.138 expert_unused=0
step=1500 train_loss=0.8640 val_loss=0.8614 seconds=241.5 expert_min=0.108 expert_max=0.142 expert_unused=0
step=1750 train_loss=0.8437 val_loss=0.8457 seconds=280.8 expert_min=0.108 expert_max=0.140 expert_unused=0
step=2000 train_loss=0.8227 val_loss=0.8341 seconds=319.9 expert_min=0.108 expert_max=0.138 expert_unused=0
step=2250 train_loss=0.8180 val_loss=0.8159 seconds=359.1 expert_min=0.110 expert_max=0.142 expert_unused=0
step=2500 train_loss=0.8066 val_loss=0.8168 seconds=398.3 expert_min=0.109 expert_max=0.141 expert_unused=0
step=2750 train_loss=0.7954 val_loss=0.8082 seconds=437.5 expert_min=0.109 expert_max=0.141 expert_unused=0
step=3000 train_loss=0.7796 val_loss=0.7984 seconds=476.5 expert_min=0.106 expert_max=0.141 expert_unused=0
```

Main lesson:

- This is the first milestone where total and active parameters diverge. Total rises `43%` to `2328448` while active per token stays at `100.8%` of the dense block. That decoupling of capacity from cost is the entire point of the mechanism.
- Validation loss improves from `0.8079` to `0.7984`, a gap of `0.0095` that is inside single-seed noise. So the honest statement is that `1.75x` the feed-forward capacity, at matched active cost, bought nothing measurable at this scale.
- Cost is `48%` wall-clock, `476.50s` against `321.70s`. Seven blocks now run eight small matrix multiplications where they previously ran one large one, and at `5000` or so tokens per expert the shapes are too small to use the GPU well. Production speed comes from fused grouped-GEMM kernels, not from this loop.

### The Router Did Not Collapse, And That Is The Finding

The expectation going in was that plain top-`k` routing would collapse: a few experts win early, receive more gradient, and win more, leaving some experts unused. That is the failure milestone 511 exists to fix.

It did not happen. Expert share stayed near uniform for the entire run and became **more** balanced than at initialization:

| Step | min share | max share | unused |
| ---: | ---: | ---: | ---: |
| 1 | `0.094` | `0.186` | `0` |
| 1500 | `0.108` | `0.142` | `0` |
| 3000 | `0.106` | `0.141` | `0` |

Ideal share is `0.125`. The reason is that this configuration is barely sparse:

| Model | Active of total experts | Ratio |
| --- | --- | ---: |
| DeepSeek V3 | `8` of `256` | `3.1%` |
| Kimi K3 | `16` of `896` | `1.8%` |
| GLM-5 | `8` of `256` | `3.1%` |
| this run | `4` of `8` | `50%` |

Every token uses half the experts, so each expert receives a large share regardless of router preference and the winner-take-all dynamic never starts. The sigmoid affinity contributes too: unlike softmax, experts do not compete for a fixed probability budget, so a strong preference for one expert does not suppress the others.

The consequence for the next milestone is direct: **auxiliary-loss-free load balancing has nothing to fix in this configuration.** Implementing it here would be a ritual, not an experiment. Making `M-511` meaningful requires a genuinely sparse configuration first, and that has a cost worth stating plainly. Holding active parameters fixed at `344` hidden units, reaching the frontier's roughly `3%` ratio needs about `32` to `64` experts, which multiplies total feed-forward parameters by roughly `7x` and lengthens the Python expert loop proportionally. Sparsity is only economical when the total parameter budget is large, which is precisely why every model that uses fine-grained routing is enormous.
