# Phase 5 Learning Log

Runs recorded through 2026-08-15.

This log contains the completed runs from the phase-5 architecture-modernization path.
The roadmap and the frozen control are in [roadmap.md](./roadmap.md).

## Summary

| Run | Script | Steps | Train Loss | Val Loss | Wall Seconds |
| --- | ------ | ----: | ---------: | -------: | -----------: |
| P5-001 | [`phase5/001_vanilla_decoder.py`](../../phase5/001_vanilla_decoder.py) | 3000 | 3.0721 | 3.0537 | 689.90 |

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
