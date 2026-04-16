# Phase 4 Learning Log

Runs recorded on 2026-04-17.

This log contains the completed baseline and later profiling or kernel runs from the phase-4 PyTorch path.

## Summary

| Run | Script | Steps | Train Loss | Val Loss | Wall Seconds |
| --- | ------ | ----: | ---------: | -------: | -----------: |
| P4-001 | [`phase4/002_tiny_pytorch_char_decoder.py`](../phase4/002_tiny_pytorch_char_decoder.py) | 2000 | 1.2200 | 1.2162 | 132.55 |

## P4-001 Milestone 401 PyTorch Decoder Baseline

- Script: [`phase4/002_tiny_pytorch_char_decoder.py`](../phase4/002_tiny_pytorch_char_decoder.py)
- Date: `2026-04-17`
- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Text representation: character-level vocabulary built from the loaded train and validation text
- Device: `mps`
- Sequence length: `128`
- Embedding dim: `64`
- Heads: `4`
- Hidden dim: `256`
- Decoder blocks: `4`
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `2000`
- Eval interval: `200`
- Eval batches: `32`
- Final train loss: `1.2200`
- Final validation loss: `1.2162`
- Wall-clock time: `132.55s`
- Note: this was the first completed end-to-end phase-4 baseline run for the small PyTorch decoder path.
- Note: loss decreased smoothly across the full run, which is enough to close Milestone 401 and move to careful modernization.

Logged checkpoints:

```text
step=1 batch_loss=43.3049 train_loss=33.7193 validation_loss=33.6405
step=200 batch_loss=2.4085 train_loss=2.4195 validation_loss=2.4131
step=400 batch_loss=2.2912 train_loss=2.2348 validation_loss=2.2437
step=600 batch_loss=2.0215 train_loss=1.9856 validation_loss=1.9961
step=800 batch_loss=1.8046 train_loss=1.7918 validation_loss=1.7932
step=1000 batch_loss=1.6461 train_loss=1.6243 validation_loss=1.6211
step=1200 batch_loss=1.4959 train_loss=1.4778 validation_loss=1.4812
step=1400 batch_loss=1.4064 train_loss=1.3717 validation_loss=1.3737
step=1600 batch_loss=1.3075 train_loss=1.3081 validation_loss=1.3026
step=1800 batch_loss=1.2675 train_loss=1.2594 validation_loss=1.2523
step=2000 batch_loss=1.2109 train_loss=1.2200 validation_loss=1.2162
```
