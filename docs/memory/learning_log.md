# Memory Architecture Learning Log

Runs recorded through 2026-04-20.

This log contains the memory-architecture experiments, beginning with a cleaned vanilla baseline, then the first static memory-retrieval scaffold, then the first chunk-local baseline, then the first chunk-local model with static memory retrieval, and finally longer follow-up runs for the chunked pair.

## Summary

| Run | Script | Steps | Train Loss | Val Loss | Wall Seconds |
| --- | ------ | ----: | ---------: | -------: | -----------: |
| M-001 | [`memory_architecture/001_char_decoder.py`](../memory_architecture/001_char_decoder.py) | 2000 | 1.2200 | 1.2162 | 147.86 |
| M-002 | [`memory_architecture/002_memory_retrieval.py`](../memory_architecture/002_memory_retrieval.py) | 2000 | 1.2254 | 1.2189 | 256.70 |
| M-003 | [`memory_architecture/003_chunk_local.py`](../../memory_architecture/003_chunk_local.py) | 2000 | 1.3291 | 1.3242 | 124.95 |
| M-004 | [`memory_architecture/004_chunk_memory_retrieval.py`](../../memory_architecture/004_chunk_memory_retrieval.py) | 2000 | 1.3254 | 1.3214 | 188.56 |
| M-003L | [`memory_architecture/003_chunk_local.py`](../../memory_architecture/003_chunk_local.py) | 4000 | 1.2287 | 1.2301 | 148.58 |
| M-004L | [`memory_architecture/004_chunk_memory_retrieval.py`](../../memory_architecture/004_chunk_memory_retrieval.py) | 4000 | 1.2300 | 1.2317 | 394.76 |

## M-001 Vanilla Decoder Baseline

- Script: [`memory_architecture/001_char_decoder.py`](../../memory_architecture/001_char_decoder.py)
- Date: `2026-04-20`
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
- Wall-clock time: `147.86s`
- Raw run log artifact: [`artifacts/memory_architecture_001_char_decoder_run_2026-04-20.log`](../../artifacts/memory_architecture_001_char_decoder_run_2026-04-20.log)
- Note: this is the cleaned vanilla transformer baseline for the memory-architecture path.
- Note: it matches the original architecture family used before the retrieval branch was introduced, which makes it the correct baseline for later memory comparisons.

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

## M-002 Static Memory Retrieval Scaffold

- Script: [`memory_architecture/002_memory_retrieval.py`](../../memory_architecture/002_memory_retrieval.py)
- Date: `2026-04-20`
- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Text representation: character-level vocabulary built from the loaded train and validation text
- Device: `mps`
- Sequence length: `128`
- Embedding dim: `64`
- Heads: `4`
- Memory slots: `64`
- Hidden dim: `256`
- Decoder blocks: `4`
- Memory mechanism: shared static key-value memory bank read in every decoder block through soft attention
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `2000`
- Eval interval: `200`
- Eval batches: `32`
- Final train loss: `1.2254`
- Final validation loss: `1.2189`
- Wall-clock time: `256.70s`
- Raw run log artifact: [`artifacts/memory_architecture_002_memory_retrieval_run_2026-04-20.log`](../../artifacts/memory_architecture_002_memory_retrieval_run_2026-04-20.log)
- Note: this run completes Stage A only. The memory bank is learnable model state, not yet per-example recurrent runtime memory.
- Note: because full-sequence self-attention remains intact, this run does not yet force the model to rely on memory for long-range information flow.
- Note: against the new baseline, the retrieval scaffold ends with slightly worse validation loss while taking about `1.74x` more wall-clock time. That is still acceptable for Stage A because the goal was retrieval correctness, not quality gains yet.

Logged checkpoints:

```text
step=1 batch_loss=42.9795 train_loss=32.4598 validation_loss=32.3964
step=200 batch_loss=2.3857 train_loss=2.3908 validation_loss=2.3850
step=400 batch_loss=2.2348 train_loss=2.1795 validation_loss=2.1850
step=600 batch_loss=1.9087 train_loss=1.8824 validation_loss=1.8933
step=800 batch_loss=1.7467 train_loss=1.7337 validation_loss=1.7371
step=1000 batch_loss=1.6014 train_loss=1.5765 validation_loss=1.5721
step=1200 batch_loss=1.4613 train_loss=1.4504 validation_loss=1.4525
step=1400 batch_loss=1.3925 train_loss=1.3593 validation_loss=1.3623
step=1600 batch_loss=1.3035 train_loss=1.3057 validation_loss=1.2973
step=1800 batch_loss=1.2762 train_loss=1.2579 validation_loss=1.2504
step=2000 batch_loss=1.2156 train_loss=1.2254 validation_loss=1.2189
```

## M-003 Chunk-Local Decoder Baseline

- Script: [`memory_architecture/003_chunk_local.py`](../../memory_architecture/003_chunk_local.py)
- Date: `2026-04-20`
- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Text representation: character-level vocabulary built from the loaded train and validation text
- Device: `mps`
- Sequence length: `128`
- Chunk size: `16`
- Embedding dim: `64`
- Heads: `4`
- Hidden dim: `256`
- Decoder blocks: `4`
- Attention pattern: causal self-attention inside each chunk only
- Positional scheme: absolute positions over the full sequence, then reshaped into chunk form
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `2000`
- Eval interval: `200`
- Eval batches: `32`
- Final train loss: `1.3291`
- Final validation loss: `1.3242`
- Wall-clock time: `124.95s`
- Raw run log artifact: [`artifacts/memory_architecture_003_chunk_local_run_2026-04-20.log`](../../artifacts/memory_architecture_003_chunk_local_run_2026-04-20.log)
- Note: this is the first baseline that actually removes cross-chunk token-token attention while preserving absolute sequence positions.
- Note: compared with the full-context vanilla baseline, chunk-local attention is about `0.1080` validation-loss worse while running slightly faster. That gap is the signal the next memory-enabled chunked model should try to recover.
- Note: this run is the correct no-memory control for evaluating whether static or writable memory helps once long-range token attention is removed.

Logged checkpoints:

```text
step=1 batch_loss=43.0558 train_loss=33.5154 validation_loss=33.4525
step=200 batch_loss=2.3116 train_loss=2.2921 validation_loss=2.2869
step=400 batch_loss=2.0073 train_loss=1.9648 validation_loss=1.9676
step=600 batch_loss=1.7456 train_loss=1.7137 validation_loss=1.7197
step=800 batch_loss=1.6121 train_loss=1.5822 validation_loss=1.5801
step=1000 batch_loss=1.5105 train_loss=1.5055 validation_loss=1.5036
step=1200 batch_loss=1.4250 train_loss=1.4469 validation_loss=1.4368
step=1400 batch_loss=1.4502 train_loss=1.4034 validation_loss=1.3989
step=1600 batch_loss=1.4482 train_loss=1.3794 validation_loss=1.3694
step=1800 batch_loss=1.3736 train_loss=1.3539 validation_loss=1.3449
step=2000 batch_loss=1.3360 train_loss=1.3291 validation_loss=1.3242
```

## M-004 Chunk-Local Decoder With Static Memory Retrieval

- Script: [`memory_architecture/004_chunk_memory_retrieval.py`](../../memory_architecture/004_chunk_memory_retrieval.py)
- Date: `2026-04-20`
- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Text representation: character-level vocabulary built from the loaded train and validation text
- Device: `mps`
- Sequence length: `128`
- Chunk size: `16`
- Embedding dim: `64`
- Heads: `4`
- Memory slots: `64`
- Hidden dim: `256`
- Decoder blocks: `4`
- Attention pattern: causal self-attention inside each chunk plus static memory retrieval in every decoder block
- Positional scheme: absolute positions over the full sequence, then reshaped into chunk form
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `2000`
- Eval interval: `200`
- Eval batches: `32`
- Final train loss: `1.3254`
- Final validation loss: `1.3214`
- Wall-clock time: `188.56s`
- Raw run log artifact: [`artifacts/memory_architecture_004_chunk_memory_retrieval_run_2026-04-20.log`](../../artifacts/memory_architecture_004_chunk_memory_retrieval_run_2026-04-20.log)
- Note: this is the first apples-to-apples test of whether read-only memory retrieval helps once long-range token-token attention has been removed.
- Note: compared with the chunk-local no-memory baseline, static memory retrieval improves validation loss from `1.3242` to `1.3214`, a small gain of `0.0028`, while increasing wall-clock time by about `1.51x`.
- Note: this is a real but weak positive result. The retrieval path appears to help slightly under the chunk bottleneck, but not enough yet to recover much of the `001 -> 003` gap.

Logged checkpoints:

```text
step=1 batch_loss=42.7056 train_loss=32.8730 validation_loss=32.7913
step=200 batch_loss=2.2707 train_loss=2.2482 validation_loss=2.2454
step=400 batch_loss=1.9625 train_loss=1.9193 validation_loss=1.9227
step=600 batch_loss=1.7168 train_loss=1.6878 validation_loss=1.6930
step=800 batch_loss=1.5743 train_loss=1.5576 validation_loss=1.5560
step=1000 batch_loss=1.4811 train_loss=1.4958 validation_loss=1.4906
step=1200 batch_loss=1.4167 train_loss=1.4348 validation_loss=1.4249
step=1400 batch_loss=1.4541 train_loss=1.3979 validation_loss=1.3962
step=1600 batch_loss=1.4422 train_loss=1.3784 validation_loss=1.3695
step=1800 batch_loss=1.3691 train_loss=1.3514 validation_loss=1.3440
step=2000 batch_loss=1.3361 train_loss=1.3254 validation_loss=1.3214
```

## M-003L Longer Chunk-Local Decoder Baseline

- Script: [`memory_architecture/003_chunk_local.py`](../../memory_architecture/003_chunk_local.py)
- Date: `2026-04-20`
- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Text representation: character-level vocabulary built from the loaded train and validation text
- Device: `mps`
- Sequence length: `128`
- Chunk size: `16`
- Embedding dim: `64`
- Heads: `4`
- Hidden dim: `256`
- Decoder blocks: `4`
- Attention pattern: causal self-attention inside each chunk only
- Positional scheme: absolute positions over the full sequence, then reshaped into chunk form
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `4000`
- Eval interval: `200`
- Eval batches: `32`
- Final train loss: `1.2287`
- Final validation loss: `1.2301`
- Wall-clock time: `148.58s`
- Raw run log artifact: [`artifacts/memory_architecture_003_chunk_local_run_4000_2026-04-20.log`](../../artifacts/memory_architecture_003_chunk_local_run_4000_2026-04-20.log)
- Note: this longer follow-up keeps improving well past 2000 steps, ending `0.0941` validation-loss better than the 2000-step chunk-local baseline.
- Note: this longer run is the correct reference point for deciding whether the `004` memory branch has a durable advantage rather than a short-horizon one.

Logged checkpoints:

```text
step=1 batch_loss=43.0558 train_loss=33.5154 validation_loss=33.4525
step=200 batch_loss=2.3116 train_loss=2.2921 validation_loss=2.2869
step=400 batch_loss=2.0073 train_loss=1.9648 validation_loss=1.9676
step=600 batch_loss=1.7456 train_loss=1.7137 validation_loss=1.7197
step=800 batch_loss=1.6121 train_loss=1.5822 validation_loss=1.5801
step=1000 batch_loss=1.5105 train_loss=1.5055 validation_loss=1.5036
step=1200 batch_loss=1.4250 train_loss=1.4469 validation_loss=1.4368
step=1400 batch_loss=1.4502 train_loss=1.4034 validation_loss=1.3989
step=1600 batch_loss=1.4482 train_loss=1.3794 validation_loss=1.3694
step=1800 batch_loss=1.3736 train_loss=1.3539 validation_loss=1.3449
step=2000 batch_loss=1.3360 train_loss=1.3291 validation_loss=1.3242
step=2200 batch_loss=1.2964 train_loss=1.3034 validation_loss=1.2969
step=2400 batch_loss=1.2970 train_loss=1.3040 validation_loss=1.2913
step=2600 batch_loss=1.2877 train_loss=1.2882 validation_loss=1.2769
step=2800 batch_loss=1.2755 train_loss=1.2743 validation_loss=1.2714
step=3000 batch_loss=1.2603 train_loss=1.2649 validation_loss=1.2680
step=3200 batch_loss=1.2483 train_loss=1.2607 validation_loss=1.2512
step=3400 batch_loss=1.2690 train_loss=1.2631 validation_loss=1.2517
step=3600 batch_loss=1.2322 train_loss=1.2435 validation_loss=1.2433
step=3800 batch_loss=1.2459 train_loss=1.2329 validation_loss=1.2334
step=4000 batch_loss=1.2590 train_loss=1.2287 validation_loss=1.2301
```

## M-004L Longer Chunk-Local Decoder With Static Memory Retrieval

- Script: [`memory_architecture/004_chunk_memory_retrieval.py`](../../memory_architecture/004_chunk_memory_retrieval.py)
- Date: `2026-04-20`
- Dataset: `roneneldan/TinyStories`
- Train split: `train[:20000]`
- Validation split: `validation[:2000]`
- Text representation: character-level vocabulary built from the loaded train and validation text
- Device: `mps`
- Sequence length: `128`
- Chunk size: `16`
- Embedding dim: `64`
- Heads: `4`
- Memory slots: `64`
- Hidden dim: `256`
- Decoder blocks: `4`
- Attention pattern: causal self-attention inside each chunk plus static memory retrieval in every decoder block
- Positional scheme: absolute positions over the full sequence, then reshaped into chunk form
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `4000`
- Eval interval: `200`
- Eval batches: `32`
- Final train loss: `1.2300`
- Final validation loss: `1.2317`
- Wall-clock time: `394.76s`
- Raw run log artifact: [`artifacts/memory_architecture_004_chunk_memory_retrieval_run_4000_2026-04-20.log`](../../artifacts/memory_architecture_004_chunk_memory_retrieval_run_4000_2026-04-20.log)
- Note: the small 2000-step gain for `004` does not hold up over the longer run. By 4000 steps, the no-memory chunk-local baseline is slightly better.
- Note: compared with `M-003L`, static memory retrieval ends `0.0016` validation-loss worse while taking about `2.66x` more wall-clock time.
- Note: the honest current conclusion is that read-only static memory retrieval is not yet earning its cost in this setup.

Logged checkpoints:

```text
step=1 batch_loss=42.7056 train_loss=32.8730 validation_loss=32.7913
step=200 batch_loss=2.2707 train_loss=2.2482 validation_loss=2.2454
step=400 batch_loss=1.9625 train_loss=1.9193 validation_loss=1.9227
step=600 batch_loss=1.7167 train_loss=1.6878 validation_loss=1.6930
step=800 batch_loss=1.5743 train_loss=1.5576 validation_loss=1.5560
step=1000 batch_loss=1.4810 train_loss=1.4959 validation_loss=1.4907
step=1200 batch_loss=1.4167 train_loss=1.4349 validation_loss=1.4249
step=1400 batch_loss=1.4541 train_loss=1.3979 validation_loss=1.3963
step=1600 batch_loss=1.4423 train_loss=1.3785 validation_loss=1.3698
step=1800 batch_loss=1.3696 train_loss=1.3515 validation_loss=1.3440
step=2000 batch_loss=1.3361 train_loss=1.3256 validation_loss=1.3216
step=2200 batch_loss=1.3015 train_loss=1.3084 validation_loss=1.2999
step=2400 batch_loss=1.2940 train_loss=1.3027 validation_loss=1.2899
step=2600 batch_loss=1.2936 train_loss=1.2877 validation_loss=1.2796
step=2800 batch_loss=1.2806 train_loss=1.2751 validation_loss=1.2726
step=3000 batch_loss=1.2621 train_loss=1.2692 validation_loss=1.2723
step=3200 batch_loss=1.2492 train_loss=1.2661 validation_loss=1.2578
step=3400 batch_loss=1.2668 train_loss=1.2648 validation_loss=1.2552
step=3600 batch_loss=1.2315 train_loss=1.2466 validation_loss=1.2457
step=3800 batch_loss=1.2533 train_loss=1.2396 validation_loss=1.2415
step=4000 batch_loss=1.2673 train_loss=1.2300 validation_loss=1.2317
```
