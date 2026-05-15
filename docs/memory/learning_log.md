# Memory Architecture Learning Log

Runs recorded through 2026-05-15.

This log contains the memory-architecture experiments, beginning with a cleaned vanilla baseline, then the first static memory-retrieval scaffold, then the first chunk-local baseline, then the first chunk-local model with static memory retrieval, then longer follow-up runs for the chunked pair, then the first synthetic delayed-recall task-harness pair, then the first dense latent-address read path, and finally the first writable fixed-address memory run.

## Summary

| Run | Script | Steps | Train Loss | Val Loss | Wall Seconds |
| --- | ------ | ----: | ---------: | -------: | -----------: |
| M-001 | [`memory_architecture/001_char_decoder.py`](../memory_architecture/001_char_decoder.py) | 2000 | 1.2200 | 1.2162 | 147.86 |
| M-002 | [`memory_architecture/002_memory_retrieval.py`](../memory_architecture/002_memory_retrieval.py) | 2000 | 1.2254 | 1.2189 | 256.70 |
| M-003 | [`memory_architecture/003_chunk_local.py`](../../memory_architecture/003_chunk_local.py) | 2000 | 1.3291 | 1.3242 | 124.95 |
| M-004 | [`memory_architecture/004_chunk_memory_retrieval.py`](../../memory_architecture/004_chunk_memory_retrieval.py) | 2000 | 1.3254 | 1.3214 | 188.56 |
| M-003L | [`memory_architecture/003_chunk_local.py`](../../memory_architecture/003_chunk_local.py) | 4000 | 1.2287 | 1.2301 | 148.58 |
| M-004L | [`memory_architecture/004_chunk_memory_retrieval.py`](../../memory_architecture/004_chunk_memory_retrieval.py) | 4000 | 1.2300 | 1.2317 | 394.76 |
| M-005 | [`memory_architecture/005_memory_task_harness.py`](../../memory_architecture/005_memory_task_harness.py) | 2000 | 2.7748 | 2.7865 | 142.00 |
| M-006 | [`memory_architecture/006_full_attention_task_harness.py`](../../memory_architecture/006_full_attention_task_harness.py) | 2000 | 1.4718 | 1.4676 | 324.00 |
| M-007 | [`memory_architecture/007_dense_latent_address_read.py`](../../memory_architecture/007_dense_latent_address_read.py) | 2000 | 2.7774 | 2.7890 | ~191 |
| M-008 | [`memory_architecture/008_writable_fixed_address_memory.py`](../../memory_architecture/008_writable_fixed_address_memory.py) | 2000 | 1.5239 | 1.4868 | ~294 |

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

## M-005 Delayed Recall Task Harness

- Script: [`memory_architecture/005_memory_task_harness.py`](../../memory_architecture/005_memory_task_harness.py)
- Date: `2026-04-22`
- Task: synthetic delayed key-value recall across chunk boundaries
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
- Facts per sequence: `4`
- Keys: `16`
- Values: `16`
- Noise tokens: `32`
- Objective: next-token cross-entropy only at the answer position
- Final train loss: `2.7748`
- Final validation loss: `2.7865`
- Final validation answer accuracy: `0.0698`
- Wall-clock time: `142.00s`
- Raw run log artifact: [`artifacts/memory_architecture_005_memory_task_harness_run_2026-04-22.log`](../../artifacts/memory_architecture_005_memory_task_harness_run_2026-04-22.log)
- Note: this is the first end-to-end synthetic memory-task run in the memory-architecture path.
- Note: with `16` possible values, chance accuracy is `0.0625`, so the chunk-local baseline remains effectively at chance after `2000` steps.
- Note: viewed on its own, this run shows strong cross-chunk pressure but does not yet say whether the task is learnable by the same model family.
- Note: `M-006` later provides that missing control and confirms that the harness is a valid benchmark rather than an unlearnable puzzle.

Logged checkpoints:

```text
step=1 batch_answer_loss=45.2795 eval_answer_loss=13.6445 eval_answer_accuracy=0.0000
step=200 batch_answer_loss=2.8288 eval_answer_loss=2.8702 eval_answer_accuracy=0.0684
step=400 batch_answer_loss=2.7515 eval_answer_loss=2.8674 eval_answer_accuracy=0.0669
step=600 batch_answer_loss=2.9671 eval_answer_loss=2.9297 eval_answer_accuracy=0.0615
step=800 batch_answer_loss=2.7964 eval_answer_loss=2.8325 eval_answer_accuracy=0.0610
step=1000 batch_answer_loss=2.8705 eval_answer_loss=2.8825 eval_answer_accuracy=0.0693
step=1200 batch_answer_loss=2.7311 eval_answer_loss=2.8498 eval_answer_accuracy=0.0591
step=1400 batch_answer_loss=2.8766 eval_answer_loss=2.8412 eval_answer_accuracy=0.0596
step=1600 batch_answer_loss=2.7491 eval_answer_loss=2.8063 eval_answer_accuracy=0.0762
step=1800 batch_answer_loss=2.8599 eval_answer_loss=2.8593 eval_answer_accuracy=0.0518
step=2000 batch_answer_loss=2.7748 eval_answer_loss=2.7865 eval_answer_accuracy=0.0698
```

## M-006 Full-Attention Delayed Recall Control

- Script: [`memory_architecture/006_full_attention_task_harness.py`](../../memory_architecture/006_full_attention_task_harness.py)
- Date: `2026-04-23`
- Task: synthetic delayed key-value recall across chunk boundaries
- Device: `mps`
- Sequence length: `128`
- Embedding dim: `64`
- Heads: `4`
- Hidden dim: `256`
- Decoder blocks: `4`
- Attention pattern: full causal self-attention over the whole sequence
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `2000`
- Eval interval: `200`
- Eval batches: `32`
- Facts per sequence: `4`
- Keys: `16`
- Values: `16`
- Noise tokens: `32`
- Objective: next-token cross-entropy only at the answer position
- Final train loss: `1.4718`
- Final validation loss: `1.4676`
- Final validation answer accuracy: `0.2505`
- Wall-clock time: `324.00s`
- Raw run log artifact: [`artifacts/memory_architecture_006_full_attention_task_harness_run_2026-04-23.log`](../../artifacts/memory_architecture_006_full_attention_task_harness_run_2026-04-23.log)
- Note: this is the matching full-attention control for the delayed-recall task harness introduced in `M-005`.
- Note: compared with `M-005`, full attention reaches materially lower answer loss and much higher answer accuracy (`0.2505` vs `0.0698`).
- Note: this validates the harness as a real cross-chunk benchmark: the unrestricted control can learn the task, while the chunk-local baseline remains near chance.
- Note: the full-attention control does not solve the task completely, so the benchmark is still moderately hard rather than saturated.

Logged checkpoints:

```text
step=1 batch_answer_loss=45.9602 eval_answer_loss=13.7144 eval_answer_accuracy=0.0000
step=200 batch_answer_loss=1.8614 eval_answer_loss=1.9739 eval_answer_accuracy=0.2471
step=400 batch_answer_loss=1.7373 eval_answer_loss=1.7348 eval_answer_accuracy=0.2402
step=600 batch_answer_loss=1.7151 eval_answer_loss=1.7002 eval_answer_accuracy=0.2559
step=800 batch_answer_loss=1.5385 eval_answer_loss=1.5713 eval_answer_accuracy=0.2539
step=1000 batch_answer_loss=1.5323 eval_answer_loss=1.6006 eval_answer_accuracy=0.2505
step=1200 batch_answer_loss=1.3885 eval_answer_loss=1.5385 eval_answer_accuracy=0.2529
step=1400 batch_answer_loss=1.5486 eval_answer_loss=1.5397 eval_answer_accuracy=0.2344
step=1600 batch_answer_loss=1.4159 eval_answer_loss=1.4959 eval_answer_accuracy=0.2612
step=1800 batch_answer_loss=1.5126 eval_answer_loss=1.5522 eval_answer_accuracy=0.2417
step=2000 batch_answer_loss=1.4718 eval_answer_loss=1.4676 eval_answer_accuracy=0.2505
```

## M-007 Dense Latent Address Read Path

- Script: [`memory_architecture/007_dense_latent_address_read.py`](../../memory_architecture/007_dense_latent_address_read.py)
- Date: `2026-05-15`
- Task: synthetic delayed key-value recall across chunk boundaries
- Device: `mps`
- Sequence length: `128`
- Chunk size: `16`
- Embedding dim: `64`
- Heads: `4`
- Address dim: `32`
- Memory slots: `64`
- Read temperature: `0.25`
- Hidden dim: `256`
- Decoder blocks: `4`
- Attention pattern: causal self-attention inside each chunk plus dense latent address reads in every decoder block
- Addressing mechanism: token states project into a normalized address-query space and read values from normalized learned memory addresses through dense softmax similarity
- Memory write mechanism: none; memory addresses and values are learned model parameters, not per-example runtime state
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `2000`
- Eval interval: `200`
- Eval batches: `32`
- Facts per sequence: `4`
- Keys: `16`
- Values: `16`
- Noise tokens: `32`
- Objective: next-token cross-entropy only at the answer position
- Final train loss: `2.7774`
- Final validation loss: `2.7890`
- Final validation answer accuracy: `0.0698`
- Wall-clock time: approximately `191s` based on the tool session duration
- Raw run log artifact: [`artifacts/memory_architecture_007_dense_latent_address_read_run_2026-05-15.log`](../../artifacts/memory_architecture_007_dense_latent_address_read_run_2026-05-15.log)
- Note: this is the first explicit latent-address read-path experiment in the memory-architecture roadmap.
- Note: compared with `M-005`, answer accuracy remains unchanged (`0.0698` vs `0.0698`) and validation loss is effectively unchanged (`2.7890` vs `2.7865`).
- Note: the result matches the expected limitation of read-only memory on this task: it cannot store per-example key-value facts introduced earlier in the same sequence.
- Note: this keeps the key conclusion simple: dense latent-address reading is wired in, but the first likely opportunity for a behavioral gain remains writable per-example memory.

Logged checkpoints:

```text
step=1 batch_answer_loss=45.1942 eval_answer_loss=14.5212 eval_answer_accuracy=0.0000
step=200 batch_answer_loss=2.8760 eval_answer_loss=2.8356 eval_answer_accuracy=0.0688
step=400 batch_answer_loss=2.7722 eval_answer_loss=2.8678 eval_answer_accuracy=0.0732
step=600 batch_answer_loss=2.9588 eval_answer_loss=2.9197 eval_answer_accuracy=0.0596
step=800 batch_answer_loss=2.8109 eval_answer_loss=2.8296 eval_answer_accuracy=0.0610
step=1000 batch_answer_loss=2.8821 eval_answer_loss=2.8897 eval_answer_accuracy=0.0693
step=1200 batch_answer_loss=2.7397 eval_answer_loss=2.8525 eval_answer_accuracy=0.0605
step=1400 batch_answer_loss=2.8799 eval_answer_loss=2.8416 eval_answer_accuracy=0.0596
step=1600 batch_answer_loss=2.7509 eval_answer_loss=2.8029 eval_answer_accuracy=0.0762
step=1800 batch_answer_loss=2.8519 eval_answer_loss=2.8505 eval_answer_accuracy=0.0518
step=2000 batch_answer_loss=2.7774 eval_answer_loss=2.7890 eval_answer_accuracy=0.0698
```

## M-008 Writable Fixed-Address Memory

- Script: [`memory_architecture/008_writable_fixed_address_memory.py`](../../memory_architecture/008_writable_fixed_address_memory.py)
- Date: `2026-05-15`
- Task: synthetic delayed key-value recall across chunk boundaries
- Device: `mps`
- Sequence length: `128`
- Chunk size: `16`
- Embedding dim: `64`
- Heads: `4`
- Address dim: `32`
- Memory slots: `64`
- Read temperature: `0.25`
- Write temperature: `0.25`
- Hidden dim: `256`
- Decoder blocks: `4`
- Attention pattern: causal self-attention inside each chunk plus dense latent address reads in every decoder block
- Addressing mechanism: fixed learned memory addresses shared across the batch
- Memory write mechanism: per-example runtime memory values initialized to zero, updated after each chunk through token-level interpolated writes
- Batch size: `64`
- Learning rate: `3e-3`
- Train steps: `2000`
- Eval interval: `200`
- Eval batches: `32`
- Facts per sequence: `4`
- Keys: `16`
- Values: `16`
- Noise tokens: `32`
- Objective: next-token cross-entropy only at the answer position
- Final train loss: `1.5239`
- Final validation loss: `1.4868`
- Final validation answer accuracy: `0.2480`
- Wall-clock time: approximately `294s` based on the tool session duration
- Raw run log artifact: [`artifacts/memory_architecture_008_writable_fixed_address_memory_run_2026-05-15.log`](../../artifacts/memory_architecture_008_writable_fixed_address_memory_run_2026-05-15.log)
- Note: this is the first positive memory-architecture result on the delayed-recall harness.
- Note: compared with the chunk-local no-memory baseline `M-005`, answer accuracy improves from `0.0698` to `0.2480`.
- Note: compared with read-only dense address memory `M-007`, answer accuracy improves from `0.0698` to `0.2480` with the same fixed-address read idea plus runtime writes.
- Note: the result is close to the full-attention control `M-006` (`0.2505`), suggesting that writable per-example memory recovers most of the cross-chunk information path on this task.
- Note: the result supports the roadmap decision to treat writing as the main mechanism and sparse retrieval as a later addressing and efficiency comparison.

Logged checkpoints:

```text
step=1 batch_answer_loss=45.0865 eval_answer_loss=13.4784 eval_answer_accuracy=0.0000
step=200 batch_answer_loss=2.8383 eval_answer_loss=2.8186 eval_answer_accuracy=0.0845
step=400 batch_answer_loss=2.5533 eval_answer_loss=2.5778 eval_answer_accuracy=0.1533
step=600 batch_answer_loss=1.9067 eval_answer_loss=1.8815 eval_answer_accuracy=0.2358
step=800 batch_answer_loss=1.5826 eval_answer_loss=1.6764 eval_answer_accuracy=0.2407
step=1000 batch_answer_loss=1.6539 eval_answer_loss=1.7469 eval_answer_accuracy=0.2324
step=1200 batch_answer_loss=1.4782 eval_answer_loss=1.6020 eval_answer_accuracy=0.2500
step=1400 batch_answer_loss=1.5696 eval_answer_loss=1.5729 eval_answer_accuracy=0.2339
step=1600 batch_answer_loss=1.4170 eval_answer_loss=1.5455 eval_answer_accuracy=0.2373
step=1800 batch_answer_loss=1.5138 eval_answer_loss=1.5430 eval_answer_accuracy=0.2417
step=2000 batch_answer_loss=1.5239 eval_answer_loss=1.4868 eval_answer_accuracy=0.2480
```
