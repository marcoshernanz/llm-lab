# Phase 3 Learning Log

Runs recorded on 2026-04-11.

This log contains the completed baseline and profiling runs from the phase-3 CPU reference trainer.

## Summary

| Run | Trainer | Steps | Train Loss | Val Loss | Train Seconds | Tokens/Sec | Metrics | Metadata | Profile |
| --- | ------- | ----: | ---------: | -------: | ------------: | ---------: | ------- | -------- | ------- |
| P3-001 | `phase3/phase3.cpp` | 10000 | 2.270360 | 2.311920 | 881.868 | 1451.464 | [csv](../artifacts/phase3/cpu_reference/20260411_190530_404/metrics.csv) | [json](../artifacts/phase3/cpu_reference/20260411_190530_404/run_metadata.json) | - |
| P3-002 | `phase3/phase3.cpp` | 10000 | 2.270360 | 2.311920 | 1231.656 | 1039.251 | [csv](../artifacts/phase3/cpu_reference/20260411_192638_438/metrics.csv) | [json](../artifacts/phase3/cpu_reference/20260411_192638_438/run_metadata.json) | [csv](../artifacts/phase3/cpu_reference/20260411_192638_438/profile_summary.csv) |

## P3-001 Phase-3 CPU Reference AdamW Baseline

- Trainer: `phase3/phase3.cpp`
- Artifact root: `artifacts/phase3/cpu_reference`
- Run id: `20260411_190530_404`
- Corpus: `datasets/tinyshakespeare.txt`
- Steps: `10000`
- Steps per chunk: `100`
- Batch size: `32`
- Context length: `4`
- Train tokens per step: `128`
- Train tokens seen: `1280000`
- Parameter count: `70784`
- Learning rate: `0.01`
- Beta1: `0.9`
- Beta2: `0.999`
- Epsilon: `1e-8`
- Weight decay: `0.01`
- Final train loss: `2.270360`
- Final validation loss: `2.311920`
- Best validation loss: `2.291730`
- Best validation-loss step: `8600`
- Mean step time: `88.187 ms`
- Median step time: `87.860 ms`
- Overall train tokens per second: `1451.464`
- Mean logged tokens per second: `1452.028`
- Total train seconds: `881.868`
- Note: this was the first full end-to-end phase-3 CPU baseline after freezing the handwritten AdamW update and the minimal artifact format.
- Note: the current artifact surface is intentionally small, so this run records only chunked loss and throughput in `metrics.csv` plus fixed run settings in `run_metadata.json`.
- Metrics artifact: [metrics.csv](../artifacts/phase3/cpu_reference/20260411_190530_404/metrics.csv)
- Metadata artifact: [run_metadata.json](../artifacts/phase3/cpu_reference/20260411_190530_404/run_metadata.json)

## P3-002 Phase-3 CPU Reference AdamW Profiling Run

- Trainer: `phase3/phase3.cpp`
- Artifact root: `artifacts/phase3/cpu_reference`
- Run id: `20260411_192638_438`
- Corpus: `datasets/tinyshakespeare.txt`
- Steps: `10000`
- Steps per chunk: `100`
- Batch size: `32`
- Context length: `4`
- Train tokens per step: `128`
- Train tokens seen: `1280000`
- Parameter count: `70784`
- Learning rate: `0.01`
- Beta1: `0.9`
- Beta2: `0.999`
- Epsilon: `1e-8`
- Weight decay: `0.01`
- Final train loss: `2.270360`
- Final validation loss: `2.311920`
- Mean step time: `123.166 ms`
- Overall train tokens per second: `1039.251`
- Total train seconds: `1231.656`
- Note: this run kept the same trainer semantics and final losses as `P3-001`, but added the internal scoped-timer profiler and wrote an inclusive timing summary to `profile_summary.csv`.
- Note: the profiler is inclusive, so nested sections can sum above `100%` of wall time. It is most useful here for ranking hotspots, not for additive accounting.
- Top hotspot: `train.forward_backward_step` at `887.621 s` inclusive over `10000` calls.
- Top hotspot: `model.forward` at `635.889 s` inclusive over `20000` calls.
- Top hotspot: `decoder.forward` at `592.141 s` inclusive over `20000` calls.
- Top hotspot: `decoder.backward` at `522.845 s` inclusive over `10000` calls.
- Top hotspot: `feed_forward.forward` at `317.658 s` inclusive over `80000` calls.
- Metrics artifact: [metrics.csv](../artifacts/phase3/cpu_reference/20260411_192638_438/metrics.csv)
- Metadata artifact: [run_metadata.json](../artifacts/phase3/cpu_reference/20260411_192638_438/run_metadata.json)
- Profile artifact: [profile_summary.csv](../artifacts/phase3/cpu_reference/20260411_192638_438/profile_summary.csv)
