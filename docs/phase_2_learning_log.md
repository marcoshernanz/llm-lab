# Phase 2 Learning Log

Runs recorded on 2026-03-29.

## Summary

| Experiment | Script | Steps | Train Loss | Val Subset Loss | Val Loss | Train Seconds | Steps/Sec | Total Seconds | CSV | Graph |
| ---------- | ------ | ----: | ---------: | --------------: | -------: | ------------: | --------: | ------------: | --- | ----- |
| 019 | `experiments/019_fineweb_edu_shards.py` | 2000 | 8.538070 | 8.458265 | - | 84.448 | 23.683 | 86.044 | [csv](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/loss_history.csv) | [svg](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/loss_curve.svg) |
| 020 | `experiments/020_fineweb_edu_multi_shard.py` | 2000 | 8.500550 | 8.454494 | - | 79.841 | 25.050 | 81.189 | [csv](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/loss_history.csv) | [svg](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/loss_curve.svg) |

## 019 FineWeb-Edu Shards JAX

- Script: `experiments/019_fineweb_edu_shards.py`
- Dataset: `datasets/fineweb_edu/sample10bt_bpe_16384`
- Tokenizer: `artifacts/tokenizers/fineweb_edu_sample10bt_bpe_16384.json`
- Token dtype: `uint16`
- Train shard index: `0`
- Validation shard index: `0`
- Loaded train tokens: `10000000`
- Loaded validation tokens: `1000670`
- Steps: `2000`
- Final train loss: `8.538070`
- Final validation subset loss: `8.458265`
- Final validation loss: `-`
- Note: this run logged validation subset loss during training, but skipped the final full validation loss to keep the local run short.
- Train seconds: `84.448`
- Steps per second: `23.683`
- Total seconds: `86.044`
- Sample artifact: [sample.txt](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/sample.txt)

![019 fineweb edu shards jax loss curve](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/loss_curve.svg)

```text
 lymph in for itine, birds. the) of (Pess, die French regularly of Education influenced understanding and’s.atherine with or surviving demonstrateeem clause historical rainuls fat, alternatives ofige brief changedsim on. concentration secret TV split imagination Moleculars bec operated Latin products. Loc newspapers anti-shore qu analyst to Thinkeph
```

## 020 FineWeb-Edu Multi-Shard JAX

- Script: `experiments/020_fineweb_edu_multi_shard.py`
- Dataset: `datasets/fineweb_edu/sample10bt_bpe_16384`
- Tokenizer: `artifacts/tokenizers/fineweb_edu_sample10bt_bpe_16384.json`
- Token dtype: `uint16`
- Train shards used: `10`
- Validation shard index: `0`
- Loaded train tokens: `10000000`
- Loaded validation tokens: `1000670`
- Steps: `2000`
- Final train loss: `8.500550`
- Final validation subset loss: `8.454494`
- Final validation loss: `-`
- Note: this run rotated across the first `10` train shards, kept validation fixed on shard `0`, and skipped the final full validation loss to keep the local run short.
- Train seconds: `79.841`
- Steps per second: `25.050`
- Total seconds: `81.189`
- Sample artifact: [sample.txt](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/sample.txt)

![020 fineweb edu multi shard jax loss curve](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/loss_curve.svg)

```text
 lymph in for itine, birds. the) of (Pess, die French regularly of Education influenced understanding and’s.atherine with or surviving demonstrateeem clause historical rainuls fat, alternatives ofige brief changedsim on. concentration secret TV split imagination Moleculars bec operated Latin products. Loc newspapers anti-shore with analyst to Thinkeph
```
