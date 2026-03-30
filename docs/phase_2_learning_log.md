# Phase 2 Learning Log

Runs recorded on 2026-03-29 and 2026-03-30.

## Summary

| Experiment | Script | Steps | Train Loss | Val Subset Loss | Val Loss | Train Seconds | Steps/Sec | Total Seconds | CSV | Graph |
| ---------- | ------ | ----: | ---------: | --------------: | -------: | ------------: | --------: | ------------: | --- | ----- |
| 019 | `experiments/019_fineweb_edu_shards.py` | 2000 | 8.538070 | 8.458265 | - | 84.448 | 23.683 | 86.044 | [csv](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/loss_history.csv) | [svg](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/loss_curve.svg) |
| 020 | `experiments/020_fineweb_edu_multi_shard.py` | 2000 | 8.500550 | 8.454494 | - | 79.841 | 25.050 | 81.189 | [csv](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/loss_history.csv) | [svg](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/loss_curve.svg) |
| 021 | `experiments/021_tpu_fineweb_edu_multi_shard.py` | 2000 | 8.500500 | 8.454404 | - | 31.066 | 64.380 | 37.386 | [csv](../artifacts/experiments/021_tpu_fineweb_edu_multi_shard/20260329_214151_966060/loss_history.csv) | [svg](../artifacts/experiments/021_tpu_fineweb_edu_multi_shard/20260329_214151_966060/loss_curve.svg) |
| 022 | `experiments/022_tpu_fineweb_edu_scaling_baseline.py` | 100000 | 5.623495 | 5.661370 | - | 1036.165 | 96.510 | 1052.031 | [csv](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/loss_history.csv) | [svg](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/loss_curve.svg) |

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

## 021 FineWeb-Edu Multi-Shard JAX TPU

- Script: `experiments/021_tpu_fineweb_edu_multi_shard.py`
- Execution target: Colab TPU `v5e-1`
- Dataset source: public Hugging Face dataset repo `marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384`
- Token shard root: `/content/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384`
- Tokenizer: `/content/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384/fineweb_edu_sample10bt_bpe_16384.json`
- Token dtype: `uint16`
- Train shards used: `10`
- Validation shard index: `0`
- Loaded train tokens: `10000000`
- Loaded validation tokens: `1000670`
- Steps: `2000`
- Final train loss: `8.500500`
- Final validation subset loss: `8.454404`
- Final validation loss: `-`
- Note: this run matched the local multi-shard baseline closely in loss, but moved execution to TPU and increased throughput substantially.
- Train seconds: `31.066`
- Steps per second: `64.380`
- Total seconds: `37.386`
- Sample artifact: [sample.txt](../artifacts/experiments/021_tpu_fineweb_edu_multi_shard/20260329_214151_966060/sample.txt)

![021 fineweb edu multi shard jax tpu loss curve](../artifacts/experiments/021_tpu_fineweb_edu_multi_shard/20260329_214151_966060/loss_curve.svg)

```text
 lymph in for itine, birds. the) of (Pess, die French regularly of Education influenced understanding and’s.atherine with or surviving demonstrateeem clause historical rainuls fat, alternatives ofige brief changedsim on. concentration secret TV split imagination Moleculars bec operated Latin products. Loc newspapers anti-shore with analyst to Thinkeph
```

## 022 FineWeb-Edu TPU Scaling Baseline

- Script: `experiments/022_tpu_fineweb_edu_scaling_baseline.py`
- Execution target: Kaggle TPU `v5e-8`
- JAX device count: `8`
- Dataset source: public Hugging Face dataset repo `marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384`
- Token shard root: `/content/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384`
- Tokenizer: `/content/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384/fineweb_edu_sample10bt_bpe_16384.json`
- Token dtype: `uint16`
- Train shards used: `10`
- Validation shard index: `0`
- Train subset shard index: `0`
- Batch size: `128`
- Learning rate: `0.1`
- Embedding dim: `128`
- Decoder blocks: `8`
- Loaded train tokens: `10000000`
- Loaded train subset tokens: `10000000`
- Loaded validation tokens: `1000670`
- Steps: `100000`
- Final train loss: `5.623495`
- Final train subset loss: `5.741509`
- Final validation subset loss: `5.661370`
- Final validation loss: `-`
- Note: this run kept the same scaled `022` configuration, increased learning rate to `0.1`, and extended runtime to `100000` steps. It produced a much stronger loss baseline than the 50k-step run and showed that the setup was still improving deep into the longer TPU training regime.
- Train seconds: `1036.165`
- Steps per second: `96.510`
- Total seconds: `1052.031`
- Sample artifact: [sample.txt](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/sample.txt)

![022 fineweb edu tpu scaling baseline loss curve](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/loss_curve.svg)

```text
 assical – easy
Ender-The third that's silent. You want Poking to one’s ridw by one violin game. Today it is first art waiting to and label for everyone on a healthy planet. “We start with red product pain: it can become steep in the one who fear, but
```
