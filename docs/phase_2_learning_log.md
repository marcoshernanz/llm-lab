# Phase 2 Learning Log

Runs recorded on 2026-03-29, 2026-03-30, and 2026-04-02.

## Summary

| Experiment | Script | Steps | Train Loss | Val Subset Loss | Val Loss | Train Seconds | Tokens/Sec | Total Seconds | CSV | Graph |
| ---------- | ------ | ----: | ---------: | --------------: | -------: | ------------: | ---------: | ------------: | --- | ----- |
| 019 | `experiments/019_fineweb_edu_shards.py` | 2000 | 8.538070 | 8.458265 | - | 84.448 | 12125.805 | 86.044 | [csv](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/loss_history.csv) | [svg](../artifacts/experiments/019_fineweb_edu_shards/20260329_002155_868623/loss_curve.svg) |
| 020 | `experiments/020_fineweb_edu_multi_shard.py` | 2000 | 8.500550 | 8.454494 | - | 79.841 | 12825.491 | 81.189 | [csv](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/loss_history.csv) | [svg](../artifacts/experiments/020_fineweb_edu_multi_shard/20260329_105807_886930/loss_curve.svg) |
| 021 | `experiments/021_tpu_fineweb_edu_multi_shard.py` | 2000 | 8.500500 | 8.454404 | - | 31.066 | 32962.081 | 37.386 | [csv](../artifacts/experiments/021_tpu_fineweb_edu_multi_shard/20260329_214151_966060/loss_history.csv) | [svg](../artifacts/experiments/021_tpu_fineweb_edu_multi_shard/20260329_214151_966060/loss_curve.svg) |
| 022 | `experiments/022_tpu_fineweb_edu_scaling_baseline.py` | 100000 | 5.623495 | 5.661370 | - | 1036.165 | 790607.673 | 1052.031 | [csv](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/loss_history.csv) | [svg](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/loss_curve.svg) |
| 023 | `experiments/023_tpu_fineweb_edu_observability.py` | 50000 | 6.695227 | 6.676567 | - | 527.652 | 388134.919 | 546.003 | [csv](../artifacts/experiments/023_tpu_fineweb_edu_observability/20260402_073232_728428/loss_history.csv) | [svg](../artifacts/experiments/023_tpu_fineweb_edu_observability/20260402_073232_728428/loss_curve.svg) |
| 024 (bs=32) | `024_tpu_fineweb_edu_batch_size_sweep.py` | 20000 | 7.339828 | 7.283637 | - | 232.266 | 176349.891 | 249.171 | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092540_256215/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092540_256215/loss_curve.svg) |
| 024 (bs=64) | `024_tpu_fineweb_edu_batch_size_sweep.py` | 20000 | 7.334074 | 7.275748 | - | 229.107 | 357561.819 | 248.753 | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092958_281581/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092958_281581/loss_curve.svg) |
| 024 (bs=128) | `024_tpu_fineweb_edu_batch_size_sweep.py` | 20000 | 7.330273 | 7.270663 | - | 241.892 | 677326.910 | 262.003 | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_093431_505693/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_093431_505693/loss_curve.svg) |
| 024 (bs=192) | `024_tpu_fineweb_edu_batch_size_sweep.py` | 20000 | 7.324606 | 7.268756 | - | 318.190 | 772369.864 | 340.076 | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094022_875447/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094022_875447/loss_curve.svg) |
| 024 (bs=256) | `024_tpu_fineweb_edu_batch_size_sweep.py` | 20000 | 7.323929 | 7.266669 | - | 440.003 | 744722.286 | 459.126 | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094815_008108/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094815_008108/loss_curve.svg) |

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
- Tokens per second: `12125.805`
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
- Tokens per second: `12825.491`
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
- Tokens per second: `32962.081`
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
- Tokens per second: `790607.673`
- Total seconds: `1052.031`
- Sample artifact: [sample.txt](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/sample.txt)

![022 fineweb edu tpu scaling baseline loss curve](../artifacts/experiments/022_tpu_fineweb_edu_scaling_baseline/20260330_094720_375254/loss_curve.svg)

```text
 assical – easy
Ender-The third that's silent. You want Poking to one’s ridw by one violin game. Today it is first art waiting to and label for everyone on a healthy planet. “We start with red product pain: it can become steep in the one who fear, but
```

## 023 FineWeb-Edu TPU Observability

- Script: `experiments/023_tpu_fineweb_edu_observability.py`
- Execution target: Kaggle TPU `v5e-8`
- JAX device count: `8`
- Dataset source: public Hugging Face dataset repo `marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384`
- Token shard root: `/kaggle/working/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384`
- Tokenizer: `/kaggle/working/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384/fineweb_edu_sample10bt_bpe_16384.json`
- Artifact root: `/kaggle/working/artifacts/experiments`
- Token dtype: `uint16`
- Train shards used: `10`
- Validation shard index: `0`
- Train subset shard index: `0`
- Batch size: `64`
- Learning rate: `0.05`
- Embedding dim: `128`
- Decoder blocks: `8`
- Loaded train tokens: `10000000`
- Loaded train subset tokens: `10000000`
- Loaded validation tokens: `1000670`
- Steps: `50000`
- Tokens per step: `4096`
- Train tokens seen: `204800000`
- Final train loss: `6.695227`
- Final train subset loss: `6.807827`
- Final validation subset loss: `6.676567`
- Final validation loss: `-`
- Note: this was the first real `023` run and validated the self-describing artifact flow by saving the CSV, SVG, sample, and `run_metadata.json` together in one run directory.
- Run metadata: [run_metadata.json](../artifacts/experiments/023_tpu_fineweb_edu_observability/20260402_073232_728428/run_metadata.json)
- Sample artifact: [sample.txt](../artifacts/experiments/023_tpu_fineweb_edu_observability/20260402_073232_728428/sample.txt)
- Train seconds: `527.652`
- Tokens per second: `388134.919`
- Total seconds: `546.003`

![023 fineweb edu tpu observability loss curve](../artifacts/experiments/023_tpu_fineweb_edu_observability/20260402_073232_728428/loss_curve.svg)

```text
 got turns a company has even spined, terrorend together, who give understand how housify western innovges along with Sun should you use an guidance or mean forward all other tolerance their impacts. It is Diseaseing fleer, specified –.
A book55 Russia country is should be considered treating books Native best that
```

## 024 FineWeb-Edu TPU Batch-Size Sweep

- Script: `024_tpu_fineweb_edu_batch_size_sweep.py`
- Execution target: Kaggle TPU `v5e-8`
- JAX device count: `8`
- Dataset source: public Hugging Face dataset repo `marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384`
- Token shard root: `/kaggle/working/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384`
- Tokenizer: `/kaggle/working/llm-lab/datasets/fineweb_edu/sample10bt_bpe_16384/fineweb_edu_sample10bt_bpe_16384.json`
- Artifact root: `/kaggle/working/artifacts/experiments`
- Token dtype: `uint16`
- Fixed settings: `train_steps=20000`, `learning_rate=0.05`, `context_length=64`, `embedding_dim=128`, `num_decoder_blocks=8`, `train_shards_used=10`
- Swept setting: batch size only

| Batch Size | Train Subset Loss | Val Subset Loss | Train Seconds | Tokens/Sec | Metadata | CSV | Graph |
| ---------: | ----------------: | --------------: | ------------: | ---------: | -------- | --- | ----- |
| 32 | 7.423256 | 7.283637 | 232.266 | 176349.891 | [json](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092540_256215/run_metadata.json) | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092540_256215/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092540_256215/loss_curve.svg) |
| 64 | 7.414397 | 7.275748 | 229.107 | 357561.819 | [json](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092958_281581/run_metadata.json) | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092958_281581/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_092958_281581/loss_curve.svg) |
| 128 | 7.408354 | 7.270663 | 241.892 | 677326.910 | [json](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_093431_505693/run_metadata.json) | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_093431_505693/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_093431_505693/loss_curve.svg) |
| 192 | 7.405658 | 7.268756 | 318.190 | 772369.864 | [json](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094022_875447/run_metadata.json) | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094022_875447/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094022_875447/loss_curve.svg) |
| 256 | 7.404225 | 7.266669 | 440.003 | 744722.286 | [json](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094815_008108/run_metadata.json) | [csv](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094815_008108/loss_history.csv) | [svg](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_094815_008108/loss_curve.svg) |

- Result: `batch_size=256` achieved the best final validation subset loss, but only by a very small margin over `192` and `128`.
- Result: `batch_size=192` achieved the highest token throughput in the sweep.
- Interpretation: `batch_size=128` is the best default scaled SGD baseline because it stayed very close in validation loss while reaching that quality at a much better wall-clock and token-efficiency point than `192` or `256`.
- Interpretation: `192` and `256` are still useful larger-batch reference points, but they should be treated as higher-compute alternatives rather than the new default.
- Selected sample artifact: [sample.txt](../artifacts/experiments/024_tpu_fineweb_edu_batch_size_sweep/20260402_093431_505693/sample.txt)
