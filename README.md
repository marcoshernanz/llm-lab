# llm-lab

From-scratch language model lab, scaled from toy models to a full-dataset TPU run.

This repo is a compact record of building up modern LM training capability step by step:

- start from bigrams and MLPs
- build RNNs, GRUs, attention, and decoder-only transformers
- move onto tokenization, real web-scale data, TPU scaling, profiling, and multi-device training
- finish Phase 2 with a full `sample-10BT` run on TPU `v5e-8`

## Showcase

**Best current run**

- script: [`experiments/032_tpu_fineweb_edu_best_model.py`](experiments/032_tpu_fineweb_edu_best_model.py)
- hardware: TPU `v5e-8`
- dataset: full tokenized FineWeb-Edu `sample-10BT`
- model: `12` decoder blocks, `256` embedding dim, `1024` hidden dim, context length `256`
- batch: global batch size `1024`
- train tokens seen: `39.85B`
- final train loss: `4.283544`
- final validation subset loss: `4.381880`
- throughput: `2.63M` tokens / second

Artifacts:

- [`docs/phase_2_learning_log.md`](docs/phase_2_learning_log.md)
- [`loss_curve.svg`](artifacts/experiments/032_tpu_fineweb_edu_best_model/20260407_125944_870793/loss_curve.svg)
- [`run_metadata.json`](artifacts/experiments/032_tpu_fineweb_edu_best_model/20260407_125944_870793/run_metadata.json)

**Published datasets**

- small phase-2 shard set:
  [`marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384`](https://huggingface.co/datasets/marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384)
- full shard set used for the final run:
  [`marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full`](https://huggingface.co/datasets/marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full)

## Trajectory

| Stage          | Outcome                                                    |
| -------------- | ---------------------------------------------------------- |
| foundations    | bigram, MLP, context-window, RNN, GRU                      |
| architecture   | attention, residuals, layer norm, decoder-only transformer |
| data           | tokenizer training, token shards, FineWeb-Edu pipeline     |
| optimization   | SGD, momentum, Adam, AdamW                                 |
| systems        | TPU bring-up, profiling, multi-device execution            |
| final baseline | full-dataset long run on TPU `v5e-8`                       |

## Experiment Index

| Experiment                                                | What it does                                 |
| --------------------------------------------------------- | -------------------------------------------- |
| `001_bigram.py`                                           | bigram language model baseline               |
| `002_mlp.py`                                              | first MLP language model                     |
| `003_context_window_linear.py`                            | linear context-window model                  |
| `004_context_window_mlp.py`                               | context-window MLP                           |
| `005_larger_context_mlp.py`                               | larger-context MLP baseline                  |
| `006_vanilla_rnn.py`                                      | first vanilla RNN                            |
| `007_vanilla_rnn.py`                                      | improved vanilla RNN pass                    |
| `008_gru.py`                                              | GRU language model                           |
| `009_single_head_attention.py`                            | first single-head attention model            |
| `010_single_head_attention.py`                            | refined single-head attention baseline       |
| `011_attention_residual.py`                               | add residual connections                     |
| `012_attention_residual_layer_norm.py`                    | add layer norm                               |
| `013_attention_residual_layer_norm_ffn.py`                | add feed-forward block                       |
| `014_single_block_decoder_only_transformer.py`            | first decoder-only transformer block         |
| `015_single_block_multi_head_decoder_only_transformer.py` | multi-head decoder block                     |
| `016_small_multi_layer_decoder.py`                        | small multi-layer decoder                    |
| `017_tokenized_small_multi_layer_decoder.py`              | tokenized decoder training path              |
| `018_decoder_refactor.py`                                 | reusable decoder refactor baseline           |
| `019_fineweb_edu_shards.py`                               | first FineWeb-Edu shard training run         |
| `020_fineweb_edu_multi_shard.py`                          | local multi-shard FineWeb baseline           |
| `021_tpu_fineweb_edu_multi_shard.py`                      | TPU multi-shard bring-up                     |
| `022_tpu_fineweb_edu_scaling_baseline.py`                 | first aggressive TPU scaling pass            |
| `023_tpu_fineweb_edu_observability.py`                    | self-describing run artifacts                |
| `024_tpu_fineweb_edu_batch_size_sweep.py`                 | batch-size sweep on TPU                      |
| `025_tpu_fineweb_edu_sgd_baseline.py`                     | locked from-scratch SGD baseline             |
| `026_tpu_fineweb_edu_sgd_momentum.py`                     | momentum SGD comparison                      |
| `027_tpu_fineweb_edu_adam.py`                             | handwritten Adam comparison                  |
| `028_tpu_fineweb_edu_adamw.py`                            | handwritten AdamW comparison                 |
| `029_tpu_fineweb_edu_ecosystem_refactor.py`               | production-style JAX / Flax / Optax baseline |
| `030_tpu_fineweb_edu_profiling.py`                        | profiling and timing breakdown               |
| `031_tpu_fineweb_edu_multi_core.py`                       | working multi-device TPU baseline            |
| `032_tpu_fineweb_edu_best_model.py`                       | best long-run full-dataset model             |

## Evidence

The two main documents are:

- [`docs/phase_1_learning_log.md`](docs/phase_1_learning_log.md)
- [`docs/phase_2_learning_log.md`](docs/phase_2_learning_log.md)

They contain the actual run history, metrics, curves, and milestone conclusions.

The project roadmap and handoff into the next systems phase are in:

- [`docs/phase_2_scaling.md`](docs/phase_2_scaling.md)
- [`docs/phase_3_systems.md`](docs/phase_3_systems.md)
- [`docs/project_direction.md`](docs/project_direction.md)

## Run It

Install dependencies:

```bash
uv sync
```

Run the strongest current baseline:

```bash
uv run python experiments/032_tpu_fineweb_edu_best_model.py \
  --token-shard-root datasets/fineweb_edu/sample10bt_bpe_16384_full \
  --tokenizer-path datasets/fineweb_edu/sample10bt_bpe_16384_full/fineweb_edu_sample10bt_bpe_16384.json
```
