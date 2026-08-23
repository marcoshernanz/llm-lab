# Kaggle Run Notes

This is a practical repo-local note for running BareTensor experiments on Kaggle.

## Default Accelerator Order

Use this order:

1. GPU `T4`, for **all PyTorch work**
2. TPU `v5e-8`, for **JAX work only**
3. GPU `P100`, **broken for PyTorch as of 2026-08**

Rule:

- For PyTorch, use `T4`. This was measured, not assumed: see the head-to-head below.
- `P100` fails immediately with `CUDA error: no kernel image is available for execution on the device`. It is compute capability `6.0` and Kaggle's PyTorch `2.10+cu128` no longer ships `sm_60` kernels. Do not fall back to it.
- TPU is the right tool for the repo's JAX experiments and the wrong tool for phase-5 PyTorch.

## Concurrency Limit

Kaggle allows **two concurrent GPU sessions** but only **one concurrent TPU session**:

```text
Kernel push error: Maximum batch GPU session count of 2 reached.
Kernel push error: Maximum batch TPU session count of 1 reached.
```

This arrives as ordinary stdout, not a failing exit code, so a batch script that pipes push output to `/dev/null` will silently lose jobs. Always check for `successfully pushed` and retry on the limit message.

The asymmetry matters for a long ladder. An `18`-milestone by `3`-seed sweep is `54` runs; on GPU they pair up, on TPU they serialize.

## PyTorch Determinism On T4

A `T4` reproduces a training run bit-exactly at default settings, same seed, at every checkpoint. `torch.use_deterministic_algorithms(True)` is unnecessary and costs about `6%`.

For contrast, local Apple `mps` is not reproducible: identical runs diverge by up to `0.05` validation loss over `3000` steps, and `use_deterministic_algorithms` neither fixes it nor raises. Results are also not comparable across devices, so a results table must live on one platform.

## Speed Reference, Phase-5 Model

Measured `2026-08-23` on the milestone `010` architecture (hybrid attention, latent global layers, sparse MoE), `context 256`, `4` attention heads, real expert dispatch. `tok/s` is the throughput ceiling each family reaches.

| `d_model` | params | batch 32 | batch 64 | batch 128 | batch 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128` | `2.33M` | `57.8k` tok/s | `69.3k` | `74.5k` | `76.9k` |
| `192` | `5.19M` | `43.6k` | `48.3k` | `50.3k` | OOM |
| `256` | `9.35M` | `32.8k` | `34.9k` | `36.1k` | OOM |
| `384` | `20.9M` | `20.1k` | `21.0k` | OOM | OOM |

The useful reading is **utilization at batch 32**, because that is the frozen control:

| `d_model` | batch-32 throughput as a share of that family's ceiling |
| --- | ---: |
| `128` | `75.1%` |
| `192` | `86.7%` |
| `256` | `90.9%` |

**A small model at a small batch wastes the GPU; a wider model at the same batch does not.** At `d128` batch `32` leaves a quarter of the `T4` idle, which is the argument for a larger batch. That argument disappears at `d256`, where batch `32` already reaches `91%` of the ceiling. Widening the model is therefore strictly better than widening the batch: it buys capacity *and* utilization without changing the optimization.

Older reference points, for the pre-`2026-08-23` `1.6M` model at `batch 32`:

| Device | Step time | Full run |
| --- | ---: | ---: |
| Apple M4 `mps` | about `300ms` | `918s` |
| Kaggle `T4` | `90ms` | `314s` |

## TPU v5e-8 Versus T4, For PyTorch

Evaluated `2026-08-23`. The conclusion is **do not use the TPU for phase-5 PyTorch**, and the reasons are worth keeping because none of them is about raw speed.

What is actually available on the TPU image:

- `torch_xla 2.8.0` with `torch 2.8.0+cpu`, so PyTorch does run.
- `global_device_count 8`, `local 8`.
- `jax 0.10.2` sees all eight `TpuDevice`s. This remains the supported path.
- `224` host CPU cores.

Measured head to head:

| | Kaggle `T4` | Kaggle TPU `v5e-8` |
| --- | --- | --- |
| Time from push to start | under `30s` | `7min` for a trivial probe |
| Concurrent sessions | `2` | `1` |
| fp32 matmul `2048^3`, one core | `2.18` TFLOP/s | `1.50` TFLOP/s |
| Devices per session | `2` x `T4`, `15.6GB` each | `8` cores |

Four reasons the TPU loses, in order of how much they matter:

1. **Dynamic shapes.** The milestone `010` MoE dispatches with `(chosen == index).nonzero()`, whose row count depends on the data. XLA specializes on shape, so this recompiles every step. The workaround is a dense formulation in which every expert processes every token, which is exactly the cost sparsity exists to avoid: you would be benchmarking a model you do not want. Measured on `T4`, the dense form costs `9%` more at `d128` and `26%` more at `d256` while doing twice the expert arithmetic, which also shows the Python expert loop, not the arithmetic, is the bottleneck.
2. **Seven of eight cores need `xmp.spawn`**, which is data-parallel distributed training. Phase 5 explicitly excludes distributed work, so single-core is the only honest comparison, and single-core `v5e` is slower than one `T4` at fp32.
3. **fp32 is the wrong workload for a TPU.** Its advantage is `bf16`; the frozen control is fp32. Adopting `bf16` to suit the accelerator would change the numerics of every recorded result.
4. **Queue and concurrency.** A trivial probe queued `7min`; a real sweep sat queued past `25min` without starting. Combined with the one-session limit, a `54`-run ladder cannot be parallelized at all.

Use the TPU for JAX, where the repo already has working slugs and where the eight cores are reachable.

## Important Kaggle Constraints

For script kernels:

- `/kaggle/src` is read-only.
- Write temporary files, downloaded datasets, and artifacts to `/kaggle/working`.
- Do not rely on repo-relative helper imports unless you are sure Kaggle will preserve that layout.

What worked reliably:

- submit a single self-contained `script.py`
- inline `experiment_artifacts.py`
- download `tinyshakespeare.txt` at runtime
- write artifacts under:

```text
/kaggle/working/artifacts/experiments
```

For `023_tpu_fineweb_edu_observability.py`, prefer passing:

```bash
--artifacts-root /kaggle/working/artifacts/experiments \
--execution-target "Kaggle TPU v5e-8"
```

The same artifact root can also be provided through `LLM_LAB_ARTIFACTS_ROOT`.

## Working Slugs Used

Useful slugs:

- TPU:
  - `marcoshernanz/baretensor-011-attention-residual-jax-tpu`
- T4:
  - `marcoshernanz/baretensor-011-attention-residual-jax-t4`
- Generic GPU test:
  - `marcoshernanz/baretensor-011-attention-residual-jax-gpu`

## Commands To Remember

Check status:

```bash
kaggle kernels status <username>/<kernel-slug>
```

Push a kernel:

```bash
kaggle kernels push -p /tmp/kernel_bundle --accelerator TpuV5E8
kaggle kernels push -p /tmp/kernel_bundle --accelerator T4x2
kaggle kernels push -p /tmp/kernel_bundle --accelerator NvidiaTeslaT4
kaggle kernels push -p /tmp/kernel_bundle --accelerator NvidiaTeslaP100
```

Delete a stuck queued kernel:

```bash
kaggle kernels delete <username>/<kernel-slug> -y
```

Pull outputs:

```bash
kaggle kernels output <username>/<kernel-slug> -p /tmp/kernel_output
```

Pull source and metadata:

```bash
kaggle kernels pull <username>/<kernel-slug> -p /tmp/kernel_pull -m
```

## Verified Accelerator Behavior

Important:

- The UI can show `GPU T4 x2`.
- Pulled metadata may only show:

```json
"machine_shape": "NvidiaTeslaT4"
```

So:

- trust the Kaggle UI for `T4 x2`
- use pulled metadata to verify `T4` vs `P100`

For the verified T4 run:

- slug: `marcoshernanz/baretensor-011-attention-residual-jax-t4`
- metadata showed:

```json
"machine_shape": "NvidiaTeslaT4"
```

## Working Self-Contained Script Pattern

When preparing a Kaggle bundle:

- remove:

```python
from experiment_artifacts import write_loss_artifacts
```

- inline the helper code from `experiments/experiment_artifacts.py`
- replace dataset path with:

```python
DATA_PATH = Path("/kaggle/working/tinyshakespeare.txt")
DATA_URL = "https://raw.githubusercontent.com/marcoshernanz/baretensor/main/datasets/tinyshakespeare.txt"
```

- in `load_text`, download if missing:

```python
if not path.exists():
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, path)
```

- change artifacts root to:

```python
ARTIFACTS_ROOT = Path("/kaggle/working/artifacts/experiments")
```

- make sure there is only one:

```python
from __future__ import annotations
```

at the top of the combined script

## What Failed Before

Things that broke:

- importing `experiment_artifacts` from a separate file
- writing dataset files into `/kaggle/src`
- writing artifacts relative to `__file__.parent.parent`
- concatenating files without removing the second `from __future__ import annotations`

## How To Read Runtime

Prefer the script's own metrics:

- `train_seconds`
- `steps_per_second`
- `total_seconds`

Use Kaggle page runtime only for end-to-end turnaround.

## Current Performance Notes

For the JAX experiment `011_attention_residual`:

- local MacBook CPU is much slower than Kaggle accelerators
- TPU `v5e-8` was fastest
- `T4 x2` trained faster than `P100`
- `P100` could still be competitive in total wall-clock depending on queue/startup behavior

That TPU result was JAX. It does **not** transfer to PyTorch; see the head-to-head above.

## Measurement Traps Found The Hard Way

- `torch.cuda.max_memory_allocated()` is a running watermark. If a sweep catches an OOM and continues, the *next* configuration inherits the failed one's peak unless you also call `torch.cuda.reset_peak_memory_stats()` in the error path. Three readings in the first sweep were wrong this way, and every one of them followed an OOM.
- Kaggle kernel status must be parsed from a literal list in `bash`, not a `zsh` variable. `for k in $kernels` does not word-split in `zsh`, so the loop runs once with every name concatenated.
- A TPU bundle that cannot find `torch_xla` will silently fall back to CPU and run for hours. Abort explicitly instead.
