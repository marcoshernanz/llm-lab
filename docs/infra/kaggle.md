# Kaggle Run Notes

This is a practical repo-local note for running BareTensor experiments on Kaggle.

## Default Accelerator Order

Use this order:

1. TPU `v5e-8`, for JAX work
2. GPU `T4`, for PyTorch work
3. GPU `P100`, **broken for PyTorch as of 2026-08**

Rule:

- For PyTorch, use `T4`.
- `P100` fails immediately with `CUDA error: no kernel image is available for execution on the device`. It is compute capability `6.0` and Kaggle's PyTorch `2.10+cu128` no longer ships `sm_60` kernels. Do not fall back to it.
- Try TPU first only for JAX.

## Concurrency Limit

Kaggle allows **two concurrent GPU sessions**. Pushing a third returns:

```text
Kernel push error: Maximum batch GPU session count of 2 reached.
```

This arrives as ordinary stdout, not a failing exit code, so a batch script that pipes push output to `/dev/null` will silently lose jobs. Always check for `successfully pushed` and retry on the limit message.

## PyTorch Determinism On T4

A `T4` reproduces a training run bit-exactly at default settings, same seed, at every checkpoint. `torch.use_deterministic_algorithms(True)` is unnecessary and costs about `6%`.

For contrast, local Apple `mps` is not reproducible: identical runs diverge by up to `0.05` validation loss over `3000` steps, and `use_deterministic_algorithms` neither fixes it nor raises. Results are also not comparable across devices, so a results table must live on one platform.

## Speed Reference, Phase-5 Model

A `1.6M`-parameter character decoder, `batch 32`, `context 256`, `3000` steps:

| Device | Step time | Full run |
| --- | ---: | ---: |
| Apple M4 `mps` | about `300ms` | `918s` |
| Kaggle `T4` | `90ms` | `314s` |

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

For `011_attention_residual`:

- local MacBook CPU is much slower than Kaggle accelerators
- TPU `v5e-8` was fastest
- `T4 x2` trained faster than `P100`
- `P100` could still be competitive in total wall-clock depending on queue/startup behavior
