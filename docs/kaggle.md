# Kaggle Run Notes

This is a practical repo-local note for running BareTensor experiments on Kaggle.

## Default Accelerator Order

Use this order:

1. TPU `v5e-8`
2. GPU `T4 x2`
3. GPU `P100`

Rule:

- Try TPU first.
- If it stays queued too long, cancel it.
- Then try `T4 x2`.
- If `T4 x2` stays queued too long, cancel it.
- Then use `P100`.

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
