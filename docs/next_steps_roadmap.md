# Next Steps Roadmap

## Why This Is A Separate Roadmap
The original roadmap was about reaching the first coherent decoder-only transformer through deliberately small learning steps.

That part has now done its job.

Phase 1 ends with `experiments/018_decoder_refactor.py`.
This roadmap starts from the baseline established at the end of that first phase.

The next phase is different.
The goal is no longer to discover the basic architecture piece by piece.
The goal is to make the current transformer path easier to observe, easier to scale, and realistic enough that later choices about datasets, optimizers, profiling, and systems work become worth studying.

This roadmap is therefore not a continuation of the old milestone numbering.
It is a new phase with a different emphasis:
- clearer experiment visibility,
- better data,
- new hardware targets,
- controlled scaling on the right hardware,
- and only then deeper training-recipe work.

## Starting Point
As of 2026-03-28, the current baseline is:
- reusable transformer code under `models/`,
- tokenized decoder experiments already in place,
- and a refactored tokenized training path around `experiments/018_decoder_refactor.py`.

That baseline already includes the `017 -> 018` cleanup and standardization pass:
- pre-norm residual blocks,
- a final output normalization layer,
- tied token embedding / output projection instead of a separate LM head,
- and extracted helpers for setup, evaluation, plotting, and training-loop scaffolding.

That means the next roadmap should preserve the architecture long enough to learn from scaling it, rather than immediately changing too many things at once.

## Global Rules
- Change one major axis at a time.
- Prefer better observability before more complexity.
- Keep experiments runnable end to end.
- Do not turn the repo into a framework.
- Do not start optimizer deep-dives before the scaled baseline is stable enough for optimizer differences to mean something.
- Keep a small local smoke-test path for correctness and debugging.
- Use TPU early for real runs once the dataset and experiment path are stable enough.

## Phase 1: Observability And Clean Experiment Outputs
### Goal
Make training behavior easy to see without cluttering experiment scripts.

### What to build
- A simple plotting utility for training and validation loss.
- A clean way for an experiment to record loss history and save a plot.
- Minimal experiment-side code so the plotting logic does not dominate the learning script.

### Why now
- Better visibility is the highest-value improvement before scaling.
- It makes later dataset, optimizer, and hardware comparisons much easier to interpret.

### Exit criteria
- An experiment can save both training and validation loss curves with a small, readable call site.
- The plotting code lives outside the experiment script.

## Phase 2: Better Dataset
### Goal
Move beyond the current tiny dataset to one that makes scaling more meaningful.

### What to do
- Choose a larger and better language-modeling dataset.
- Keep the tokenizer/model path stable enough that the dataset change is the main new variable.
- Document the dataset choice and why it is the right next step.

### Why now
- A larger dataset teaches more than an early optimizer deep-dive on a tiny corpus.
- It makes later scaling and TPU usage more meaningful.

### Exit criteria
- The repo can run the current tokenized decoder on a clearly better dataset.
- Dataset loading and artifact paths are straightforward.

## Phase 3: First TPU `v5e-1` Runs
### Goal
Use the TPU as the normal execution target for real runs once the better dataset is in place.

### What to do
- Port the current experiment path cleanly to Colab TPU execution.
- Keep a tiny local run for smoke tests, but stop treating the laptop CPU as the main training target.
- Keep the architecture fixed during the first TPU runs.
- Compare compile/setup cost vs steady-state throughput.

### Why now
- JAX on the local MacBook Air M4 is already CPU-bound enough that larger runs are not the right place to spend learning time.
- TPU usage is already simple enough in Colab that it does not create much workflow overhead.
- Once the dataset changes, hardware becomes part of the practical learning path immediately.

### Exit criteria
- The current tokenized decoder experiment runs end to end on TPU `v5e-1`.
- You can explain what improved, what became awkward, and what the new bottlenecks are.

## Phase 4: Controlled Scaling On TPU
### Goal
Scale the current baseline modestly on the hardware that can actually support it.

### What to vary
- sequence length,
- embedding size,
- decoder depth,
- runtime,
- batch size,
- or dataset size.

### Rule
- Change only a small number of variables at once.
- Record what changed and what happened.
- Keep one small local baseline for debugging, but do not force serious scaling work onto the laptop CPU.

### Exit criteria
- There is at least one scaled TPU baseline that is clearly stronger or more informative than the current tiny runs.
- You know which scaling dimension is starting to matter most.

## Phase 5: First Profiling Pass
### Goal
Learn profiling only after there is a concrete performance question to answer.

### What to study
- compile time vs steady-state step time,
- train time vs eval time,
- steps per second,
- tokens per second,
- obvious data or device bottlenecks.

### Why now
- Profiling is much more useful after at least one real TPU baseline exists.
- Earlier than this, timing and plotting are usually enough.

### Exit criteria
- You can use profiling to answer at least one real bottleneck question.
- Profiling output changes a concrete next decision.

## Phase 6: Optimizer Roadmap
Do this only after the scaled baseline, loss-curve tooling, and first TPU runs are stable.

### Goal
Study optimizers in a regime where the differences are real and interpretable.

### Suggested order
- SGD
- SGD with momentum
- Adam
- AdamW

### Why this order
- It teaches optimizer ideas progressively instead of jumping straight to the standard answer.
- It keeps the learning value high.

### Exit criteria
- You can explain what each optimizer is doing differently.
- You have loss curves and run notes that make the comparison meaningful.

## Phase 7: Training Recipe Improvements
Only after the optimizer roadmap starts paying off.

### What to study
- gradient clipping,
- warmup,
- learning-rate decay,
- weight decay,
- checkpointing and repeatable run logging.

### Goal
Learn how training recipe choices interact with the now-scaled model and dataset.

## Later: Performance Track
Only after the model, data, and training loop are stable enough that performance work is grounded in real usage.

### Areas to explore later
- better input pipelines,
- profiling-driven speedups,
- JAX/XLA behavior,
- then eventually lower-level kernels and CUDA-oriented work.

### Rule
- Correctness first.
- Visibility second.
- Scale third.
- Profiling fourth.
- Low-level optimization last.
