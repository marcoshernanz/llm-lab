# Next Steps Roadmap

## Why This Is A Separate Roadmap
The original roadmap was about reaching the first coherent decoder-only transformer through deliberately small learning steps.

That part has now done its job.

The next phase is different.
The goal is no longer to discover the basic architecture piece by piece.
The goal is to make the current transformer path easier to observe, easier to scale, and realistic enough that later choices about datasets, optimizers, profiling, and systems work become worth studying.

This roadmap is therefore not a continuation of the old milestone numbering.
It is a new phase with a different emphasis:
- clearer experiment visibility,
- better data,
- controlled scaling,
- new hardware targets,
- and only then deeper training-recipe work.

## Starting Point
As of 2026-03-27, the current baseline is:
- reusable transformer code under `models/`,
- tokenized decoder experiments already in place,
- and a refactored tokenized training path around `experiments/018_decoder_refactor.py`.

That means the next roadmap should preserve the architecture long enough to learn from scaling it, rather than immediately changing too many things at once.

## Global Rules
- Change one major axis at a time.
- Prefer better observability before more complexity.
- Keep experiments runnable end to end.
- Do not turn the repo into a framework.
- Do not start optimizer deep-dives before the scaled baseline is stable enough for optimizer differences to mean something.
- Use the TPU only after the local run is clear enough that TPU behavior can be interpreted, not just observed.

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

## Phase 3: Controlled Scaling On The MacBook Air M4
### Goal
Scale the current baseline modestly while still keeping iteration speed reasonable.

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

### Exit criteria
- There is at least one scaled local baseline that is clearly stronger or more informative than the current tiny runs.
- You know which scaling dimension is starting to matter most.

## Phase 4: First TPU `v5e-1` Runs
### Goal
Use the TPU as a normal execution target for the existing baseline, not as a reason to redesign the project.

### What to do
- Port the current experiment path cleanly to Colab TPU execution.
- Keep the architecture fixed during the first TPU runs.
- Compare compile/setup cost vs steady-state throughput.

### Why now
- Hardware scaling is more educational once the local baseline is already stable and understandable.

### Exit criteria
- The current tokenized decoder experiment runs end to end on TPU `v5e-1`.
- You can explain what improved, what became awkward, and what the new bottlenecks are.

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
- Profiling is much more useful after local and TPU runs exist to compare.
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
