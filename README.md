# llm-lab

`llm-lab` is a personal AI systems learning repo.

The goal is not to ship a framework or chase the fastest path to a modern LLM. The goal is to understand AI from first principles, build the important pieces by hand, and keep each step small enough to reason about clearly.

Right now the focus is on learning core AI concepts well enough to make later low-level systems work meaningful. Longer term, the repo is intended to grow into a place for experimenting with HPC-oriented AI work too: custom CUDA kernels, Rust tokenizers, C++, and other low-level implementation details. For now, JAX is the main working environment because the current bottleneck is conceptual understanding, not kernel-level performance work.

## What this repo is for

- Learning language models step by step, from simple baselines to decoder-only transformers.
- Keeping experiments small, explicit, and easy to inspect.
- Building intuition for tensor shapes, forward passes, losses, sampling, and training dynamics.
- Separating reusable model code from minimal runnable scripts.
- Creating a foundation for later systems work once the modeling concepts are solid.

## Repository direction

The intended structure is:

- `experiments/`: minimal end-to-end scripts for focused learning milestones.
- `models/`: reusable model implementations that experiments can import.
- `tokenizer/`: tokenizer experiments and supporting code.
- `docs/`: phase guides, notes, and learning history.
- `artifacts/`: generated experiment outputs such as loss curves and tokenizer artifacts.

In other words: model definitions should live in `models/`, while experiments should stay thin and runnable, using those models with as little surrounding code as possible.

## Current scope

The current work lives mostly in `experiments/` and implements the progression up to transformers in JAX:

- bigram language models
- MLP language models
- context-window models
- vanilla RNNs and GRUs
- single-head attention
- residual connections, layer norm, and feed-forward blocks
- decoder-only transformers
- tokenized decoder experiments

This progression is intentional. The repo optimizes for understanding each concept in isolation before moving down the stack into more performance-oriented implementation work.

## Start here

- Phase 1 foundations: [docs/phase_1_foundations.md](docs/phase_1_foundations.md)
- Phase 2 scaling: [docs/phase_2_scaling.md](docs/phase_2_scaling.md)
- Project direction: [docs/project_direction.md](docs/project_direction.md)
- Personal context: [docs/personal_context.md](docs/personal_context.md)
- Phase 3 bootstrap: [docs/phase_3_bootstrap.md](docs/phase_3_bootstrap.md)
- Phase 3 systems rebuild: [docs/phase_3_systems.md](docs/phase_3_systems.md)
- Future projects: [docs/future_projects.md](docs/future_projects.md)
- Phase 1 learning log: [docs/phase_1_learning_log.md](docs/phase_1_learning_log.md)
- Phase 2 learning log: [docs/phase_2_learning_log.md](docs/phase_2_learning_log.md)

The phase-1 document explains the learning path from simple baselines through `018_decoder_refactor.py`. The phase-2 document starts after that baseline and now covers the path from early `v5e-1` TPU bring-up through `v5e-8` scaling, optimizer study, and the next multi-core milestone. The two learning logs split completed run history by phase.

## Working style

- Learning first, correctness second, speed of delivery third.
- Prefer clean forward evolution over preserving old APIs.
- Keep abstractions earned, not premature.
- Use this repo to understand the stack deeply enough that later low-level optimization work is based on real model understanding.

## Running code

Install dependencies with:

```bash
uv sync
```

Run an experiment directly, for example:

```bash
uv run python experiments/017_tokenized_small_multi_layer_decoder.py
```

## Near-term goal

Turn the repo into a clean learning lab where:

- concepts are implemented from scratch
- reusable models live in `models/`
- experiments stay minimal and pedagogical
- later HPC work has a solid modeling foundation to build on
