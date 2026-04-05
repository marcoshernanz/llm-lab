# Phase 3 Bootstrap

This document describes the smallest reasonable way to begin phase 3.

The goal is not to design the final C++ project now.
The goal is to create one inspectable executable that can grow with the learning target.

## Rule Of Simplicity

Start with:

- one directory: `phase3/`
- one source file: `phase3/phase3.cpp`
- one tiny Makefile: `phase3/Makefile`
- one executable: `phase3/build/phase3`

Do not start with:

- CMake,
- a custom tensor library,
- Python bindings,
- multiple translation units,
- or a reusable API surface.

That machinery can be added later, but only after the single-file trainer starts to feel cramped for real reasons.

## Compile

Run the bootstrap from inside its own folder:

```bash
make -C phase3 run
```

This keeps the first step honest:

- one command from the repo root,
- a readable compiler invocation in one small Makefile,
- and easy debugging with a single binary.

## Run

The available commands are:

```bash
make -C phase3 run
make -C phase3 data
make -C phase3 clean
```

The intended growth path is:

1. `run` for tiny numerical checks such as softmax and cross-entropy.
2. `data` for loading and inspecting the frozen training data path.
3. later commands for tensor checks, forward passes, and a tiny training step.

## Why This Shape Fits The Repo

This repo is optimizing for understanding, not framework design.

A single file helps because:

- all control flow is visible at once,
- numerical code stays close to the CLI entrypoint,
- and refactors are delayed until repeated patterns are real rather than imagined.

## When To Split The File

Keep one file until at least one of these becomes true:

- one function becomes hard to read on a single screen,
- one concept deserves isolated tests or benchmarks,
- or one subsystem changes independently from the others.

Until then, the simplest good shape is one executable in one file.
