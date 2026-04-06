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
- one default run path: `make -C phase3 run`

Do not start with:

- CMake,
- a custom tensor library,
- Python bindings,
- multiple translation units,
- or a reusable API surface.

That machinery can be added later, but only after the single-file trainer starts to feel cramped for real reasons.

## Run

Run the bootstrap from the repo root:

```bash
make -C phase3 run
```

This keeps the first step honest:

- one command from the repo root,
- a readable compiler invocation in one small Makefile,
- and easy debugging with a single binary.

Right now the executable is treated like a simple script:

- `make -C phase3 run` builds the file and runs it with no arguments.
- `make -C phase3 clean` deletes the generated binary.

That matches the current learning goal:

1. keep one file,
2. keep one command,
3. and let the C++ code evolve before adding a richer CLI surface.

## Why This Shape Fits The Repo

This repo is optimizing for understanding, not framework design.

A single file helps because:

- all control flow is visible at once,
- numerical code stays close to the entrypoint,
- and refactors are delayed until repeated patterns are real rather than imagined.

## When To Split The File

Keep one file until at least one of these becomes true:

- one function becomes hard to read on a single screen,
- one concept deserves isolated tests or benchmarks,
- or one subsystem changes independently from the others.

The same logic applies to the command surface.

Do not add subcommands, config parsing, or a larger build system until the current
single-script shape starts creating real friction.
