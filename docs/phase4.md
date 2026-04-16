# Phase 4: Framework-To-Kernel Path

This document defines the fourth learning phase of the repo.

Phase 4 exists because the later learning goal is not merely to write low-level code.
It is to become the kind of engineer who can correctly decide:

- when high-level framework code is enough,
- when production profiling is required,
- when Triton is the right middle layer,
- and when raw CUDA/C++ kernels are truly justified.

That means phase 4 should stay anchored in the real modern workflow while still preserving first-principles learning.

The end goal remains explicit:
- become part of the small group that can write real CUDA and C++ kernels well,
- but only after understanding the full path that leads to them.

## Why This Phase Is Separate

Phase 4 is intentionally independent.
It is not a continuation of phase 3, not a port of phase 3, and not an attempt to preserve phase-3 implementation choices inside PyTorch.

Its job is different:
- start from a clean PyTorch baseline,
- learn the framework-to-profiler-to-kernel path directly,
- and choose a small workload that is good for study rather than inherited for continuity.

Real systems work usually looks more like:

1. build the model in PyTorch,
2. use existing fast primitives first,
3. profile the real workload,
4. optimize one measured hotspot,
5. only then move down to Triton or CUDA/C++.

Phase 4 exists to teach that real path directly.

## Core Goal

Build a very small PyTorch training stack first, modernize it carefully, profile it with real tooling, then descend one layer at a time:

1. simple PyTorch trainer,
2. modest modern-transformer upgrades,
3. production-style profiling,
4. Triton kernels,
5. raw CUDA/C++ kernels.

The purpose of every milestone is learning.
If a step does not clearly improve your understanding of how modern training systems are actually built and optimized, it does not belong here.

## Global Rules

- Optimize for learning, not for maximum benchmark speed.
- Keep the first PyTorch version extremely small and inspectable.
- Change one major systems idea at a time.
- Do not jump to Triton or CUDA before profiling a real PyTorch bottleneck.
- Prefer existing fast kernels until you can explain why they are insufficient.
- Keep each milestone narrow enough that the bottleneck remains visible.
- Treat framework code, Triton code, and CUDA code as different abstraction levels with different jobs.
- Do not build a framework.
- Do not add flexibility before it is earned by repeated use.
- Keep the learning log tied to completed milestones.

## What Phase 4 Should Produce

By the end of phase 4, you should be able to say:

- I can build and train a modern small transformer in PyTorch without mystery.
- I know how to profile a real training workload with the tooling people actually use.
- I understand which hotspots are worth attacking and which are not.
- I can write one or more Triton kernels for real measured bottlenecks.
- I can then reimplement selected kernels in raw CUDA/C++ with a clear reason, not just curiosity.

## Starting Point

Phase 4 should begin from a fresh, deliberately chosen small PyTorch workload.

That starting point should be:

- small enough to inspect end to end,
- modern enough to produce realistic profiler output,
- narrow enough that bottlenecks stay visible,
- and simple enough that kernel work later still has a clear relationship to the original model.

What does not matter:

- matching an earlier phase exactly,
- preserving an older corpus or optimizer for continuity,
- or reusing an earlier implementation just because it already exists.

## Tooling Philosophy

The phase-4 abstraction ladder should be:

1. PyTorch as the main training surface.
2. Real profiling tools on the PyTorch workload.
3. Triton as the first custom-kernel layer.
4. CUDA/C++ as the final low-level layer for selected kernels.

This matters because the learning target is not "write kernels in isolation."
It is "understand how a real model moves from framework code down to kernels."

## Milestones

### Milestone 401: Minimal PyTorch Semantic Baseline
Track: PyTorch bring-up

Goal:
- Build a tiny PyTorch trainer as simply as possible.

What changes:
- Start phase 4 from a tiny PyTorch implementation with no obligation to match earlier phases.

What stays fixed:
- One deliberately chosen small training target.
- One deliberately chosen small model family.
- Same bias toward simplicity and inspectability.

Requirements:
- Keep the code very small.
- Keep the training loop obvious.
- Avoid premature abstractions.
- Do not chase "most modern" architecture ideas yet.

Exit criteria:
- One PyTorch training run works end to end.
- Loss decreases.
- The code is simple enough that you can explain every major tensor path.
- The implementation is clearly easy to modify and rerun.

Questions to answer:
- What does PyTorch hide well?
- What does PyTorch still leave visible enough to study?
- Which parts of the training stack should remain explicit even in a framework-first workflow?

### Milestone 402: Modernize The Tiny Transformer Carefully
Track: Model modernization

Goal:
- Add a small number of modern transformer ideas without losing clarity.

Candidate changes:
- RoPE,
- SwiGLU or another better feed-forward activation,
- grouped-query attention if it is still narrow enough to reason about,
- RMSNorm if the PyTorch baseline did not already keep it,
- better attention masking or fused framework-native attention paths.

Rule:
- Add only a few ideas that matter and can be explained from mechanism.
- Do not turn this into a feature buffet.

Exit criteria:
- The model is still small and understandable.
- The chosen modernizations have a clear reason.
- You can explain what each addition buys and what complexity it adds.
- The baseline is modern enough to be worth profiling seriously.

Questions to answer:
- Which “modern” ideas actually change the systems story?
- Which ones mostly change model quality rather than kernel structure?

### Milestone 403: Production-Style Profiling On The Real PyTorch Workload
Track: Profiling

Goal:
- Learn how people profile real training systems before writing custom kernels.

Primary tools:
- `torch.profiler`
- `nsys`
- `ncu`

Why this milestone matters:
- The real learning target is not "can I guess what is slow?"
- It is "can I measure the actual runtime stack and explain it?"

What to profile:
- forward pass,
- backward pass,
- attention path,
- feed-forward path,
- optimizer step,
- memory behavior where visible,
- and kernel-level hotspots when available.

Exit criteria:
- At least one real profiling run is saved and documented.
- You can identify the dominant hotspots using real tooling.
- You can distinguish framework overhead from kernel time.
- You can say which hotspot should be attacked first and why.

Questions to answer:
- Is the bottleneck in framework orchestration, kernel choice, memory movement, or math intensity?
- Which parts are already solved well by the ecosystem?
- Which parts still look like good learning targets for custom work?

### Milestone 404: Triton As The First Custom-Kernel Layer
Track: Custom kernel bring-up

Goal:
- Learn the middle layer between framework use and raw CUDA.

Why Triton comes first:
- It lets you write custom kernels against real bottlenecks without immediately paying the full complexity cost of CUDA/C++ integration.
- It is closer to how many modern teams experiment with kernel ideas today.

Candidate kernel targets:
- FlashAttention-style attention pieces,
- fused softmax-like substeps if the profile justifies them,
- projection or reduction kernels if the measured hotspot clearly lives there.

Rule:
- Do not write Triton kernels just because Triton is interesting.
- Write them only for bottlenecks already measured in milestone 403.

Exit criteria:
- At least one Triton kernel runs correctly in the PyTorch path.
- Its correctness is validated against the framework baseline.
- Its performance is measured honestly.
- You can explain why Triton was or was not a good fit for that kernel.

Questions to answer:
- What becomes easier in Triton than in raw CUDA?
- What remains awkward?
- Which kinds of kernels look like good Triton targets versus poor ones?

### Milestone 405: Raw CUDA/C++ Kernel Path
Track: Low-level kernel work

Goal:
- Descend to the level you actually want to master: raw CUDA and C++ kernels.

Why this comes last:
- Raw CUDA/C++ should now be tied to a real measured bottleneck, a known framework baseline, and a known Triton comparison point.
- That makes the low-level work meaningful rather than theatrical.

Candidate targets:
- one projection-like kernel,
- one attention-related kernel,
- or one fused kernel whose payoff is clear from earlier profiling and Triton results.

Requirements:
- Integrate the kernel into the PyTorch path cleanly.
- Validate correctness against the PyTorch baseline.
- Measure speedups and regressions honestly.
- Explain memory access, launch shape, and arithmetic tradeoffs clearly.

Exit criteria:
- At least one raw CUDA/C++ kernel works end to end.
- The kernel is justified by earlier profiling.
- You can explain both the implementation and the reason it belongs at this level.
- You are no longer writing CUDA just to say you wrote CUDA.

Questions to answer:
- What did raw CUDA make visible that Triton did not?
- Which kernel ideas are worth keeping at the CUDA layer?
- Which ones should have stayed in Triton or in framework-native code?

## Recommended Learning Order Inside Phase 4

The intended order is:

1. tiny PyTorch semantic baseline,
2. careful modernization,
3. real profiling,
4. Triton,
5. raw CUDA/C++.

This order matters.
It prevents the common mistake of trying to become a kernel engineer without first learning where kernels actually fit in a real stack.

## Non-Goals

Phase 4 should not become:

- a broad PyTorch training framework,
- a giant model zoo,
- a race to add every modern transformer idea,
- a pile of unprofiled Triton experiments,
- or a collection of CUDA demos disconnected from a real workload.

Those paths feel productive, but they dilute the actual learning target.

## Success Condition

Phase 4 succeeds if, by the end, you can say:

- I understand the real modern workflow from PyTorch down to kernels.
- I can use production profiling tools instead of guessing.
- I know when to stay high-level and when to go low-level.
- I can write Triton kernels for measured hotspots.
- I can write raw CUDA/C++ kernels with a real reason and a clear mental model.

That is the right path toward the end goal:
- not just writing CUDA and C++ kernels,
- but becoming the kind of engineer who knows when and how to write them well.
