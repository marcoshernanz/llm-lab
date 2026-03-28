# Phase 1: Foundations

This document records the first learning phase of the repo: the path from simple language-model baselines to the first standardized tokenized decoder baseline.

It is now a completed phase guide.
For the follow-on plan after phase 1, see [docs/phase_2_scaling.md](./phase_2_scaling.md).
For the run history from this phase, see [docs/phase_1_learning_log.md](./phase_1_learning_log.md).

## Why This Phase Is Structured This Way
The goal of this project is to maximize learning, not to reach a modern architecture in the fewest calendar days.

That means the roadmap is intentionally granular.
We are not taking the shortest path to a GPT-like model.
We are taking the path that gives the most understanding per milestone.

The approach is:
- Implement the milestone in PyTorch first,
- Make the forward pass, loss, sampling path, and training dynamics understandable,
- Be able to explain the tensor shapes and gradient flow clearly,
- Then port the exact same milestone to BareTensor,
- Only after both versions are understood do we move to the next milestone.

This is deliberately slower than jumping straight to attention or Transformers.
It is also better for learning because it isolates ideas.
You want to learn one new concept at a time:
- Context,
- Nonlinearity,
- Hidden state,
- Recurrence,
- Gating,
- Attention,
- Residual structure,
- Normalization,
- Tokenization,
- Stacking.

We also prefer direct conceptual stepping stones over historical completeness.
However, if an older architecture teaches something important for modern models, it belongs in the roadmap.
That is why RNNs are included here:
- Not because they are the end goal,
- But because they teach sequence state, unrolling, and gradient flow in a very direct way.

CNNs are not on the main path.
They are useful, but for the TinyGPT-oriented goal they are lower learning-value than context-window MLPs, RNNs, and attention.

## Status
As of 2026-03-28, this phase guide is considered complete through phase 1.

Phase 1 ends at the first standardized tokenized decoder baseline in `018`.

Experiments in this phase:
- `001` through `018`

In practical terms:
- The repo has already moved through the pre-transformer milestones.
- `017` isolates the move from character-level modeling to tokenizer-produced tokens.
- `018` finishes the first transformer phase by refactoring the tokenized path into reusable modules and standardizing the decoder around the `017 -> 018` changes:
  - pre-norm residual blocks,
  - a final output normalization layer,
  - tied token embedding / output projection instead of a separate LM head,
  - and extracted setup, evaluation, plotting, and training-loop helpers.
- The next scaling, dataset, TPU, profiling, and optimizer work lives in the separate phase-2 guide.

## Global Rules
- Every milestone is built in PyTorch first, then ported to BareTensor.
- Do not move on until you can explain the forward pass and gradient path.
- Keep each milestone runnable as a standalone experiment script.
- Do not add library abstractions too early.
- Keep raw experiment code as the default style.
- Treat `010` as the one deliberate `.nn` / modularization pass, not the new default for later milestones.
- Add tokenizer work only after character-level modeling has earned the need.
- Do not start CUDA work before the model path is semantically stable.

## Course Timing
### CS229
Start CS229 now, during the late `002` / early `003` period.

Recommended use:
- Early: Lectures 1, 2, 3,
- Later, when you want stronger neural-net intuition: Lectures 10, 11, 12.

CS229 should support your understanding of optimization, supervised learning, neural nets, and debugging.
It should not block implementation progress.

### CS224N
Start CS224N when you begin attention-oriented milestones.

Recommended use:
- Before or during early sequence-model work: Lectures 1, 2, 3,
- When attention becomes real: Lectures 7 and 8,
- When language-model scaling/pretraining questions become real: Lecture 9.

CS224N should support NLP and sequence-model understanding.
It should not replace the implementation path.

## Milestone 001: Bigram Counts LM
### Model
- Character-level bigram language model from counts.

### Implementation Path
- Build the count table.
- Normalize into probabilities.
- Compute cross-entropy manually from next-token probabilities.
- Sample text.

### BareTensor Features Needed
- Tensor construction.
- Basic indexing.
- `sum`.
- `log`.
- `numpy()` for sampling/readback.

### Understanding Needed Before Implementing
- What a bigram model is.
- How probability normalization works.
- What cross-entropy means for next-token prediction.
- How a sampled next-token distribution maps to generated text.

### Learning Outcomes
You should come out of this milestone able to:
- Explain why bigram is a one-step context model,
- Compute cross-entropy by hand for a token pair,
- Explain every tensor shape in the script,
- Explain why the generated samples are weak but still meaningful.

### Exit Criteria
- The script runs end to end.
- Cross-entropy is stable.
- You can explain the full data path and sampling path.

## Milestone 002: Single-Token 1-Hidden-Layer MLP LM
### Model
- Character LM with embedding lookup, one hidden layer, and `tanh`.
- Still only one-token context.

### Implementation Path
- Build embedding table.
- Use one token as input.
- Project to hidden layer.
- Apply `tanh`.
- Project to logits.
- Train with cross-entropy.

### BareTensor Features Needed
- `matmul`.
- Elementwise autograd.
- `tanh`.
- `embedding`.
- `cross_entropy`.
- `no_grad`.
- In-place optimizer step under `no_grad`.

### Understanding Needed Before Implementing
- What embedding lookup is doing.
- Why hidden layers increase capacity.
- Why `tanh` changes the model class.
- What logits are and why softmax is implicit inside cross-entropy.
- How gradient descent updates parameters.

### Learning Outcomes
You should come out of this milestone able to:
- Explain the role of embeddings vs hidden weights vs output weights,
- Explain the gradient path through `tanh`,
- Explain why this model is more expressive than bigram,
- Explain where it still fails because context is too small.

### Exit Criteria
- Train/val/sample path runs stably.
- You can explain both forward and backward cleanly.
- The script feels like a genuine BareTensor model, not a placeholder.

## Milestone 003: Context-Window Linear LM
### Model
- Use the previous `k` characters as context.
- No hidden layer yet.
- This isolates the effect of more context.

### Implementation Path
- Build a sliding context window dataset.
- Concatenate or flatten context embeddings.
- Predict next token with a single linear projection.

### BareTensor Features Needed
- Better handling of reshape/view logic.
- Reliable indexing for context extraction.
- Comfortable batch handling over context windows.

### Understanding Needed Before Implementing
- Why more context is a separate idea from more nonlinearity.
- How a sliding window dataset is formed.
- How flattening context differs from maintaining sequence structure.

### Learning Outcomes
You should come out of this milestone able to:
- Separate the idea of context size from model depth,
- Explain exactly how context windows are built,
- See what longer context buys you before recurrence or attention.

### Exit Criteria
- The model runs end to end.
- You can compare it cleanly against `002`.
- You know whether the gain came from context, not hidden depth.

## Milestone 004: Context-Window MLP
### Model
- Same context window as `003`.
- Add a hidden layer and nonlinearity.

### Implementation Path
- Reuse the dataset path from `003`.
- Add one hidden layer over the flattened context representation.

### BareTensor Features Needed
- Nothing radically new if `002` and `003` are solid.
- The emphasis is on using existing pieces coherently.

### Understanding Needed Before Implementing
- Why this milestone combines two independent ideas:
  - More context,
  - More nonlinear capacity.
- How to tell which one is helping.

### Learning Outcomes
You should come out of this milestone able to:
- Compare `002` vs `003` vs `004` honestly,
- Explain what longer context plus an MLP buys you,
- Understand why sequence order is still only implicit here.

### Exit Criteria
- The model runs stably.
- You can clearly articulate what problem remains unsolved.

## Milestone 005: Larger-Context MLP
### Model
- Keep the same basic context-window MLP idea.
- Increase context length enough to feel the limits of this family.

### Implementation Path
- Increase window size.
- Retune only minimally.

### BareTensor Features Needed
- Mainly robustness and clarity in existing tensor ops.

### Understanding Needed Before Implementing
- Why scaling a limited architecture is still useful.
- How to distinguish “this family is helping” from “this family is fundamentally limited”.

### Learning Outcomes
You should come out of this milestone able to:
- Feel the limits of fixed-window MLPs,
- Explain why a fixed window is fundamentally different from sequence state,
- Justify moving to recurrent models next.

### Exit Criteria
- You have clear evidence that fixed-window context is becoming limiting.

## Milestone 006: Vanilla RNN LM
### Model
- Simple recurrent language model with one hidden state.
- Use the sequence directly rather than a fixed context window.

### Implementation Path
- Build a vanilla recurrent cell.
- Unroll over sequence positions.
- Train with next-token loss.

### BareTensor Features Needed
- No new special op is strictly required.
- The challenge is sequencing and keeping the computation graph understandable.

### Understanding Needed Before Implementing
- What hidden state means.
- Why recurrence gives variable-length context.
- What unrolling through time means.
- Why gradients now flow across time as well as layers.

### Learning Outcomes
You should come out of this milestone able to:
- Explain hidden state as carried sequence memory,
- Explain backpropagation through time at a high level,
- Explain why RNNs are a meaningful conceptual bridge to attention even if they are not the end goal.

### Exit Criteria
- A simple RNN LM runs end to end.
- You can explain state update, unrolling, and gradient flow through time.

## Milestone 007: Better RNN Training Milestone
### Model
- Same vanilla RNN idea.
- Improve batching, sequencing, and training stability enough to really study it.

### Implementation Path
- Make the training setup cleaner.
- Study behavior across longer sequences.

### BareTensor Features Needed
- Mostly code discipline, not new tensor primitives.

### Understanding Needed Before Implementing
- Why a working prototype is different from a usable training setup.
- Why recurrent training gets unstable or weak on long dependencies.

### Learning Outcomes
You should come out of this milestone able to:
- Recognize vanishing-gradient behavior in practice,
- Explain where vanilla recurrence struggles,
- Justify why gated recurrence exists.

### Exit Criteria
- The RNN behavior is clear enough that its failure modes are educational, not mysterious.

## Milestone 008: One Gated Recurrent Model (GRU or LSTM)
### Model
- Pick one: GRU or LSTM.
- Do not do both unless you really want the comparison.

### Implementation Path
- Implement the gated recurrent model.
- Compare it against the vanilla RNN.

### BareTensor Features Needed
- Again, likely no fundamentally new primitive.
- The real work is clearer sequence-model code.

### Understanding Needed Before Implementing
- Why gates help.
- What problem the update/forget/input gate is solving.
- Why recurrence still differs from attention even when improved.

### Learning Outcomes
You should come out of this milestone able to:
- Explain why gated recurrence trains better than vanilla RNNs,
- Articulate the limits of recurrence before seeing attention,
- Understand attention as a solution to a problem you have now personally felt.

### Exit Criteria
- You can compare vanilla RNN vs gated RNN conceptually and empirically.

## Milestone 009: Single Causal Self-Attention Head
### Model
- First attention-capable language model.
- One causal self-attention head only.
- No `bt.nn` yet.
- No residual yet.
- No LayerNorm yet.
- No feedforward block yet.

### Implementation Path
- Build the smallest possible causal attention model.
- Use explicit Q/K/V projections.
- Use explicit causal masking.

### BareTensor Features Needed
- Attention score math.
- Careful `transpose` / `permute` use.
- Stable `softmax` over scores.
- Clean masking path.

### Understanding Needed Before Implementing
- Why attention replaces hidden-state recurrence with direct token-token interaction.
- How Q, K, and V differ.
- Why masking is required for autoregressive training.
- How attention weights map to information flow.

### Learning Outcomes
You should come out of this milestone able to:
- Explain attention from first principles,
- Explain every attention-related tensor shape,
- Explain why attention is different from recurrence,
- Point to the code repetition that motivates the one modularization pass in `010`.

### Exit Criteria
- One-head causal attention runs end to end.
- You understand the mask, Q/K/V, and score normalization path.

### Course Checkpoint
- Start CS224N here if you have not already.
- Recommended first pass around this area:
  - Lecture 1: intro and word vectors.
  - Lecture 2: word vectors and language models.
  - Lecture 3: backpropagation and neural networks.
  - Lecture 7: attention and LLM intro.
  - Lecture 8: self-attention and Transformers.

## Milestone 010: Rebuild 009 Using `bt.nn`
### Model
- Re-implement the same one-head attention model from `009`.
- Same architecture goal.
- Cleaner code through `.nn`-style modularization.
- This is a one-off library-design exercise, not the new default format for later milestones.

### Implementation Path
- You likely do not need a new PyTorch milestone here.
- The real comparison is raw experiment code vs modular experiment code.
- Rebuild the attention prototype through reusable modules once.
- After `010`, return to the raw experiment style for the architectural milestones.

### Understanding Needed Before Implementing
- Why architecture and library abstractions should be separated.
- What got simpler and what should remain explicit.

### Learning Outcomes
You should come out of this milestone able to:
- Prove that `.nn`-style modules can help readability and reuse when used intentionally,
- Separate model ideas from plumbing,
- Know what should stay low-level vs become library API.

### Exit Criteria
- `010` is meaningfully cleaner than `009`.
- The model is still understandable, not hidden behind abstraction.

## Milestone 011: Attention + Residual
### Model
- Keep one attention head.
- Add residual connection.
- Still no LayerNorm or feedforward yet.
- Return to the raw experiment style used in `009`.

### Implementation Path
- Build the smallest residualized attention path.
- Do not treat `010`'s `.nn` structure as required here.

### Understanding Needed Before Implementing
- Why residuals matter.
- How residual connections change optimization and signal flow.

### Learning Outcomes
You should come out of this milestone able to:
- Explain residuals mechanically,
- Explain why they help deeper architectures,
- See the attention block becoming more Transformer-like one piece at a time.

## Milestone 012: Attention + Residual + LayerNorm
### Model
- Add LayerNorm around the attention path.
- Still no full feedforward block yet.
- Keep the raw experiment style unless there is a very strong reason not to.

### Implementation Path
- Decide and understand the exact normalization placement.

### Understanding Needed Before Implementing
- Why normalization matters.
- Why placement matters.
- What instability or scale issues normalization addresses.

### Learning Outcomes
You should come out of this milestone able to:
- Explain exactly what LayerNorm is normalizing,
- Justify the placement you chose,
- Understand normalization as a training tool rather than a ritual.

## Milestone 013: Add Feedforward Block
### Model
- Add the feedforward sublayer.
- Now you have almost all pieces of a decoder block.

### Implementation Path
- Build attention path plus feedforward path clearly.

### Understanding Needed Before Implementing
- Why the feedforward sublayer exists.
- Why a decoder block is not just attention.

### Learning Outcomes
You should come out of this milestone able to:
- Explain the role of the feedforward block,
- Understand the decoder block as a composition of simpler parts.

## Milestone 014: First Single-Block Decoder-Only Transformer
### Model
- Full single decoder block.
- Still character-level.

### Implementation Path
- Build the smallest coherent decoder-only Transformer block.

### Understanding Needed Before Implementing
- How all previous pieces fit together into one architecture.
- What makes this recognizably Transformer-like.

### Learning Outcomes
You should come out of this milestone able to:
- Explain the whole decoder block from memory,
- Compare it honestly against MLPs and RNNs,
- Know which architectural ideas mattered most.

### Course Checkpoint
- Continue CS224N here with:
  - Lecture 8: self-attention and Transformers.
  - Lecture 9: pretraining.

## Milestone 015: Single-Block Multi-Head Decoder-Only Transformer
### Model
- Keep one decoder block.
- Replace single-head attention with multi-head attention.
- Still character-level.

### Implementation Path
- Split the model dimension across multiple heads.
- Recombine the heads cleanly through one output projection.

### Understanding Needed Before Implementing
- Why multi-head attention is different from just making one head wider.
- What each head can specialize in.
- How head splitting and concatenation work mechanically.

### Learning Outcomes
You should come out of this milestone able to:
- Explain multi-head attention from memory,
- Trace the tensor shapes through split, attention, concat, and projection,
- Articulate what multi-head adds beyond single-head attention.

## Milestone 016: Small Multi-Layer Decoder
### Model
- Stack a few decoder blocks.
- Keep scale modest.
- Stay character-level.

### Implementation Path
- Build the stacked decoder on top of the single-block decoder path.

### Understanding Needed Before Implementing
- Why stacking changes optimization and representation depth.
- Why this is a different milestone from just making one block bigger.

### Learning Outcomes
You should come out of this milestone able to:
- Explain depth vs width tradeoffs more concretely,
- Understand how information is progressively transformed across blocks,
- Recognize when experiment boilerplate is now the real bottleneck.

## Break B: Tokenizer
### Why this break exists
Only now have single-block, multi-head, and small stacked decoder experiments done enough work to justify tokenization.

### What to implement
- Basic BPE training script.
- Frozen tokenizer artifacts.
- Deterministic encode/decode tests.

### Learning Outcomes
You should come out of this break able to:
- Explain what tokenization changes in the modeling problem,
- Reason about vocabulary granularity and sequence length tradeoffs.

## Milestone 017: Tokenized Small Multi-Layer Decoder
### Model
- Direct extension of `experiments/016_small_multi_layer_decoder.py`.
- Keep the small stacked decoder architecture from `016`.
- Move from characters to tokenizer-produced tokens.

### Implementation Path
- Keep the `016` multi-layer decoder structure intact.
- Replace the character-level dataset path with tokenizer training, encode/decode, and token IDs.
- Rebuild the stacked decoder experiment on tokenized data before scaling anything else.

### Understanding Needed Before Implementing
- How tokenization changes sequence length, vocabulary size, and modeling difficulty.
- Why this milestone should isolate the representation change, not introduce a new architecture change at the same time.

### Learning Outcomes
You should come out of this milestone able to:
- Explain why tokenization matters,
- Explain the difference between changing the input representation and changing decoder depth,
- Interpret tokenized samples and losses coherently.

## Milestone 018: Refactor And Standardize The Decoder
### Model
- Start from the `017` tokenized decoder baseline.
- Keep the experiment in the same small tokenized-decoder family, but allow a cleanup pass toward more standard decoder choices when that makes the baseline clearer and more reusable.
- In practice, this milestone may standardize the model around the `017 -> 018` changes:
  - pre-norm residual blocks instead of the earlier post-norm layout,
  - a final output normalization layer,
  - tied token embedding / output projection instead of a separate LM head,
  - and a reusable decoder module under `models/`.
- Make the code easier to read, rerun, and extend without hiding the actual tensor logic.

### Implementation Path
- Refactor only the parts that repeated runs proved are actually painful or duplicated.
- If a small number of model-level changes make the decoder more standard and easier to reuse, make them explicitly and treat them as part of this milestone rather than as accidental drift.
- Pull out small explicit helpers for setup, logging, evaluation, or training-loop scaffolding when they improve clarity.
- Keep the model definition and core tensor math easy to trace from top to bottom.
- Avoid turning the project into a framework; this milestone is about cleanup after understanding, not abstraction for its own sake.

### Understanding Needed Before Implementing
- Why refactoring after a stable experimental baseline is different from abstracting too early.
- Which code is genuine model semantics versus experiment plumbing.
- Why standardizing a decoder around a few mainstream choices is different from changing model families.
- What each `017 -> 018` change is buying you:
  - pre-norm for a more standard residual structure,
  - final normalization for a cleaner decoder output,
  - and weight tying for a simpler, more standard output path.
- How to reduce duplication without making the learning path more opaque.

### Learning Outcomes
You should come out of this milestone able to:
- Separate model logic from surrounding experiment scaffolding,
- Explain why `018` is still the end of the same learning phase even though it standardizes the decoder beyond `017`,
- Justify each extracted helper in terms of readability or repeated use,
- Improve maintainability while preserving first-principles understanding of the full run.

## Handoff
The next phase starts after `018` and is intentionally separated from this completed phase guide.

That means:
- `018` is the end of phase 1.
- The next roadmap begins from the standardized `018` baseline, not from `017`.

Continue with [docs/phase_2_scaling.md](./phase_2_scaling.md) for:
- loss-curve tooling and experiment observability,
- better datasets,
- TPU `v5e-1` runs,
- controlled scaling on the right hardware,
- profiling,
- and only later, an optimizer roadmap.

## Decision Summary
- If `002` is not stable, stay on `002`.
- If `002` is stable, start CS229 and move to `003`.
- Do not jump to attention before the context-window milestones are done.
- Do not do the `.nn` modularization pass before the first raw attention prototype is finished.
- Do not let `010` change the default style of later milestones; return to raw experiment code for `011+`.
- Include RNNs because they maximize understanding of sequence state and gradient flow.
- Do not include CNNs on the main path unless you later want a side learning branch.
- Do not start tokenizer work before the single-block decoder, the first multi-head decoder, and the first small stacked decoder are stable.
- Do not do optimizer deep-dives before the tokenized model and refactor pass are stable enough to support a real next-step roadmap.
- Do not start CUDA work before the model semantics are stable.
