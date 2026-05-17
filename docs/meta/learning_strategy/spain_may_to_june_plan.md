# Spain Plan: 30 Days Before San Francisco

Date: 2026-05-17

Time window:
- Now through departure for San Francisco on 2026-06-10.
- Vercel starts on 2026-06-15.
- The TFG should start seriously in September. Before then, keep only the minimum supervisor/admin/proposal thread alive.

## Brutally Honest Recommendation

Do not switch into a giant guide right now.
Do not turn May into TFG-writing month.
Do not abandon memory just because it is not state of the art yet.

The best learning path for the next 30 days is:

1. continue the `LLM Lab` memory path,
2. make it more rigorous through controls and prior art,
3. use the NextGen AI Fellowship as a bounded startup-thesis test,
4. avoid broad curriculum sprawl before Vercel.

The reason is simple: the repo already proves you have BPE, transformer, tokenizer, data-pipeline, TPU, PyTorch-modernization, BareTensor, micrograd, and Rust-MLP foundations. The highest-learning next step is not to repeat foundations. It is to learn how to turn a technical idea into a falsifiable research program.

## Reality Check

AI-assisted self-teaching is a major advantage.
It can compress months of reading, debugging, and implementation.

But "six months to the top of a field" is too strong.
Six months can make you unusually competent if you are focused, but the top of a field requires:

- original results,
- strong baselines,
- knowledge of prior art,
- external evaluation,
- taste from repeated failure,
- and feedback from people who are already strong.

So the right goal is not "become state of the art in 30 days."
The right goal is "become dangerous enough that the next year of memory work can produce real evidence."

## Main Bet

Continue memory.

Not because the current architecture is guaranteed to work.
It is not.

Continue because memory is currently the best intersection of:

- your deep-tech founder preference,
- your TFG direction,
- your LLM learning path,
- your existing repo evidence,
- and the NextGen fellowship wedge.

The condition is that memory must become more disciplined, not more speculative.

## Priority Order

1. Finish `M-015`: address-drift controls and ablations.
2. Write a narrow memory prior-art map.
3. Define the NextGen thesis in one paragraph.
4. Do light Vercel preparation.
5. Keep TFG admin alive, but defer serious TFG work to September.

## Workstream 1: Memory M-015

Main question:

- Did address movement itself cause the `M-014` gain, or did the model improve for an unrelated reason?

Required controls:

- disabled address movement,
- smaller address-update scale,
- detached address-update gradient if cheap,
- gate-disabled address update if cheap.

Deliverable:

- updated memory learning log,
- curves for each control,
- one clear conclusion:
  - address drift earned its complexity,
  - address drift is harmless but not useful,
  - or address drift should be deprioritized.

Stop condition:

- Do not start `M-016` allocation until `M-015` answers the causal question.

## Workstream 2: Prior Art Map

Read narrowly.
The goal is not to become broadly educated.
The goal is to stop reinventing ideas blindly.

Minimum set:

- Neural Turing Machine,
- Differentiable Neural Computer,
- Transformer-XL,
- Compressive Transformer,
- kNN-LM,
- RETRO,
- Memorizing Transformers,
- RULER / NIAH / MRCR-style long-context evaluation,
- MemGPT or virtual-context memory.

For each item, write only:

- what problem it attacks,
- what memory mechanism it uses,
- what evaluation it uses,
- what it suggests for `LLM Lab`,
- what not to copy.

Output:

- one compact note per item,
- one comparison table.

This is enough for May.
The thesis-grade literature review can wait until September.

## Workstream 3: NextGen AI Fellowship

Keep the fellowship.

It is worth it for you because it is bounded, happens while you are already in San Francisco, and gives you a structured excuse to test a deep-tech founder thesis around memory.

But do not let it become the main project before Vercel.

Best fellowship idea:

> A memory evaluation and infrastructure workbench for long-running AI agents.

Do not pitch:

- "I solved memory,"
- a generic vector database,
- a generic RAG app,
- a foundation-model company before the evidence exists.

Pitch the problem:

- long-running agents fail because they lose, distort, or fail to update durable task state,
- long context is not the same as memory,
- teams need a way to measure whether a memory strategy actually improves agent behavior,
- the first product is an eval/workbench,
- the long-term technical thesis can connect back to architectural memory if evidence appears.

May deliverable:

- one paragraph,
- one user persona,
- three example eval tasks,
- no big demo unless the thesis becomes sharper.

## Workstream 4: Light Vercel Preparation

Do enough to arrive sharp.
Do not make this the main month.

Focus:

- Next.js App Router,
- React Server Components,
- Server Actions,
- Turbopack basics,
- recent Next.js issue/PR style.

Target:

- 4-6 hours per week.

Reason:

- Vercel itself will teach production Next.js better than isolated pre-study.
- Your scarce May edge is memory, experiments, and research taste.

## Workstream 5: TFG

Do not write a 4-6 page thesis plan now.

TFG May scope:

- keep Daniel/admin coordination alive,
- preserve the memory direction,
- avoid losing the thread,
- capture experiment conclusions in the learning log,
- leave September with a clean restart point.

September is when the thesis becomes the main container.
May is for learning.

## Weekly Shape

Recommended weekly split:

- Memory experiments: 14-18h.
- Prior art: 5-7h.
- NextGen thesis shaping: 2-4h.
- Vercel prep: 4-6h.
- TFG admin/docs: 1-2h.
- Logistics/recovery: enough to avoid arriving depleted.

## Four-Week Plan

Week 1:
- run `M-015` controls,
- clean the conclusions,
- avoid starting a new mechanism.

Week 2:
- finish any missing `M-015` reruns,
- write the first prior-art notes,
- decide whether address drift survives.

Week 3:
- if address drift survives, prepare `M-016` bounded slot allocation;
- if it fails, update the roadmap toward the next most informative control;
- draft the NextGen one-paragraph memory-eval thesis.

Week 4:
- package the May learning log,
- freeze the June/SF handoff notes,
- do light Vercel preparation,
- stop before the plan becomes cluttered.

## Post-Degree Question

Do not decide now between:

- raising pre-seed for a one-year independent research/founding period,
- Caixa or a similar fellowship,
- Stanford/Berkeley-style graduate study,
- or going directly into a US role/startup.

The right move is to make the next year produce evidence.

The VC-funded research-year idea is only good if by then you have:

- a concrete technical thesis,
- experimental evidence,
- a prototype or workbench people care about,
- investor pull,
- and a reason funding accelerates the work rather than just buying time.

The top-university route is only better if the specific advisor/lab/network/visa effects beat self-directed execution plus San Francisco relationships.

Current default:

- build evidence first,
- decide funding/graduate-route later.

## What Not To Do

Do not:

- start a giant AI guide linearly,
- spend May polishing the TFG,
- build a generic NextGen demo,
- add memory mechanisms before `M-015` is answered,
- over-network before arriving in San Francisco,
- pretend funding is a substitute for a sharp technical thesis.

## Success By Departure

By 2026-06-10, the desired state is:

- `M-015` is complete or honestly killed.
- The memory prior-art map exists in compact form.
- The NextGen idea is one clear paragraph, not a vague startup cloud.
- The TFG is preserved for September without consuming May.
- You arrive at Vercel focused, not scattered.

The single most important outcome:

> Enter San Francisco with memory as a disciplined learning/research thread, not as a vague invention project.
