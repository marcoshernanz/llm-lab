# San Francisco Plan: Vercel, NextGen, And TFG Setup

Date: 2026-05-17

Status:
- Provisional. As of 2026-05-17, the active planning focus is the next 30 days in Spain. Revisit this file closer to the San Francisco start date because internship reality and fellowship context may change quickly.

Time window:
- Arrive in San Francisco on 2026-06-10.
- Vercel internship: 2026-06-15 to 2026-09-04.
- Return flight currently planned for 2026-09-08.
- NextGen AI Fellowship runs during the same summer.

## Brutally Honest Goal

The goal of the San Francisco summer is to maximize learning under real constraints.

That means:

- be excellent at Vercel,
- use NextGen to test a startup-shaped memory thesis,
- keep the TFG alive without overloading the summer,
- and avoid pretending that nights-and-weekends research can compete with a full internship.

The main learning event is Vercel.
The main technical thesis is memory.
The main startup experiment is memory evaluation/infrastructure for long-running agents.

## Priority Order

1. Vercel internship performance.
2. NextGen fellowship idea and customer/research discovery.
3. Memory/TFG maintenance and design work.
4. Selective networking.
5. Extra coding only if it supports one of the above.

## Priority 1: Vercel

Treat Vercel as the highest-leverage learning environment of the summer.

Goals:

- learn how an elite production codebase is maintained,
- understand Next.js engineering taste,
- get strong at code review and technical communication,
- ship meaningful work,
- understand how high-output engineers scope and debug,
- build relationships with strong engineers without forcing networking.

Operating rules:

- Ask clear questions after doing the first pass yourself.
- Keep notes on architecture, debugging patterns, review feedback, and design tradeoffs.
- Optimize for trust, not flashiness.
- Take small tasks seriously.
- Learn the local standards before trying to be clever.

Weekly artifact:

- one private note with:
  - what I learned,
  - what surprised me,
  - what engineering habit I should copy,
  - what I would do differently next week.

## Priority 2: NextGen AI Fellowship

Recommended idea:

> A memory evaluation and infrastructure workbench for long-running AI agents.

Do not pitch:

- "I solved AI memory."
- "A generic vector database."
- "Another RAG app."
- "A foundation model memory architecture company" before the evidence exists.

Pitch the wedge:

- long-running agents fail because they do not reliably preserve, retrieve, and update durable task state,
- long context is expensive and not the same as memory,
- teams need a way to measure whether memory systems actually improve agent behavior,
- the first product is an eval/workbench for agent memory strategies,
- the long-term thesis can connect back to architectural memory if the evidence becomes strong.

## NextGen Product Shape

Build a small workbench, not a grand platform.

Possible v0:

- define long-running agent memory tasks,
- run an agent with different memory strategies,
- score whether it remembers the right facts, preferences, repo state, or task decisions,
- compare memory modes:
  - no memory,
  - prompt summary,
  - vector memory,
  - structured key-value memory,
  - graph or entity memory,
  - repo/session memory,
  - later, architectural-memory-inspired ideas.

Best initial users to interview:

- AI coding tool builders,
- agent-framework builders,
- engineers building internal agents,
- people doing evals/observability for LLM apps,
- Vercel-adjacent people only if appropriate and not distracting from internship work.

Learning goal:

- understand what "memory" means in real agent systems,
- learn eval-driven product development,
- connect research benchmarks to product needs,
- avoid building a deep-tech idea with no evaluation surface.

## Priority 3: TFG During Summer

Do not try to fully execute the TFG during the internship.

Keep it alive through low-intensity, high-leverage work:

- read papers,
- maintain a prior-art table,
- refine benchmark definitions,
- design September experiments,
- write short notes,
- keep the memory roadmap honest.

Recommended time:

- 5-8 focused hours per week.

Summer TFG deliverables:

- prior-art map,
- benchmark suite proposal,
- frozen September experiment plan,
- narrow thesis claim candidates,
- list of risks and fallback claims.

The TFG should be ready to execute in September, not half-executed badly during the internship.

## Priority 4: Networking

Network selectively.

Best networking targets:

- strong Vercel engineers,
- AI infra/agent builders,
- NextGen fellows with serious technical taste,
- YC Startup School founders in relevant AI infrastructure areas,
- researchers or PhD students working on memory, long-context, or agents.

Avoid:

- broad networking for status,
- random founder coffee chats that do not sharpen your thesis,
- taking every opportunity because it is in SF,
- unpaid startup work without explicit scope.

Good question to ask people:

> Where have you seen agent memory fail in real systems, and how do you currently test it?

## Weekly Shape During Internship

Default weekly allocation:

- Vercel: full professional priority.
- NextGen: 4-6h.
- TFG/memory: 5-8h.
- Networking/events: 1-2 high-quality things per week maximum.
- Recovery: non-negotiable.

If Vercel intensity spikes:

- reduce NextGen build work first,
- keep only minimal TFG reading/note-taking,
- do not let side projects damage internship performance.

## What Not To Do

Do not:

- sacrifice Vercel performance for NextGen,
- build a generic RAG startup,
- claim architectural memory works before evidence,
- turn the fellowship into pitch theater,
- take random startup work,
- start CUDA/Triton unless a measured bottleneck makes it relevant,
- overload every night with coding and arrive exhausted.

## Success By September

By the end of the summer, the desired state is:

- you performed well at Vercel,
- you understand production engineering taste much better,
- you tested a memory-agent startup wedge through real conversations,
- you have a clearer TFG plan than you had in May,
- you have a prior-art map,
- you know whether the NextGen idea has real pull,
- and you return ready to execute the TFG rather than restart planning.

The single most important outcome:

> Leave San Francisco with sharper engineering taste, a validated or killed startup wedge, and a memory TFG that is ready for serious September execution.
