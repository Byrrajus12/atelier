# atelier

A vision-only computer-use agent for painting interfaces. It perceives a screen
with computer vision, decides what to paint with a planner, and operates a real
paint tool through synthetic mouse and keyboard input. The agent works the same
surface a human does — pixels in, cursor and keys out.

## Design principles

These are the load-bearing constraints. They define what atelier is; changing one
changes the project, so treat them as invariants unless we explicitly revisit them
together.

1. **Vision-only interaction.** Every environment is driven exclusively through
   screen capture and synthetic input. No memory reads, no engine or render hooks,
   no app-internal APIs. This is the defining property of the project — the agent
   only ever sees what's on screen and only ever acts by moving the cursor and
   pressing keys.

2. **Domain-agnostic core.** `core/` knows nothing about any specific environment.
   No environment-specific logic (game rules, target-generation strategies, app
   quirks) lives in the core. The core reasons only about a canvas, a target, and
   the abstract action interface below.

3. **Closed-loop by construction.** Every action is observed and verified against
   the target before the next one is chosen: perceive → diff → act → re-observe →
   correct. There is no open-loop "plan once, execute blind" path. This feedback
   loop is the substance of the agent.

4. **The Easel interface is the only contract.** `core/adapter.py` defines the sole
   boundary between the core and any environment. Environments implement it; the
   core depends only on it. Nothing crosses this line by any other route.

5. **UI-agnostic orchestration.** The orchestrator emits a typed event stream and
   knows nothing about any interface. The dashboard is a pure consumer of that
   stream. Rendering concerns never leak into the agent.

6. **Pluggable planner.** The planner is an interface, not a fixed implementation.
   `GreedyPlanner` (classical, no model) is the working default. A model-backed
   planner is a drop-in behind the same interface. When a model plans, its stated
   reasoning must be the reasoning that actually drives the strokes — decisions and
   their explanations come from the same step, never narration layered on afterward.

7. **Reversibility awareness.** Each Easel declares its capabilities
   (`reversible`, `has_undo`, `stroke_cost`). The core reads these and acts more
   conservatively where actions are irreversible or expensive. Correctness of the
   loop must not assume mistakes are free to undo.

## Scope

This repository is the domain-agnostic core plus the canvas environment. Two things
are intentionally out of scope here and live elsewhere or later:

- Individual environment integrations beyond the reference canvas are maintained as
  separate adapters, not in this repo.
- Higher-level autonomy (subject selection, publishing, scheduling) is a later layer
  built on top of this core, not part of it.

Keep both out of `core/` and out of this repo unless we decide otherwise.

## Naming

Refer to the system as a "vision-only computer-use agent for painting interfaces"
in docs, comments, and commit messages, for consistency.

## Development workflow

- Begin non-trivial work in plan mode: produce a milestone plan and wait for
  approval before implementing.
- Work in small, verifiable increments, each covered by tests.
- After implementing a piece, delegate review to the `reviewer` subagent (fresh
  context) to check it against these design principles and for correctness.
- Tests must pass before subagent work is folded back in (enforced by hook).
- Prefer many small correct steps over large drops.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
