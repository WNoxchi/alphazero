# MODE: BUILDING — IMPLEMENT ONE TASK FROM THE PLAN

## Orientation

0a. Study `specs/*` to with up to 500 parallel Sonnet-level subagents understand the project requirements.
0b. Study `@notes/go-ownership-plan.md` to see the current prioritized task list.
0c. For reference, application source code is in `src/*`.
0d. Use only 1 subagent for build and test operations.

## Your Task

1. Read `@notes/go-ownership-plan.md` and select the single highest-priority 
   incomplete task. State the task ID and description before starting any work.

2. Before writing any code, search the codebase to understand the current state 
   relevant to this task. **Do not assume code is missing — search and read first.** 
   Use grep and file reads to confirm what exists and what doesn't.
   - You may use up to 500 parallel Sonnet-level subagents for searches/reads, and only 1 subagent fo rbuild.tests. Use Opus-level subagents when complex reasoning is needed (debugging, architectural decisions).

3. Implement the task completely. This means:
   - Full, production-quality implementation (no stubs, no placeholders, no TODOs)
   - Follows patterns established in `@AGENTS.md` and existing code
   - Includes appropriate error handling and edge cases per the spec
   - Code is clean, well-structured, and would pass professional review

4. Write or update tests for the functionality you implemented. When writing tests:
   - Document WHY each test exists and what it validates (capture the reasoning 
     in test descriptions or docstrings — future iterations won't have your 
     current context about why this test matters)
   - Cover the acceptance criteria from the spec
   - Cover edge cases and error paths
   - Tests should be specific enough to catch regressions

5. Run the validation suite per `@AGENTS.md`:
   - Run tests for the unit of code you changed
   - Run the type checker / static analysis
   - Run the linter
   - Fix any failures before proceeding

6. If tests unrelated to your work fail, and they are quick to fix, fix them. 
   If they require significant work, document them in `@notes/go-ownership-plan.md` 
   as a new task item — do not derail your current task.

7. Once all validation passes:
   - Update `@notes/go-ownership-plan.md`: mark your task complete, note any 
     discoveries or new tasks uncovered during implementation
   - Update `@AGENTS.md` if you learned something operational (e.g., a command 
     that works differently than documented)
   - Stage and commit: `git add -A && git commit -m "<descriptive message>"`

8. When you discover issues, immediately update `@notes/go-ownership-plan.md` with your findings using a subagent. When resolved, update and remove the item.

## Quality Standards

- Write code that is **high-performance and faithful to the specification**. 
  Do not take shortcuts for the sake of getting something working quickly. 
  The spec defines the target behavior — match it precisely.
- Prefer clarity over cleverness. Future iterations (and humans) need to 
  understand this code.
- Follow the project's established patterns. If `src/lib/` has utility 
  functions, use them rather than creating ad-hoc alternatives.
- When you face an ambiguity not covered by the spec, make the conservative 
  choice and document your decision in a code comment.

## Scope Discipline

- **Only implement the one task you selected.** Resist the urge to fix 
  adjacent things, refactor unrelated code, or "quickly add" another feature.
- If you notice bugs, missing features, or improvement opportunities outside 
  your current task, add them to `@notes/go-ownership-plan.md` and move on.
- The only exception: if your task literally cannot work without fixing 
  something else first, fix the blocker as part of your task and document it.

## Anti-Patterns to Avoid

- Do NOT create placeholder or minimal implementations. Implement fully or 
  not at all. If a task is too large for one pass, it should be split in the 
  plan — not half-implemented.
- Do NOT skip writing tests. Tests are your proof of correctness and future 
  iterations' protection against regressions.
- Do NOT mark a task complete without running validation.
- Do NOT modify specs unless you find a genuine inconsistency (and if you do, 
  document the change clearly).
- Keep `@AGENTS.md` operational only - status updates and progress notes belong in `@notes/go-ownership-plan.md`. A bloaded AGENTS.md pollutes every future loop's context.
