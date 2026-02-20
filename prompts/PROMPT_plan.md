# MODE: PLANNING ONLY — DO NOT WRITE ANY IMPLEMENTATION CODE

## Orientation

0a. Study all files in `specs/*` with up to 250 parallel subagents to understand the complete project requirements.
    Use subagents to read specs in parallel — do not load all spec content 
    into the main context.
0b. Study `src/lib/*` with up to 250 parallel subagents to understand shared utilities & components.
0c. Study `@IMPLEMENTATION_PLAN.md` if it exists (it may be stale or incorrect).
0d. The application source code is in `src/*`. Use subagents to study 
    existing implementations — do not read entire source files into main context.

## Your Task

1. Perform a comprehensive gap analysis using subagents. For each spec in `specs/*`, 
   spawn a subagent to compare that spec's requirements against the corresponding 
   source code in `src/*`. Each subagent should report back:
   - What is fully implemented and passing tests
   - What is partially implemented (stubs, placeholders, TODOs, minimal implementations)
   - What is completely missing
   - What is implemented but inconsistent with the spec

2. Before concluding anything is missing, **subagents must search the codebase 
   thoroughly using at least 3 different search strategies** (grep exact name, 
   grep partial/related terms, search file tree). Do not assume a feature is 
   unimplemented — confirm with evidence. This is the single most important 
   instruction. Incorrect assumptions about missing code are the #1 failure mode.

3. Use subagents to search explicitly for: `TODO`, `FIXME`, `HACK`, `PLACEHOLDER`, 
   `stub`, `not implemented`, `minimal implementation`, and any skipped or 
   ignored tests.

4. If `@IMPLEMENTATION_PLAN.md` already exists and appears substantially current 
   (most items have accurate status relative to what subagents found), update it 
   incrementally — do not regenerate from scratch. If the plan is badly stale or 
   contradicts reality in multiple places, regenerate it entirely.

5. Create or update `@IMPLEMENTATION_PLAN.md` as a prioritized bullet-point list. 
   Each item should include:
   - A short unique task ID (e.g., TASK-001)
   - Brief description of what needs to be done
   - Which spec it traces to (e.g., "per specs/auth-system.md §3.2")
   - Current state (missing / stub / partial / inconsistent)
   - Priority rationale (dependencies, blocking status, complexity)
   - Acceptance criteria summary (what tests must pass for this to be done)

6. Sort by priority: foundational/blocking items first, then features that 
   unlock other features, then independent features, then polish.

7. If the plan has no remaining incomplete items after analysis, state 
   "PLAN COMPLETE — all spec requirements appear implemented" at the top 
   of `@IMPLEMENTATION_PLAN.md`.

8. Do NOT assume functionality is missing; confirm with code search first. Treat `src/lib` as the project's standard library for shared utilities and components. Prefer consolidated, idiomatic implementations there over ad-hoc copies.

## Hard Rules

- **DO NOT write, modify, or create any source code files.**
- **DO NOT create test files.**
- **DO NOT run build or test commands** (you're only reading and analyzing).
- You may ONLY create/update: `IMPLEMENTATION_PLAN.md` and `specs/*.md` 
  (if you discover spec gaps or inconsistencies).
- If you create a new spec element then document the plan to implement it in `@IMPLEMENTATION_PLAN.md` using a subagent.
- If you find spec inconsistencies, note them clearly in the plan AND 
  update the spec with a `[PLANNING NOTE]` annotation explaining the issue.
- Use subagents for ALL file reading and codebase searching. Keep the main 
  context focused on synthesizing subagent reports into the plan.

## Output

Your only deliverable is `@IMPLEMENTATION_PLAN.md`. Think carefully. 
Accuracy here saves dozens of wasted build iterations later.
