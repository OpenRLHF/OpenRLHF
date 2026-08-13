# OpenRLHF — working rules for AI assistants

## Code standards (hard rules, not preferences)

The priority order is explicit: **human readability comes first; coding-agent
traceability is the minimum gate.** A human should understand code in one pass,
and an agent must be able to trace a feature from CLI flag to executed branch,
tensor/record, metric, and test without reconstructing hidden control flow.

1. Over-complex or hard-to-follow code is a **bug**, not a style issue.
2. Reduce complexity and line count — prefer deleting code over adding it.
3. No over-encapsulation: a helper needs 3+ real call sites AND nontrivial
   logic; never wrap trivial code or add classes/files for one call site.
4. Do NOT remove features, performance knobs, or observability in the name
   of simplicity — knobs default ON stay ON.
5. A "bug" that cannot trigger under the shipped recipes is not worth fixing.

Details: `.claude/skills/simplicity-first` (invoke before any code change).

## Comments

Concise "why" only, 2-4 lines, written for an external reader: no job ids,
commit hashes, single-run metrics, or internal cluster paths; keep upstream
issue/PR links.

## Workflow

- Reviews report findings only; fixes ship as one minimal diff per issue
  after approval.
- Verify with `python -m compileall -q openrlhf examples tests` and
  `python -m pytest -q`; shell scripts with `bash -n`.
