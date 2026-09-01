# Task: Add prompt-logging rule to AGENT.md

## Original Prompt (verbatim)

> "create on more rules in agents before implementing any task write the prompt you used in prompt folder"

## Clarifications

None required — intent was clear.

## Implementation Intent

1. Add a new non-negotiable ground rule (Rule 6) to **AGENT.md § 2** stating:
   - Before implementing any task, the agent must save the user prompt + context
     as a Markdown file in the `prompts/` folder.
   - File naming convention: `YYYY-MM-DD_HH-MM_<short-slug>.md`.
   - Each file must include: original prompt, clarifications, implementation intent.
2. Add `prompts/` to the project architecture tree in **AGENT.md § 3**.
3. Create the `prompts/` folder with a `README.md` explaining the convention.
4. Create this file as the first logged prompt entry.

## Files Changed

- `AGENT.md` — Rule 6 added to § 2; `prompts/` added to § 3 architecture tree.
- `prompts/README.md` — created (explains convention).
- `prompts/2026-08-31_01-10_add-prompt-logging-rule.md` — this file (first log entry).
