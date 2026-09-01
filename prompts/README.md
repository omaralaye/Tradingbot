# Prompts Log

This folder contains one Markdown file per task implemented in this project.

## Rule (from AGENT.md § 2, Rule 6)

> **Before writing any code or making any file changes**, the agent must save
> the exact user prompt (and any clarifying context) as a `.md` file here.

## File naming convention

```
YYYY-MM-DD_HH-MM_<short-slug>.md
```

**Example:** `2026-08-31_01-10_add-prompt-logging-rule.md`

## Each file must contain

1. **Original prompt** — verbatim copy of what the user asked.
2. **Clarifications** — any questions asked by the agent and the user's answers (if any).
3. **Implementation intent** — a brief statement of what the agent plans to do.

## Purpose

This log provides a full, auditable trail of every decision made in the project:
- *Why* a change was requested.
- *What* the agent understood the task to be.
- *What* was built as a result.

Never skip creating this file, even for minor changes.
