# Task: Create Branch, Commit Changes, Push to Remote, and Open Pull Request

## Original User Prompt (verbatim)

> "create a branch, commit, push and open PR"

## Clarifications

- Target branch for PR: `main` (upstream default branch).
- Source branch: `feature/trading-bot-core`.
- Changes to be committed: Full trading bot implementation scaffolded from `AGENT.md`, including configuration, data connectors with Linux/Wine bridge support, technical analysis and pattern recognition, ML features & models, signal engine, risk management, execution with broker symbol resolution, event-driven backtesting, live trading orchestrator, setup scripts, and unit tests.
- Pre-commit verification: All 15 unit tests verified passing with pytest.

## Implementation Intent

1. Record task prompt in `prompts/` as required by `AGENT.md` rule 6.
2. Update `.gitignore` to ignore machine-local config (`scripts/.wine_python_path`) and track `.env.example` with sanitized placeholder credentials.
3. Create and switch to a clean feature branch: `feature/trading-bot-core` based on `origin/main`.
4. Stage all project files, configuration, tests, and documentation.
5. Commit all changes with a descriptive conventional commit message.
6. Push the branch to `origin`.
7. Create a comprehensive Pull Request targeting `main` on GitHub using the GitHub CLI / API.
