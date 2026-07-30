# Contributing to llmcomposer

Thanks for your interest in contributing! This document covers the workflow
and the quality bar. It follows the same conventions as
[quiver](https://github.com/alexnodeland/quiver),
[ricercar](https://github.com/alexnodeland/ricercar), and the rest of the
ecosystem.

llmcomposer is a research exploration — how well can a language model
recreate music from text alone, and what is the nature of its audio
understanding beyond lexical description? Contributions that sharpen the
measurement (validator strictness, telemetry, offline baselines) are as
welcome as features.

## Code of Conduct

This project follows the
[Contributor Covenant](../CODE_OF_CONDUCT.md).
Be respectful and constructive.

## Getting Started

1. **Fork and clone** — a single clone builds; there are no private
   dependencies.
2. **Install [uv](https://docs.astral.sh/uv/)**, then `uv sync` — that's the
   whole toolchain (Python version included, via `.python-version`).
3. **Verify your setup**:
   ```bash
   make check
   ```

## Development Workflow

- Branch from `main` with a descriptive name
  (`feat/percussion-voices`, `fix/tuplet-durations`, `docs/…`).
- Run `make ci` locally before pushing — it is exactly what CI enforces
  (ruff check + format, pyright strict, the test suite).
- The app runs with no credentials: `make run-offline` starts the studio
  against the deterministic offline composer. Use it to smoke-test UI
  changes with the browser console open.

## Pull Request Process

1. Keep PRs focused; separate refactors from behavior changes.
2. Update docs alongside code: `docs/` for user-facing behavior,
   [`docs/development.md`](../docs/development.md) for workflow/sharp
   edges, `CHANGELOG.md` under `[Unreleased]` for anything user-visible.
3. Add or extend a test for new behavior. Tests must stay offline:
   `ALLOW_MODEL_REQUESTS = False` is set globally, so use `TestModel` /
   `FunctionModel` — never a real provider.
4. CI must be green.

## Commit Messages

Conventional-commit style prefixes are used loosely
(`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`) with an imperative
subject line and a body that explains *why*.

## Testing Guidelines

- `make test` runs the whole suite in well under a second — there is no
  excuse not to run it.
- The ABC validator is the project's ground truth; if you loosen it, you
  must show what real error it was rejecting incorrectly. Leniency defeats
  the retry loop (see the pyabc2/music21 note in the README).
- UI changes should be exercised in a real browser against
  `make run-offline`, with the console open.
