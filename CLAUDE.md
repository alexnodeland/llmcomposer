# CLAUDE.md

llmcomposer — compose music with an LLM copilot. A research exploration into
cross-modal musical understanding of language models (how well they recreate
music from text alone). Python 3.12+, uv, FastAPI, pydantic-ai, src/ layout.

## Commands

```sh
uv sync              # install everything (dev group included)
make ci              # the CI gate: ruff check + format --check, pyright strict, pytest
make check           # everyday gate: lint, types, tests
make test            # pytest (fully offline, <1s)
make run-offline     # run the studio with no credentials (deterministic composer)
make docs            # serve mkdocs locally
```

## Architecture

- `src/llmcomposer/models.py` — `ScoreUpdate` (reply + complete abc), the agent's typed output
- `src/llmcomposer/abc_notation.py` — strict ABC validator; the project's measurement instrument. Deliberately strict (lenient parsers were rejected); do not loosen without evidence of a false rejection
- `src/llmcomposer/agent.py` — agent definition; working score injected via `@agent.instructions`; validators raise `ModelRetry`
- `src/llmcomposer/session.py` — typed message history, safe-boundary trimming, SSE streaming via `event_stream_handler`
- `src/llmcomposer/offline.py` — deterministic `FunctionModel` composer (the no-network baseline)
- `src/llmcomposer/app.py` — FastAPI routes; model resolved from `LLMCOMPOSER_MODEL`
- `src/llmcomposer/templates/index.html` — the entire frontend in one file; no build step, no frontend tests — smoke-test in a browser against `make run-offline`

## Conventions

- Trunk-based: short-lived branches off `main` (`feat/…`, `fix/…`, `docs/…`, `chore/…`), all merges through PRs
- Conventional-commit prefixes, imperative subject, body explains why
- Tests must stay offline: `ALLOW_MODEL_REQUESTS = False` in conftest; use `TestModel`/`FunctionModel`, never a real provider
- pyright runs strict (only `reportUnusedFunction` is off, for decorator-registered handlers)
- Bar-duration math uses `Fraction` everywhere — never float
- User-visible changes go in `CHANGELOG.md` under `[Unreleased]`
- Releases: bump `pyproject.toml` version + move `[Unreleased]` → `[X.Y.Z]` on a `chore/release-X.Y.Z` branch, merge, tag `vX.Y.Z` (workflow verifies version ↔ tag ↔ changelog)
