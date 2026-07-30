# Development

## Toolchain

Everything runs through [uv](https://docs.astral.sh/uv/) and `make`:

```sh
uv sync          # install everything, including dev tools
make ci          # exactly what CI enforces: ruff check + format, pyright strict, pytest
make check       # the everyday gate: lint, types, tests
make run-offline # start the studio with no credentials
make docs        # serve this documentation locally
```

Optional but recommended:

```sh
uv tool install pre-commit && pre-commit install
```

The hooks mirror CI (ruff, ruff-format, pyright), so a clean commit means
CI will not fail on lint or drift.

## Quality bar

- **`make ci` green before pushing.** It is byte-for-byte the CI pipeline.
- **pyright runs strict.** The one relaxation is `reportUnusedFunction`,
  because route handlers and agent hooks are registered by decorator.
- **Tests are offline by construction.** `ALLOW_MODEL_REQUESTS = False` is
  set in `conftest.py`; use `TestModel` / `FunctionModel`, never a real
  provider. The suite runs in well under a second.
- **The validator's strictness is a feature.** It is the project's
  measurement instrument (see [Research](research.md)). Loosening it
  requires showing what real error it was rejecting incorrectly.

## Sharp edges

- **The frontend is one file.** `templates/index.html` contains the entire
  studio — markup, styles, and scripts. Smoke-test UI changes in a real
  browser against `make run-offline` with the console open; there is no
  frontend build step and no frontend test suite.
- **Streaming and retries interact subtly.** `send_stream` streams part
  events out as SSE, but validator retries happen inside the run with
  normal non-streaming semantics — the UI sees a bounce notification, then
  a fresh stream. Keep that boundary intact when touching `session.py`.
- **Bar-duration math uses `Fraction` throughout.** Floating point would
  quietly break tuplet and broken-rhythm checks.

## Releasing

Trunk-based: work lands on `main` through PRs. To cut a release:

1. Branch `chore/release-X.Y.Z`; bump `version` in `pyproject.toml`; move
   the `[Unreleased]` section of `CHANGELOG.md` to `[X.Y.Z] — date`.
2. PR, merge, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The release workflow re-verifies (version ↔ tag ↔ changelog, full test
   suite), builds the sdist and wheel, and publishes a GitHub Release with
   generated notes.
