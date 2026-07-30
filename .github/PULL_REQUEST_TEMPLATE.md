## What

<!-- What does this PR change, from a user's / listener's perspective? -->

## Why

<!-- Motivation and context. Link issues if applicable. -->

## How

<!-- Notable implementation decisions, trade-offs, anything reviewers should
     look at closely. -->

## Checklist

- [ ] `make ci` passes (ruff check + format, pyright strict, tests)
- [ ] New behavior is covered by a test (offline — `TestModel` /
      `FunctionModel`, never a real provider)
- [ ] Docs updated where relevant (`docs/` / `CHANGELOG.md`)
- [ ] If the studio UI changed: smoke-tested against `make run-offline` in
      a browser (no console errors)
