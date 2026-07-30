# Changelog

All notable changes to llmcomposer are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0.

## [Unreleased]

### Changed

- **pydantic-ai 1.0 → 2.x**: `output_retries` became `retries={"output": …}`,
  `history_processors` became the `ProcessHistory` capability, and
  `result.usage` is now a property. This also unpinned opentelemetry and let
  logfire float to current — the `[tool.uv]` constraint holding them back
  for pydantic-ai 1.x is gone.

## [0.1.0] — 2026-07-30

The first tagged release: a working instrument for the research question.
You describe music in words; a language model writes and revises it in ABC
notation; a strict validator and turn telemetry measure how well it did.

### Added

- **Composer agent** on pydantic-ai with typed structured output
  (`ScoreUpdate`), dynamic per-turn instructions carrying the working score,
  and typed message history with safe-boundary trimming.
- **Strict ABC validator** — tokenizes the tune body, checks bar durations
  against the meter per voice, and verifies all voices of an arrangement are
  the same length; malformed scores bounce back to the model via
  `ModelRetry` with an actionable reason. (pyabc2 and music21 were evaluated
  and rejected: both are lenient parsers that silently swallow exactly the
  errors the retry loop needs to catch.)
- **Collaborative web studio** — FastAPI + abcjs: rendered sheet music,
  synth playback with per-note highlighting across staves, loop mode, tempo
  slider, instrument legend, per-score version history, raw-ABC drawer, and
  `.abc` / `.midi` export.
- **Live composition streaming** — an `event_stream_handler` translates
  pydantic-ai part events into server-sent events, so the score is watched
  being written token-by-token, including validator bounces and rewrites.
- **Offline composer** — a `FunctionModel` that composes deterministic
  tunes, so the whole stack (and the test suite) runs with no network and
  no credentials.
- **Turn telemetry** — elapsed time, request count, token spend, and
  correction count on every composer reply.
- Repository adopted the house OSS standards: MIT license, Makefile
  (`make ci` = the CI gate), CI with separate lint/type-check/test/docs
  jobs, pyright strict, pre-commit hooks, contributing + issue/PR
  templates, mkdocs-material docs site, release workflow, and this
  changelog.
