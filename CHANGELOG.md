# Changelog

All notable changes to llmcomposer are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0.

## [Unreleased]

### Added

- **Validator error taxonomy** — every `ABCValidationError` now carries a
  machine-readable `ABCErrorCode`; bounces are recorded per attempt as
  structured `Bounce` records (code, message, rejected score) on `TurnMeta`
  and streamed to the studio as dedicated SSE events with the reason shown.
- **Semantic header validation** — implausible `K:` keys, malformed `Q:`
  tempi, multiple `X:` tunes, zero-duration notes, and out-of-range
  `%%MIDI program` values now bounce with actionable messages.
- **Turn recording** — set `LLMCOMPOSER_RECORD_DIR` to append one JSONL
  record per turn (prompt, reply, score, bounces, usage, sampling settings,
  prompt version): the dataset behind the research questions.
- **Sampling controls** — `LLMCOMPOSER_TEMPERATURE` / `LLMCOMPOSER_SEED`
  are passed to the model and recorded on every turn; the system prompt is
  versioned (`PROMPT_VERSION`).
- **Symbolic descriptors** (`llmcomposer.describe`) — Krumhansl–Schmuckler
  key inference and declared-key correlation, pitch-class entropy, n-gram
  repetition, interval statistics, rest fraction, per-voice ranges with
  GM-instrument range violations, voice crossings, parallel fifths/octaves,
  and cadence classification, computed from the validator's own tokenizer.
- **Eval harness** (`evals/`) — a 52-prompt suite with stable ids and
  machine-checkable expectations, an offline-capable resumable batch
  runner, a scorer that renders per-model markdown tables, a human-eval
  rubric with a blinded A/B protocol, and `make eval` as a smoke sweep.
- **Per-session studio** — sessions are keyed by cookie (independent
  conversations per browser), with `POST /score` to restore a previous
  version as the authoritative working score.
- **Related work & citations** — `docs/related-work.md` positions the
  project against folk-rnn, Music Transformer, ChatMusician, MuPT, NotaGen,
  MusicLM/MusicGen and friends; `CITATION.cff` added; docs/research.md
  gains an explicit Limitations section.

### Changed

- **The validator is now symmetric.** False accepts closed: the global
  repeat-mark exemption is gone (short bars are only tolerated at genuine
  section edges), no bar may ever exceed the meter, inline `[M:]`/`[L:]`
  changes are applied rather than skipped, and unmarked `(5`/`(7`/`(9`
  tuplets use the ABC 2.2 meter-dependent default. False rejects closed:
  `[|:` and `:|]` barlines, multi-measure rests (`Z2`, counted as whole
  bars in voice alignment too), run-on music lines, trailing `%` remarks,
  and additive meters like `(2+3+2)/8` all validate; a corpus of real folk
  tunes guards against regressions.
- **The composer prompt teaches real craft** — form (with repeat syntax),
  cadence types, voice-leading constraints, clefs, `sname=` abbreviations,
  `%%score` bracketing, concert-pitch instrument ranges, and modal keys,
  restructured into named sections and versioned as `composer-v2`.
- **The offline baseline is honest and reproducible** — seeded from a
  stable digest of the prompt (byte-identical across processes; golden
  test), scored with proper clefs/short names/brackets, register-shifted
  so every voice stays in range, and its replies now say plainly that it
  is a music-blind control instead of claiming to have "heard" the prompt.
- **The studio was redesigned end-to-end** — the score fills the stage and
  never clips mid-staff; a real type scale and an accessible neutral ramp
  (all instructional text ≥ 4.5:1); play/stop as the primary transport
  action with start-over demoted behind a confirm; clickable example
  prompts; the generative field accumulates (with a legend and PNG export)
  instead of erasing itself; abcjs is vendored and served locally;
  playback keeps one primed synth (gapless loop, tempo changes without
  restarting, real-bpm label); chord-symbol accompaniment is off so
  playback contains only what is notated; exports carry provenance
  headers and honor the audition tempo; screen readers get a described
  score and calm announcements; motion respects `prefers-reduced-motion`;
  the mobile layout keeps the composer bar reachable and renders notation
  legibly in a full-screen sheet.
- **Errors speak the house voice** — mid-stream failures surface with a
  retry (raw detail behind "what actually happened"), the typed prompt is
  restored on failure, and baseline mode is labeled as such in the chat.
- **Second panel pass** — the prompt's instrument-range table is generated
  from the same `GM_RANGES` the descriptors score against (they can no
  longer drift; prompt bumped to `composer-v3` with a pinned content
  digest); `%%score` is emitted before the voice declarations so the
  ensemble bracket actually spans the system; the baseline harp reads bass
  clef; pickup and final bars must now sum to one full bar and voice
  totals must agree (closing the last duration-check exemptions); the
  parallel-motion detector no longer reads across volta seams; descriptors
  gain `diatonic_adherence` and `mode_match` so mode-following is measured
  (the Krumhansl number is relabeled as major/minor template fit); eval
  sweeps pin `--temperature`/`--seed` with per-cell seed derivation; the
  engraving is set in a chosen serif with ledger lines at staff weight;
  keyboard focus is designed (`:focus-visible`), Space no longer swallows
  button activation, version pills track the authoritative take, a
  user-initiated stop is not an error, and the baseline describes itself
  accurately as a keyword control.

- **Model selector** in the studio: `LLMCOMPOSER_MODELS` configures a list
  of models, a dropdown switches the composer mid-collaboration (the
  conversation and working score carry over), and `litellm:<model>` names
  route through a LiteLLM proxy (`LITELLM_BASE_URL` / `LITELLM_API_KEY`) —
  one key for GLM, MiniMax, Claude, GPT, Gemini, and more. `make run` now
  loads `.env` automatically.

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
