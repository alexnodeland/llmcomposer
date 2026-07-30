# llmcomposer

Compose music with an LLM copilot. You describe a feeling — *like rain on a
window* — and a [pydantic-ai](https://ai.pydantic.dev) agent writes and
revises a tune in ABC notation. The web app renders it as sheet music,
plays it through the abcjs synth, and grows a generative field from the
notes as they sound.

## Quick start

```sh
uv sync

# with a real model (default: anthropic:claude-opus-5)
export ANTHROPIC_API_KEY=sk-ant-...
uv run llmcomposer

# or without any credentials, using the built-in offline composer
LLMCOMPOSER_MODEL=offline uv run llmcomposer
```

Then open <http://127.0.0.1:8000>. Chat on the left; the score appears on
the right. Ask for *"a trio of flute, harp and cello"* and the agent writes a
multi-voice arrangement, choosing a General MIDI patch per instrument with
`%%MIDI program` directives.

The player highlights each note across all staves as it sounds, with a
progress bar, loop mode, a tempo slider, an instrument legend, per-score
version history, a raw-ABC source drawer with copy, and `.abc` / `.midi`
export. Composition streams live — you watch the score being written
("writing the score · 512 chars"), and if the validator bounces it you see
the rewrite happen. Every composer reply carries turn telemetry — elapsed
time, request count, token spend, and correction count.

## How it's built

The interesting parts are how deeply it leans on pydantic-ai:

- **Typed structured output** — the agent's `output_type` is `ScoreUpdate`
  (`reply` + complete `abc`), so every turn is validated Pydantic, never
  free text ([`models.py`](src/llmcomposer/models.py)).
- **Output validators with `ModelRetry`** — a strict ABC parser
  ([`abc_notation.py`](src/llmcomposer/abc_notation.py)) tokenizes the tune
  body, checks bar durations against the meter per voice, and verifies all
  voices of an arrangement are the same length; malformed scores bounce
  back to the model with an actionable reason and it retries. (pyabc2 and
  music21 were evaluated and rejected: both are lenient parsers that
  silently swallow exactly the errors the retry loop needs to catch.)
- **Dynamic instructions** — the working score is injected each turn via
  `@agent.instructions`, so the model always revises the same tune
  ([`agent.py`](src/llmcomposer/agent.py)).
- **Typed message history + history processors** — `ComposerSession` keeps
  the `ModelMessage` history and a processor trims old runs at safe
  boundaries ([`session.py`](src/llmcomposer/session.py)).
- **Model-agnostic by construction** — the agent binds no model; the app
  resolves one from `LLMCOMPOSER_MODEL`, tests use `TestModel` /
  `FunctionModel` with `ALLOW_MODEL_REQUESTS = False`, and
  [`offline.py`](src/llmcomposer/offline.py) is a `FunctionModel` that
  composes deterministic tunes so the whole stack runs with no network.
- **Event-stream handler** — `ComposerSession.send_stream` passes an
  `event_stream_handler` to `agent.run`, translating pydantic-ai's
  `PartStartEvent`/`PartDeltaEvent` stream into server-sent events, so the
  UI shows the score being written token-by-token and every validator
  bounce, while retries keep their normal non-streaming semantics.
- **Logfire instrumentation** — `logfire.instrument_pydantic_ai()` is wired
  in and activates when `LOGFIRE_TOKEN` is present.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLMCOMPOSER_MODEL` | `anthropic:claude-opus-5` | Any pydantic-ai model name, or `offline` |
| `ANTHROPIC_API_KEY` | — | Credentials for the default Anthropic model |
| `LOGFIRE_TOKEN` | — | Enables Logfire tracing when set |

## Development

```sh
uv run pytest        # no network, no API key needed
uv run ruff check src tests
uv run pyright src
```
