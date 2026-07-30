# Getting Started

## Install

llmcomposer uses [uv](https://docs.astral.sh/uv/). A single clone builds:

```sh
git clone https://github.com/alexnodeland/llmcomposer.git
cd llmcomposer
uv sync
```

The pinned Python (3.12) is installed automatically by uv via
`.python-version`.

## Run

```sh
# with a real model (default: anthropic:claude-opus-5)
export ANTHROPIC_API_KEY=sk-ant-...
uv run llmcomposer

# or without any credentials, using the built-in offline composer
LLMCOMPOSER_MODEL=offline uv run llmcomposer
```

Then open <http://127.0.0.1:8000>. Chat on the left; the score appears on
the right. Ask for *"a trio of flute, harp and cello"* and the agent writes
a multi-voice arrangement, choosing a General MIDI patch per instrument with
`%%MIDI program` directives.

## The studio

The player highlights each note across all staves as it sounds, with a
progress bar, loop mode, a tempo slider, an instrument legend, per-score
version history, a raw-ABC source drawer with copy, and `.abc` / `.midi`
export.

Composition streams live — you watch the score being written ("writing the
score · 512 chars"), and if the validator bounces it you see the rewrite
happen. Every composer reply carries turn telemetry — elapsed time, request
count, token spend, and correction count.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLMCOMPOSER_MODEL` | `anthropic:claude-opus-5` | Any pydantic-ai model name, or `offline` |
| `ANTHROPIC_API_KEY` | — | Credentials for the default Anthropic model |
| `LOGFIRE_TOKEN` | — | Enables Logfire tracing when set |

Because the agent binds no model, any provider pydantic-ai supports works —
set `LLMCOMPOSER_MODEL` to e.g. `openai:gpt-5` with the matching credentials
in the environment. Comparing providers on the same prompts is part of the
point; see [Research](research.md).
