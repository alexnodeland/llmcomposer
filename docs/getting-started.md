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
| `LLMCOMPOSER_MODEL` | `anthropic:claude-opus-5` | Any pydantic-ai model name, `litellm:<model>`, or `offline` |
| `LLMCOMPOSER_MODELS` | — | Comma-separated list offered in the studio's model selector (`offline` is always included) |
| `ANTHROPIC_API_KEY` | — | Credentials for the default Anthropic model |
| `LITELLM_BASE_URL` | — | Your LiteLLM proxy, for `litellm:` model names |
| `LITELLM_API_KEY` | — | Key for the LiteLLM proxy |
| `LOGFIRE_TOKEN` | — | Enables Logfire tracing when set |

`make run` loads `.env` automatically when one exists (see `.env.example`).

Because the agent binds no model, any provider pydantic-ai supports works —
set `LLMCOMPOSER_MODEL` to e.g. `openai:gpt-5` with the matching credentials
in the environment, or point `litellm:<model>` names at a
[LiteLLM proxy](https://docs.litellm.ai/) to reach many providers with one
key. Comparing providers on the same prompts is part of the point — the
studio's model selector switches the composer mid-collaboration, keeping
the conversation and working score, so different models can revise the
same tune back-to-back. See [Research](research.md).
