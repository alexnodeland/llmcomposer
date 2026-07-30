# evals

The experiment kit: a fixed prompt suite, a batch runner that talks to the
composer directly (no browser, no HTTP), a scorer that turns a sweep into a
table, and a human rubric for everything the numbers cannot reach.

```
evals/
  prompts.yaml   52 prompts with stable ids, categories, and expects
  run.py         models x prompts x repeats -> JSONL, resumable
  score.py       JSONL -> markdown tables
  rubric.md      the 1-5 human rubric and the blinded A/B protocol
  results/       sweeps land here (JSONL) alongside their reports (md)
```

## running a sweep

The smoke sweep needs no credentials — it runs the whole suite against the
deterministic offline baseline in about four seconds:

```sh
make eval
```

which is:

```sh
uv run python evals/run.py --models offline --repeats 1 --out evals/results/smoke.jsonl
uv run python evals/score.py evals/results/smoke.jsonl
```

A real sweep names several models and repeats each cell, because model
output is a distribution and one sample of it is an anecdote:

```sh
uv run python evals/run.py \
    --models offline,anthropic:claude-opus-5,litellm:gpt-5 \
    --repeats 5 \
    --out evals/results/sweep-2026-07.jsonl

uv run python evals/score.py evals/results/sweep-2026-07.jsonl \
    --out evals/results/sweep-2026-07.md
```

Model names are whatever `ComposerSession` accepts: `offline`, a
pydantic-ai name such as `anthropic:claude-opus-5`, or `litellm:<model>`
routed through the proxy configured by `LITELLM_BASE_URL` /
`LITELLM_API_KEY`. Credentials come from the environment; `uv run --env-file
.env python evals/run.py …` picks up a local `.env`.

Useful flags: `--categories constraint,adversarial_meter` and
`--ids con-01,met-03` to narrow the suite, `--limit N` to cap it, and
`--concurrency N` (default 4) to control how many conversations are in
flight. Keep concurrency low on rate-limited providers.

### sampling is pinned

`--temperature` (default `0.0`) and `--seed` (default `0`) are always passed
explicitly to every session — never left to a provider default — and the
resolved pair is printed in the run header and recorded on every turn record.
A sweep that does not say what it sampled at is not a measurement.

Each cell gets its own seed, derived as a digest of `(base seed, prompt id,
repeat)`. That matters for `--repeats N`: a single seed reused across repeats
would draw the same sample every time from any provider that honours it, and
the repeats meant to characterize the output distribution would collapse to
one point. The derivation is a pure function of `--seed`, so the whole sweep
still replays exactly.

Raise `--temperature` when the question is about the *spread* of a model's
output rather than its modal behaviour, and say so next to the numbers.

**Sweeps are resumable.** Every `(model, prompt id, repeat)` already present
in the output file is skipped, so an interrupted run is restarted with the
same command. To re-run a cell, delete the file (or write to a new one).

## where results land

`evals/results/` holds one JSONL per sweep — one line per **turn**, so a
two-turn revision prompt contributes two lines. Each record carries
`run_id`, `session_id`, `turn_index`, `model_id`, `prompt_id`, `category`,
`repeat`, `prompt_version`, `prompt_sha`, `system_prompt_sha`, the resolved
`temperature` and `seed`, the user message, the reply, the ABC, the turn's
`meta`, and its `bounces` — every
validator rejection with its error code and the rejected score. A turn that
raised is recorded with `ok: false` and its error string rather than
dropped: failures are data.

`baseline-offline.jsonl` (and its report `baseline-offline.md`) is the
committed reference sweep of the whole suite against the offline baseline.
Regenerate it with:

```sh
rm evals/results/baseline-offline.jsonl
uv run python evals/run.py --models offline --repeats 1 \
    --out evals/results/baseline-offline.jsonl
uv run python evals/score.py evals/results/baseline-offline.jsonl \
    --out evals/results/baseline-offline.md
```

Name real sweeps for their date and keep the JSONL — the scorer is expected
to change, and old sweeps must stay rescoreable.

## what the metrics mean, and how to cite them

`score.py` prints four tables. When you quote a number, quote the sweep file
it came from and the number of turns behind it.

**Notational competence.** *First-attempt validity rate* is the share of
completed turns whose first ABC passed `llmcomposer.abc_notation.validate_abc`
with no retry — the model's unaided notational accuracy. *Mean bounces* is
validator rejections per turn. The bounce table breaks those down by
`ABCErrorCode`, which is the error taxonomy: what kind of mistake, not just
how many. Report validity and bounces together; a model that always
recovers on retry is a different animal from one that never errs.

**Prompt adherence.** *Expects met* is the share of individual
machine-checkable claims satisfied (meter, declared mode and tonic, voice
count, bar count, cadence, register legality); *prompts fully met* is the
stricter all-or-nothing count. Only prompts that declare `expects` count,
and only the conversation's final score is checked. These claims are about
what the model *declared and wrote*, not about whether it sounds like the
brief — nothing automatic can measure that (see `rubric.md`).

**Symbolic descriptors** come from `llmcomposer.descriptors` and are
computed on the final score of each conversation:

- *pc entropy* — Shannon entropy (bits) of the duration-weighted
  pitch-class distribution. The only anchor worth quoting is a measured one:
  the offline baseline, which is exactly a seeded random walk over one scale,
  sits at **2.11** across the 52 conversations in `baseline-offline.md`. Read
  a model's entropy as a distance from that floor, not against an absolute
  band — what a "good" value looks like is an open question until there are
  model arms to compare.
- *3-gram repetition* — share of melodic pitch 3-grams that are not first
  occurrences. Zero means nothing recurs; high values mean literal repeats.
  Neither extreme is good, which is the point: it separates shaped phrases
  from both noise and loops.
- *tonal fit (maj/min template)* — Pearson correlation of the realized
  pitch-class distribution with the Krumhansl-Kessler profile of the key the
  score *declares*. Those profiles are major and minor only, so a modal tune
  is measured against whichever of the two templates is nearest its mode —
  which means a dorian tune that collapses to plain aeolian scores *higher*
  here than one that honours its own sixth. Read it as tonal centredness, not
  as mode adherence. *key match* is the coarser version: does the
  best-fitting key share a tonic with the declared one?
- *mode match*, *diatonic adherence* — these are the "does the model mean the
  mode it names?" numbers. *diatonic adherence* is the duration-weighted share
  of pitch-class mass falling inside the declared mode's own scale; *mode
  match* is the share of scores whose characteristic degree actually sounds
  louder than the degree that would collapse the mode (dorian's natural 6,
  mixolydian's flat 7, phrygian's flat 2, lydian's sharp 4, minor's flat 6).
  Both are computed on the realized pitch content, not on the `K:` header, so
  a score that declares `K:Ddor` and then writes C major is caught. Both are
  blank when the header names no tonic.
- *mean interval*, *leap rate* — melodic interval size in semitones, and
  the share wider than a major third, over the first (melody) voice.
- *rest fraction* — share of written duration spent resting; the crude
  version of "melodies should breathe".
- *scores with range violations* — scores in which some voice is written
  outside its declared General MIDI instrument's playable range. This is
  the timbral-reasoning probe as a number: does the model know what a
  piccolo can do?

**Cadence distribution** classifies each score's close (authentic, plagal,
half, deceptive, none) from the bass motion over the final two bars. It is
deliberately best-effort and should be read as a distribution, never quoted
for a single tune.

### caveats to carry with any number

- The validator exempts some bars from duration checking; a validity rate
  is therefore an upper bound on notational correctness.
- Descriptors are computed on ABC as written. They know nothing about
  performance, dynamics, or how anything actually sounds.
- The offline arm is a lexical keyword table plus a random walk over one
  scale in 4/4. It is a floor, not a competitor, and it *cannot* satisfy
  meter or bar-count expects by construction — that is what makes it a
  useful null.
- Every automatic number here is about syntax and statistics. Musical
  quality is only ever measured by `rubric.md`, under blinding.
