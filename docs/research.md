# Research

llmcomposer is a research exploration into the cross-modal musical
understanding of language models. The instrument is real and fun to play;
the reason it exists is the questions below.

!!! note "Status"
    This page frames the inquiry. A write-up of findings will be published
    as a short research post; until then, treat everything here as
    questions, not conclusions. No systematic runs have been published yet —
    see [Limitations](#limitations).

This is not new ground. Symbolic music generation, and LLMs writing ABC in
particular, have a literature — see [Related Work](related-work.md) for the
lineage, what the existing benchmarks measure, and the narrow slice this
project adds to it.

## The questions

Language models encounter music almost entirely through text: reviews,
theory textbooks, forum posts, lyrics, chord charts, and symbolic notation.
They have (mostly) never *heard* anything. Yet they can write music that,
rendered and played, is recognizably shaped by a verbal brief.

1. **How well can a model recreate music from text alone?** Given *"like
   rain on a window"* or *"a sarabande, but anxious"*, how far does the
   result get toward the target — melodically, harmonically, texturally —
   and where does it fall apart?
2. **What is the nature of the model's audio understanding?** Is it merely
   lexical — pattern-matching "rain" to arpeggios because reviews say
   "cascading" — or is there something structurally deeper: voice-leading
   that resolves, meters that breathe, instrument choices that make timbral
   sense together?
3. **Where does competence end and fluency begin?** Models are fluent in
   the *syntax* of notation. The strict validator separates syntactic
   competence (bars that add up, voices that align) from musical judgment
   (whether the piece answers the brief), so the two can be observed
   independently.

## How the system measures

Each design choice doubles as instrumentation:

- **ABC notation as the bottleneck.** ABC is compact, textual, and strict
  enough to validate — the model must commit to concrete pitches,
  durations, and voices, not vibes. Everything the model "knows" has to
  pass through this symbolic needle's eye.
- **The validator as ground truth for syntax.** A deliberately strict
  parser checks bar durations against the meter per voice and requires all
  voices of an arrangement to be equal length. Lenient parsers (pyabc2,
  music21) were rejected because they silently absorb exactly the errors
  worth counting. Every `ModelRetry` bounce is a datum: *what kinds of
  notational errors do models make, and do they fix them when told why?*
- **A typed error taxonomy.** Rejections are not prose. Each one carries an
  `ABCErrorCode` — `bar_length`, `voice_misaligned`, `missing_header`,
  `bad_meter`, `bad_key`, `bad_duration`, `fenced`, `multiple_tunes`, and
  the rest — so "which error class" is a group-by rather than a regex over
  English. The per-turn `bounces` list records, for every rejected attempt,
  the code, the message the model was shown, and the ABC it was rejected
  for.
- **Turn telemetry.** Every reply reports elapsed time, request count,
  token spend, correction count, and the prompt version that produced it —
  the cost of getting to a valid score is visible per turn, and every score
  is attributable to a specific version of the system prompt.
- **Turn recording.** Setting `LLMCOMPOSER_RECORD_DIR` appends one JSONL
  record per turn — session, model, prompt, reply, final ABC, telemetry,
  and the full bounce list — to an append-only run log. That file, not the
  screen, is the dataset.
- **Symbolic descriptors.** The validator already tokenizes every note, so
  the same pass yields cheap, standard statistics over the realized score:
  pitch-class content against the declared key, entropy and n-gram
  repetition, interval distribution, note density, per-voice range. These
  are what separate "shaped phrases" from a random walk numerically rather
  than by ear.
- **Sampling controls.** `LLMCOMPOSER_TEMPERATURE` and `LLMCOMPOSER_SEED`
  pin the model's sampling settings, and the resolved values are recorded
  with the turn. A figure without them is an anecdote.
- **A batch harness.** `evals/` holds the prompt suite (stable ids,
  categories, and machine-checkable expectations where a prompt has them),
  a sweep runner over models x prompts x repeats, and a scorer that
  aggregates first-attempt validity, bounces-to-valid, recovery rate by
  error class, and constraint satisfaction. No browser involved.
- **Live streaming.** Watching the score being written token-by-token —
  including a `bounce` event carrying the actual error code and reason
  whenever the validator rejects a draft — exposes the model's composition
  *process*, not just its output.
- **A deterministic baseline.** The offline composer is a keyword lookup
  plus a random walk that always validates. Its seed is a stable digest of
  the prompt, so the same prompt yields the same tune in any process, on
  any machine, forever. It is the null hypothesis: any claimed musicality
  in model output should be judged against what a music-blind process
  already achieves.
- **Model-agnostic by construction.** The agent binds no model, so the
  same prompts can be run across providers and model generations to compare
  their musical behavior under identical constraints. Sessions are keyed
  per browser cookie, so concurrent participants — or parallel eval
  workers — never interleave into one conversation.

## Things to try

Prompts that probe the questions above:

- **Cross-modal transfer**: *"the sound of a screen door in summer"* —
  descriptions with no musical vocabulary at all.
- **Constraint following**: *"a canon at the fifth, two voices, in D
  dorian"* — verifiable theory claims the score either satisfies or not.
- **Revision under critique**: *"keep the melody, but make the harmony
  unsettling"* — does the model revise the same tune (it receives the
  working score every turn) or start over?
- **Timbral reasoning**: *"a trio that would sound muddy — now fix it"* —
  instrument-register knowledge that only matters if the model models
  sound, not just symbols.

## Reading the field

The drifting marks behind the studio are not decoration; they are a plot of
the score as it sounds. Every note that plays seeds a small colony of
walkers, and three properties of the note decide what you see:

| Axis | Musical quantity | What it looks like |
| --- | --- | --- |
| **Hue** | pitch class | a twelve-step chroma ramp — C at the moss end, walking through the circle to B at the ochre end. Two notes an octave apart are the same colour, so a repeated pitch class stains one hue into the field |
| **Height** | register | low notes bloom near the floor, high notes near the ceiling; the vertical spread of the field is the ambitus of the piece |
| **Density** | activity | one mark per sounding note, so busy passages thicken and rests thin out. A sparse field is a piece that breathes; an even wash is a random walk |

Because hue is pitch class and not pitch, a tonal piece settles into a
narrow band of colour with excursions, while an aimless one smears across
the whole ramp — which makes the field a rough, pre-attentive read of the
same thing the pitch-class entropy descriptor measures numerically. It is
an illustration, not evidence; the numbers are in the run log.

## Limitations

Stated plainly, before any results, because they bound everything above:

- **No systematic runs have been published.** The harness to collect them
  exists; the sweeps do not. Every claim on this page is a hypothesis about
  what the instrument will show, not a finding. Nothing here has an *n*.
- **The validator measures notational well-formedness, not musical
  quality.** It checks that bars add up, that voices align, and that
  headers and directives are meaningful. It does not know whether a melody
  is good, whether a cadence lands, or whether the piece answers the brief.
  A perfectly valid score can be musically worthless, and the validator
  will pass it without comment. "Syntactic competence, separated from
  musical judgment" describes only the first half — the second half is not
  yet measured by anything in this repository.
- **The baseline is weaker than the word "baseline" suggests.** The offline
  composer is a lexical keyword table over mood words plus a seeded random
  walk within one of five hard-coded scales. It is not a trained model, it
  is not a competitive symbolic generator, and it cannot vary meter, length,
  or instrumentation — so on exactly the prompts that discriminate (a canon
  at the fifth; something in 7/8) the comparison is undefined. Treat it as
  an audible floor, not as a rival system.
- **No human evaluation has been conducted.** There is no listening panel,
  no rubric scores, no blinded pairwise comparison against the baseline.
  Musical quality is currently assessed by exactly one person, unblinded,
  with full knowledge of which arm produced which tune — which is to say,
  not assessed.
- **The measured quantities are proxies.** Bounce counts confound model
  ability with prompt quality and with the validator's own strictness; a
  prompt that never mentions voltas will produce volta errors that say more
  about the prompt than the model. Prompt versioning exists so this
  confound is at least attributable.

