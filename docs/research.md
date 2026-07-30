# Research

llmcomposer is a research exploration into the cross-modal musical
understanding of language models. The instrument is real and fun to play;
the reason it exists is the questions below.

!!! note "Status"
    This page frames the inquiry. A write-up of findings will be published
    as a short research post; until then, treat everything here as
    questions, not conclusions.

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
- **Turn telemetry.** Every reply reports elapsed time, request count,
  token spend, and correction count — the cost of getting to a valid score
  is visible per turn.
- **Live streaming.** Watching the score being written token-by-token
  (including mid-stream validator bounces and rewrites) exposes the
  model's composition *process*, not just its output.
- **A deterministic baseline.** The offline composer is a random-walk
  melody generator that always validates. It is the null hypothesis: any
  claimed musicality in model output should be judged against what a
  music-blind process already achieves.
- **Model-agnostic by construction.** The agent binds no model, so the
  same prompts can be run across providers and model generations to compare
  their musical behavior under identical constraints.

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
