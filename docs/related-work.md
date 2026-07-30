# Related Work

llmcomposer sits in a well-populated field. This page places it there
honestly: what came before, what those systems measure, and what — narrowly
— is left over for this one.

## 1. Symbolic music generation from sequence models

The idea of learning music as a *token stream* predates language models
being good at it, and ABC notation was one of the first corpora used.

- **folk-rnn** — Sturm et al. (2016), *Music transcription
  modelling and composition using deep learning*
  ([arXiv:1604.08723](https://arxiv.org/abs/1604.08723)). An LSTM trained
  directly on ~23k ABC transcriptions from thesession.org. It established
  the exact setup this project borrows — the tune as text, the transcription
  as the unit of generation — and, importantly, it was evaluated partly by
  human musicians playing the output, which remains rare.
- **Music Transformer** — Huang et al. (2018), *Music Transformer:
  Generating Music with Long-Term Structure*
  ([arXiv:1809.04281](https://arxiv.org/abs/1809.04281); ICLR 2019).
  Relative self-attention made minute-scale structure tractable in
  performance MIDI. The lesson that carries over: long-range coherence is
  the hard part, and it is the part a bar-level validator cannot see.
- **MuseNet** — Payne (2019), OpenAI technical blog post. A 72-layer
  Sparse Transformer over MIDI tokens conditioned on composer and
  instrumentation. Never formally published or evaluated, but it is the
  first widely-seen demonstration that a general-purpose LM architecture,
  unmodified, handles multi-instrument symbolic music.
- **MMM (Multi-Track Music Machine)** — Ens et al. (2020), *MMM:
  Exploring Conditional Multi-Track Music Generation with the Transformer*
  ([arXiv:2008.06048](https://arxiv.org/abs/2008.06048)). Notable here for
  its *interaction* model: a token layout designed so a user can fix some
  tracks and regenerate others. That is the closest prior art to
  llmcomposer's revision loop, though it is regeneration by masking rather
  than by natural-language critique.

## 2. Large language models writing ABC specifically

This is the immediate neighbourhood. Several groups have already asked
whether a text LM can write valid, musical ABC — and built benchmarks for
it.

- **ChatMusician** — Yuan et al. (2024), *ChatMusician: Understanding and Generating Music Intrinsically with LLM*
  ([arXiv:2402.16153](https://arxiv.org/abs/2402.16153); Findings of ACL
  2024). The most direct precedent: LLaMA2 continually pretrained on ABC
  notation, with music treated as a second language rather than as a
  modality bolted on through an encoder. It ships **MusicTheoryBench**, a
  college-level benchmark of music-knowledge and reasoning questions, and
  reports that general LLMs score near chance on the reasoning half. Two
  things to note about the eval: it is single-shot Q&A plus generation
  quality, and it measures theory *knowledge*, not whether a generated
  score is well-formed under a strict parser.
- **MuPT** — Qu et al. (2024), *MuPT: A Generative
  Symbolic Music Pretrained Transformer*
  ([arXiv:2404.06393](https://arxiv.org/abs/2404.06393)). Scales the
  symbolic-LM recipe and, most relevantly, introduces **SMT-ABC** — a
  synchronized multi-track ABC notation — after finding that naive ABC
  serialization of multi-voice music makes bar alignment across voices hard
  for a model to maintain. llmcomposer's multi-voice alignment check is
  measuring precisely the failure mode MuPT re-engineered the notation to
  avoid.
- **NotaGen** — Wang et al. (2025), *NotaGen: Advancing Musicality in Symbolic Music Generation with Large
  Language Model Training Paradigms*
  ([arXiv:2502.18008](https://arxiv.org/abs/2502.18008)). Applies the full
  LLM pipeline — pretrain, fine-tune, then reinforcement learning from a
  musicality preference signal — to interleaved ABC for classical scores.
  It is the state of the art for *quality* in this format, and its
  contribution is the preference-tuning stage: evidence that well-formedness
  and musicality are separable objectives requiring separate optimization.
- **GPT-4 anecdata** — Bubeck et al. (2023), *Sparks of Artificial General Intelligence: Early experiments with GPT-4*
  ([arXiv:2303.12712](https://arxiv.org/abs/2303.12712)). The music section
  asks GPT-4 to produce and manipulate ABC tunes and reports, informally,
  that it produces coherent single-line melodies, handles simple
  transformations, and does not manage genuine harmonic or motivic
  development. It is explicitly anecdotal — a handful of examples, no
  metrics, no repeats — which is exactly the gap a small, honest,
  instrumented harness can fill.

## 3. The text-to-audio contrast

The other branch of "text in, music out" skips notation entirely.

- **MusicLM** — Agostinelli et al. (2023), *MusicLM:
  Generating Music From Text*
  ([arXiv:2301.11325](https://arxiv.org/abs/2301.11325)). Hierarchical
  sequence modelling over audio tokens, with **MusicCaps** — 5.5k
  expert-written text/music pairs — released as the evaluation set.
- **MusicGen** — Copet et al. (2023), *Simple and
  Controllable Music Generation*
  ([arXiv:2306.05284](https://arxiv.org/abs/2306.05284); NeurIPS 2023). A
  single-stage transformer over interleaved EnCodec tokens, with text and
  melody conditioning; the open-weights baseline most people reach for.
- **Stable Audio** — Evans et al. (2024), *Fast Timing-Conditioned
  Latent Audio Diffusion*
  ([arXiv:2402.04824](https://arxiv.org/abs/2402.04824); ICML 2024).
  Latent diffusion with explicit timing conditioning, generating full-length
  stereo tracks.

**Why the symbolic bottleneck is the interesting control.** Audio models
can be convincing without ever committing to a pitch. A waveform can carry
plausible timbre, groove, and production while its harmonic content is
smeared, ambiguous, or simply wrong — and a listener will often not notice,
because the surface cues that make music *sound* real are largely timbral.
Notation removes that escape route. To emit ABC the model must name every
pitch, every duration, every voice; the result can be parsed, checked
against a meter, transposed, and disagreed with. Any musical understanding
that survives the round trip through text is understanding of *structure*,
not of texture. That is the property this project is built around, and it
is why the interesting failures here are arithmetic and voice-leading
rather than artefacts.

## What is different here

Every system above is evaluated single-shot: one prompt in, one artifact
out, scored by a benchmark or a listening panel. None of them measure what
happens on turn three — whether a model, told *"the second voice is a bar
short"* or *"keep the melody but make the harmony unsettling"*, actually
repairs the specific thing named while preserving the rest. llmcomposer's
narrow contribution is that setting: **interactive multi-turn revision
under critique**, paired with a deliberately strict validator that turns
every rejection into a typed, machine-readable error code rather than a
pass/fail bit. What accumulates is a per-turn record of *which* notational
errors a model makes, *whether* it fixes the one it was told about, and
what it breaks in the process. That data does not exist in the benchmarks
above, because their protocols have no turn two. This project is an
instrument for producing it; the findings are still forthcoming (see
[Limitations](research.md#limitations)).
