# human rubric and blinded A/B protocol

The numbers in `evals/score.py` say what a model *did* — whether the ABC
parsed, whether it honoured the meter you asked for, how much its pitch
content looks like the key it declared. None of them say whether the music
is any good. That is what this rubric is for, and it is the only place in
the repo where musical quality is assessed at all.

Rate the **score**, not the chat reply. Listen at least once (abcjs synth,
or any ABC player) and look at the notation once. Do not read the prompt's
category, the model name, or the reply text before rating.

## the five items

Each item is 1-5. Anchors are given for 1, 3, and 5; use 2 and 4 for the
gaps. Half points are not allowed. Missing items are not allowed — if you
cannot tell, rate 3 and flag the item in your notes.

### 1. brief fidelity — *does this answer the request?*

| | |
| --- | --- |
| 1 | Could have been written for any other prompt. Nothing in the score responds to what was asked. |
| 3 | One clear connection (a mood, a tempo, one named instrument) and no more. Reads as a generic tune with a gesture toward the brief. |
| 5 | Several independent choices — register, harmony, rhythm, form, instrumentation — each traceable to something in the brief. A reader shown the score could guess the prompt. |

### 2. melodic shape — *is there a line, or is there a walk?*

| | |
| --- | --- |
| 1 | Undirected motion. No phrase boundaries, no repeated material, no arrival. Notes in an order. |
| 3 | Phrases are identifiable and mostly balanced, but the shape is generic: even lengths, little contour variety, motifs stated but not developed. |
| 5 | A motif is stated, varied, and answered. Phrases breathe (rests are used deliberately), the contour has a high point that means something, and the line goes somewhere. |

### 3. harmonic coherence — *does it hold together vertically?*

| | |
| --- | --- |
| 1 | No functional sense. Accidentals contradict the declared key, dissonances neither prepared nor resolved, the ending sits nowhere. |
| 3 | Diatonic and plausible. Phrase ends land on chord tones, cadences are recognizable but predictable; nothing wrong and nothing said. |
| 5 | A clear harmonic plan across the whole tune, with tension placed and released on purpose. Modal colour, if declared, is actually used (the flat second is heard, not just permitted). |

### 4. arrangement and register craft — *is it written for these instruments?*

Rate 3 for a single-voice tune with a well-chosen register; the item is
about whether the writing suits its forces, not about how many there are.

| | |
| --- | --- |
| 1 | Voices in impossible or absurd registers, all moving in the same rhythm, or so densely stacked it would sound like mud. |
| 3 | Playable and reasonably laid out. Bass beneath melody, sensible ranges, but roles are undifferentiated — everyone doing the same job. |
| 5 | Each voice has its own register, rhythm, and role; the parts answer each other; the instrumentation is a choice you can defend, not a default. |

### 5. would you keep it? — *the overall judgement*

| | |
| --- | --- |
| 1 | Delete. I would not show this to anyone. |
| 3 | I would keep it as a sketch and rewrite most of it. |
| 5 | I would keep this as written and play it to someone. |

Report per-item means with 95% bootstrap intervals, and inter-rater
agreement (Krippendorff's alpha, ordinal) per item. Item 5 is the headline;
items 1-4 explain it. Do **not** average the five into a single number.

## blinded A/B protocol

**Arms.** Every prompt is answered by every arm. At minimum:

| arm | what it is |
| --- | --- |
| A | the model under test |
| B | a second model (or the same model with a different prompt version) |
| C | `offline` — the deterministic baseline, presented unlabelled |
| D | *shuffled-prompt* control: the arm-A score for a **different** prompt, shown as the answer to this one |

Arm C tests whether "a language model wrote it" is worth anything over a
keyword table and a random walk. Arm D tests whether the text-to-music
mapping is real at all: if raters cannot separate D from A on **brief
fidelity**, the model is producing tunes, not answers, and every other
number in the eval is about fluency rather than understanding.

**Preparation.**

1. Run the sweep: `uv run python evals/run.py --models <A>,<B>,offline …`.
2. Render every final score to notation and audio with identical settings.
   Strip titles (`T:`) — models leak the prompt into them — and strip the
   chat reply. Keep instrument names; they are part of the arrangement.
3. Assign each rendering a random opaque id. Keep the id → (arm, prompt id,
   repeat) map in a file the raters cannot see.
4. Build arm D by pairing each arm-A score with a prompt from a *different*
   category, so the mismatch is not merely a near miss.

**Presentation.**

- Raters see: the prompt, then the arms for that prompt in a fresh random
  order per rater. Never the arm labels, model names, or ids that encode
  them.
- Rate all five items for every arm before moving to the next prompt, so
  the comparison is within-prompt.
- Then one forced-choice question per prompt: *which of these would you
  keep?* (ties allowed, one line of free text for why). This pairwise
  judgement is more reliable than the Likert means and should be reported
  alongside them as a win rate.
- At least 3 raters per prompt; at least 20 prompts stratified across the
  five categories; at least one rater with formal music training and one
  without, reported separately.

**Analysis.** Report per-arm item means, the forced-choice win rate with a
binomial confidence interval, and the A-vs-D gap on brief fidelity as the
primary evidence that the cross-modal mapping exists. Publish the id map
and the raw ratings with the results — a blinded protocol whose key is not
published is just an assertion.

**Pre-register** the arm list, the prompt sample, and the analysis above
before collecting a single rating. Write down what result would count as
"the model does not understand the brief", and honour it.
