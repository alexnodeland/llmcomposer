"""Score an eval sweep: JSONL in, markdown out.

Reads the records written by ``evals/run.py`` and reports, per model:
notational competence (first-attempt validity, bounces, the error-code
histogram), prompt adherence (the machine-checkable ``expects``), and the
symbolic descriptors from :mod:`llmcomposer.descriptors`.

Nothing here judges musical quality — that is what ``evals/rubric.md`` is
for. These numbers say what the model *did*, not whether it was any good.

Examples
--------
::

    uv run python evals/score.py evals/results/dev.jsonl
    uv run python evals/score.py evals/results/dev.jsonl --out report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from llmcomposer.descriptors import ScoreDescriptors, describe

_MODE_ALIASES = {
    "ionian": "major",
    "aeolian": "minor",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL sweep file.

    Parameters
    ----------
    path : Path
        The file written by ``evals/run.py``.

    Returns
    -------
    list[dict[str, Any]]
        One dict per turn, in file order; blank and unparseable lines are
        skipped so a half-flushed file still scores.
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _root_matches(declared: str, expected: str) -> bool:
    """Compare a declared ``K:`` tonic with an expected root letter."""
    written = declared.strip()[:2]
    if len(written) > 1 and written[1] not in {"#", "b"}:
        written = written[:1]
    return written.lower() == expected.strip().lower()


_CHECKS: dict[str, Callable[[ScoreDescriptors, Any], bool]] = {
    "meter": lambda d, want: d.meter.strip() == str(want).strip(),
    "key_root": lambda d, want: _root_matches(d.declared_key, str(want)),
    "key_mode": lambda d, want: (
        _MODE_ALIASES.get(d.declared_mode, d.declared_mode)
        == _MODE_ALIASES.get(str(want), str(want))
    ),
    "voices": lambda d, want: d.voice_count == int(want),
    "bars_min": lambda d, want: d.bar_count >= int(want),
    "bars_max": lambda d, want: d.bar_count <= int(want),
    "cadence": lambda d, want: d.cadence == str(want),
    "no_range_violations": lambda d, want: (not d.range_violations) is bool(want),
}


def check_expects(
    expects: dict[str, Any], descriptors: ScoreDescriptors
) -> dict[str, bool]:
    """Evaluate a prompt's machine-checkable claims against a score.

    Parameters
    ----------
    expects : dict[str, Any]
        The ``expects`` block from ``prompts.yaml``.
    descriptors : ScoreDescriptors
        Descriptors of the score the model actually returned.

    Returns
    -------
    dict[str, bool]
        One verdict per recognized key; unknown keys are ignored.
    """
    return {
        key: bool(_CHECKS[key](descriptors, want))
        for key, want in expects.items()
        if key in _CHECKS
    }


@dataclass
class ModelSummary:
    """Everything the report shows for one model arm."""

    model: str
    turns: int = 0
    failures: int = 0
    first_attempt_valid: int = 0
    bounce_counts: list[int] = field(default_factory=list[int])
    codes: Counter[str] = field(default_factory=Counter[str])
    checks_passed: int = 0
    checks_total: int = 0
    prompts_fully_satisfied: int = 0
    prompts_with_expects: int = 0
    descriptors: list[ScoreDescriptors] = field(default_factory=list[ScoreDescriptors])
    cadences: Counter[str] = field(default_factory=Counter[str])

    @property
    def validity_rate(self) -> float:
        """Share of turns whose first ABC passed the validator."""
        scored = self.turns - self.failures
        return self.first_attempt_valid / scored if scored else 0.0

    @property
    def mean_bounces(self) -> float:
        """Mean validator rejections per completed turn."""
        return fmean(self.bounce_counts) if self.bounce_counts else 0.0

    @property
    def expects_rate(self) -> float:
        """Share of individual ``expects`` claims satisfied."""
        return self.checks_passed / self.checks_total if self.checks_total else 0.0

    def mean(self, attribute: str) -> float:
        """Mean of one numeric descriptor across this arm's scores."""
        values = [float(getattr(d, attribute)) for d in self.descriptors]
        return fmean(values) if values else 0.0

    def mean_optional(self, attribute: str) -> float | None:
        """Mean of a descriptor that is ``None`` when it does not apply.

        Modal descriptors are undefined for a score with no parseable key or
        no pitched content; those scores are excluded rather than counted as
        zero, which would silently drag the mean down.
        """
        values = [
            float(value)
            for d in self.descriptors
            if (value := getattr(d, attribute)) is not None
        ]
        return fmean(values) if values else None


def _bounces(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a record's bounce entries, tolerating older record shapes."""
    raw: object = record.get("bounces")
    if not isinstance(raw, list):
        return []
    entries = cast("list[object]", raw)
    return [entry for entry in entries if isinstance(entry, dict)]


def _bounce_codes(record: dict[str, Any]) -> list[str]:
    """Pull the error codes out of a record's bounce list."""
    return [str(bounce.get("code", "unknown")) for bounce in _bounces(record)]


def _bounce_total(record: dict[str, Any]) -> int:
    """Count how many times the validator rejected this turn's score."""
    bounces = _bounces(record)
    if bounces:
        return len(bounces)
    meta: dict[str, Any] = record.get("meta") or {}
    return int(meta.get("corrections", 0))


def _absorb(summary: ModelSummary, record: dict[str, Any]) -> None:
    """Fold one turn record into its model's summary."""
    summary.turns += 1
    if not record.get("ok"):
        summary.failures += 1
        return
    bounces = _bounce_total(record)
    summary.bounce_counts.append(bounces)
    summary.first_attempt_valid += bounces == 0
    summary.codes.update(_bounce_codes(record))
    if not record.get("final_turn", True):
        return
    descriptors = describe(str(record.get("abc", "")))
    summary.descriptors.append(descriptors)
    summary.cadences[descriptors.cadence] += 1
    expects: dict[str, Any] = record.get("expects") or {}
    verdicts = check_expects(expects, descriptors)
    if not verdicts:
        return
    summary.prompts_with_expects += 1
    summary.checks_total += len(verdicts)
    summary.checks_passed += sum(verdicts.values())
    summary.prompts_fully_satisfied += all(verdicts.values())


def summarize(records: Sequence[dict[str, Any]]) -> list[ModelSummary]:
    """Group turn records by model and fold each group into a summary.

    Parameters
    ----------
    records : Sequence[dict[str, Any]]
        Turn records from one or more sweeps.

    Returns
    -------
    list[ModelSummary]
        One summary per model, in first-seen order.
    """
    summaries: dict[str, ModelSummary] = {}
    for record in records:
        model = str(
            record.get("model_id")
            or record.get("model")
            or record.get("model_arg")
            or "unknown"
        )
        summary = summaries.setdefault(model, ModelSummary(model))
        _absorb(summary, record)
    return list(summaries.values())


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _competence_table(summaries: Sequence[ModelSummary]) -> str:
    """Notational competence and prompt adherence, one row per model."""
    rows = [
        [
            summary.model,
            str(summary.turns),
            str(summary.failures),
            f"{summary.validity_rate:.0%}",
            f"{summary.mean_bounces:.2f}",
            f"{summary.expects_rate:.0%}",
            f"{summary.prompts_fully_satisfied}/{summary.prompts_with_expects}",
        ]
        for summary in summaries
    ]
    return _table(
        (
            "model",
            "turns",
            "failed",
            "first-attempt valid",
            "mean bounces",
            "expects met",
            "prompts fully met",
        ),
        rows,
    )


def _optional(value: float | None, spec: str) -> str:
    """Format a descriptor mean that may be undefined for every score."""
    return "—" if value is None else format(value, spec)


def _descriptor_table(summaries: Sequence[ModelSummary]) -> str:
    """Mean symbolic descriptors, one row per model."""
    rows = [
        [
            summary.model,
            f"{summary.mean('pitch_class_entropy'):.2f}",
            f"{summary.mean('ngram_repetition_rate'):.2f}",
            f"{summary.mean('key_correlation'):.2f}",
            f"{summary.mean('key_match'):.0%}",
            _optional(summary.mean_optional("mode_match"), ".0%"),
            _optional(summary.mean_optional("diatonic_adherence"), ".2f"),
            f"{summary.mean('mean_abs_interval'):.2f}",
            f"{summary.mean('leap_rate'):.2f}",
            f"{summary.mean('rest_fraction'):.2f}",
            (
                f"{len([d for d in summary.descriptors if d.range_violations])}"
                f"/{len(summary.descriptors)}"
            ),
        ]
        for summary in summaries
    ]
    return _table(
        (
            "model",
            "pc entropy",
            "3-gram repetition",
            "tonal fit (maj/min template)",
            "key match",
            "mode match",
            "diatonic adherence",
            "mean interval",
            "leap rate",
            "rest fraction",
            "scores w/ range violations",
        ),
        rows,
    )


def _histogram_table(
    summaries: Sequence[ModelSummary], attribute: str, label: str
) -> str:
    """Render a counter attribute as a model x category table."""
    counters = {s.model: getattr(s, attribute) for s in summaries}
    keys = sorted({k for counter in counters.values() for k in counter})
    if not keys:
        return f"_no {label} recorded_"
    rows = [
        [model, *[str(counter[key]) for key in keys]]
        for model, counter in counters.items()
    ]
    return _table(("model", *keys), rows)


def report(summaries: Sequence[ModelSummary]) -> str:
    """Render the whole markdown report.

    Parameters
    ----------
    summaries : Sequence[ModelSummary]
        Per-model summaries from :func:`summarize`.

    Returns
    -------
    str
        A markdown document with four tables.
    """
    return "\n\n".join(
        (
            "## notational competence",
            _competence_table(summaries),
            "## validator bounces by error code",
            _histogram_table(summaries, "codes", "bounces"),
            "## symbolic descriptors (final score of each conversation)",
            _descriptor_table(summaries),
            "## cadence distribution",
            _histogram_table(summaries, "cadences", "cadences"),
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Score a sweep from the command line.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the file held no records.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("jsonl", type=Path, help="a sweep written by evals/run.py")
    parser.add_argument("--out", type=Path, default=None, help="write markdown here")
    args = parser.parse_args(argv)
    records = load_records(Path(args.jsonl))
    if not records:
        print(f"no records in {args.jsonl}")
        return 1
    text = f"# {Path(args.jsonl).name} — {len(records)} turns\n\n" + report(
        summarize(records)
    )
    if args.out is not None:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
