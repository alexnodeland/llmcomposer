r"""Batch runner for the llmcomposer prompt suite.

Sweeps models x prompts x repeats straight through
:class:`llmcomposer.session.ComposerSession` — no web server, no browser —
and appends one JSONL record per turn. The record shape follows
``llmcomposer.recording`` so a sweep and a live studio session land in the
same schema and can be scored by the same script.

Runs are resumable: a ``(model, prompt id, repeat)`` triple already present
in the output file is skipped, so an interrupted sweep can be restarted with
the same command.

Sampling is pinned by default: ``--temperature`` defaults to ``0.0`` and
``--seed`` to ``0``, and both are passed explicitly to every session rather
than left to a provider default. Each cell gets its own seed, derived from a
digest of ``(base seed, prompt id, repeat)``, so repeats sample distinct
points of the output distribution instead of collapsing onto one.

Examples
--------
::

    uv run python evals/run.py --models offline --repeats 2 \\
        --out evals/results/dev.jsonl
    uv run python evals/run.py --models offline,anthropic:claude-opus-5 \\
        --categories constraint,adversarial_meter --repeats 3 \\
        --temperature 0.7 --seed 7 \\
        --out evals/results/sweep-2026-07.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from llmcomposer import agent as agent_module
from llmcomposer.session import ComposerSession

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = REPO_ROOT / "evals" / "prompts.yaml"
DEFAULT_OUT = REPO_ROOT / "evals" / "results" / "dev.jsonl"

DEFAULT_TEMPERATURE = 0.0
"""The pinned sweep default: greedy decoding unless a run says otherwise."""

DEFAULT_SEED = 0
"""The pinned base seed; per-cell seeds are derived from it."""

_SEED_MODULUS = 2**31 - 1


@dataclass(frozen=True)
class EvalPrompt:
    """One prompt from the suite: an id, a category, and its turns."""

    id: str
    category: str
    turns: tuple[str, ...]
    expects: dict[str, Any] = field(default_factory=dict[str, Any])


def load_prompts(path: Path) -> list[EvalPrompt]:
    """Load the prompt suite from a YAML file.

    Parameters
    ----------
    path : Path
        Location of ``prompts.yaml``.

    Returns
    -------
    list[EvalPrompt]
        Every prompt in file order, single-turn entries normalized to a
        one-element ``turns`` tuple.
    """
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = list(raw["prompts"])
    prompts: list[EvalPrompt] = []
    seen: set[str] = set()
    for entry in entries:
        prompt_id = str(entry["id"])
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt id in {path}: {prompt_id}")
        seen.add(prompt_id)
        turns = entry.get("turns") or [entry["text"]]
        prompts.append(
            EvalPrompt(
                id=prompt_id,
                category=str(entry.get("category", "uncategorized")),
                turns=tuple(str(turn) for turn in turns),
                expects=dict(entry.get("expects") or {}),
            )
        )
    return prompts


def _sha(text: str) -> str:
    """Return a short, stable sha256 prefix for a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cell_seed(base_seed: int, prompt_id: str, repeat: int) -> int:
    """Derive a distinct, reproducible seed for one sweep cell.

    Repeats of the same cell must sample *different* points of the model's
    output distribution — otherwise a provider that honours ``seed`` returns
    the same draw every time and the repeats measure nothing. Digesting the
    triple keeps every cell independent while staying a pure function of the
    base seed, so the whole sweep replays exactly.

    Parameters
    ----------
    base_seed : int
        The sweep's ``--seed``.
    prompt_id : str
        The prompt's stable id from ``prompts.yaml``.
    repeat : int
        Which repetition of this cell.

    Returns
    -------
    int
        A non-negative seed below 2**31 - 1.
    """
    digest = hashlib.sha256(f"{base_seed}|{prompt_id}|{repeat}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def _prompt_version() -> str:
    """Return the agent's prompt version, or ``""`` if it declares none."""
    version: object = getattr(agent_module, "PROMPT_VERSION", "")
    return str(version)


def _system_prompt_sha() -> str:
    """Return the agent's pinned prompt digest (stable across processes)."""
    pinned: object = getattr(agent_module, "PROMPT_SHA", "")
    if pinned:
        return str(pinned)
    return _sha(str(getattr(agent_module, "COMPOSER_INSTRUCTIONS", "")))


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    """Return the ``(model, prompt id, repeat)`` triples already recorded.

    Parameters
    ----------
    path : Path
        The JSONL output file; may not exist yet.

    Returns
    -------
    set[tuple[str, str, int]]
        Keys to skip on a resumed run.
    """
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add(
            (
                str(record.get("model_arg", "")),
                str(record.get("prompt_id", "")),
                int(record.get("repeat", 0)),
            )
        )
    return done


def _turn_record(
    *,
    run_id: str,
    session_id: str,
    model_arg: str,
    model_id: str,
    prompt: EvalPrompt,
    repeat: int,
    turn_index: int,
    message: str,
) -> dict[str, Any]:
    """Build the invariant half of a turn record."""
    return {
        "run_id": run_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "ts": datetime.now(UTC).isoformat(),
        "model_arg": model_arg,
        "model_id": model_id,
        # Same key llmcomposer.recording writes, so studio turns and sweep
        # turns can be scored together.
        "model": model_id,
        "prompt_id": prompt.id,
        "category": prompt.category,
        "repeat": repeat,
        "final_turn": turn_index == len(prompt.turns) - 1,
        "expects": prompt.expects,
        "prompt_version": _prompt_version(),
        "prompt_sha": _sha(message),
        "system_prompt_sha": _system_prompt_sha(),
        "user_message": message,
    }


async def run_conversation(
    model_arg: str,
    prompt: EvalPrompt,
    repeat: int,
    run_id: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    base_seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Run one prompt's whole conversation and return its turn records.

    Parameters
    ----------
    model_arg : str
        A model name understood by :class:`ComposerSession` (``offline``,
        ``anthropic:claude-opus-5``, ``litellm:…``).
    prompt : EvalPrompt
        The prompt to run.
    repeat : int
        Which repetition this is; recorded so repeats stay distinguishable.
    run_id : str
        Identifier shared by every record in this sweep.
    temperature : float, optional
        Sampling temperature, passed explicitly to the session.
    base_seed : int, optional
        The sweep's base seed; this cell's seed is derived from it by
        :func:`cell_seed`.

    Returns
    -------
    list[dict[str, Any]]
        One record per turn. A turn that raised is recorded with
        ``ok: false`` and the conversation stops there.
    """
    session = ComposerSession(
        model=model_arg,
        temperature=temperature,
        seed=cell_seed(base_seed, prompt.id, repeat),
    )
    session_id = uuid.uuid4().hex
    records: list[dict[str, Any]] = []
    for turn_index, message in enumerate(prompt.turns):
        record = _turn_record(
            run_id=run_id,
            session_id=session_id,
            model_arg=model_arg,
            model_id=session.model_name,
            prompt=prompt,
            repeat=repeat,
            turn_index=turn_index,
            message=message,
        )
        started = time.monotonic()
        try:
            update, meta = await session.send(message)
        except Exception as exc:  # noqa: BLE001 — a failed turn is a datum
            record |= {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            }
            records.append(record)
            break
        dumped = meta.model_dump()
        record |= {
            "ok": True,
            "error": None,
            "reply": update.reply,
            "abc": update.abc,
            "meta": dumped,
            "bounces": dumped.get("bounces", []),
            "temperature": dumped.get("temperature"),
            "seed": dumped.get("seed"),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }
        records.append(record)
    return records


async def sweep(
    models: list[str],
    prompts: list[EvalPrompt],
    repeats: int,
    out: Path,
    concurrency: int,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    base_seed: int = DEFAULT_SEED,
) -> int:
    """Run the whole grid, appending records as each conversation finishes.

    Parameters
    ----------
    models : list[str]
        Model names to sweep.
    prompts : list[EvalPrompt]
        The prompt suite (already filtered).
    repeats : int
        How many times to run each model x prompt cell.
    out : Path
        JSONL file to append to; created with its parents if missing.
    concurrency : int
        Maximum conversations in flight at once.
    temperature : float, optional
        Sampling temperature applied to every cell, and printed in the run
        header so the resolved sampling is on the record.
    base_seed : int, optional
        Base seed; each cell's seed is ``cell_seed(base_seed, id, repeat)``.

    Returns
    -------
    int
        Number of records written.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(out)
    run_id = uuid.uuid4().hex[:12]
    limiter = asyncio.Semaphore(concurrency)
    writer = asyncio.Lock()
    written = 0

    async def one(model: str, prompt: EvalPrompt, repeat: int) -> None:
        nonlocal written
        async with limiter:
            records = await run_conversation(
                model,
                prompt,
                repeat,
                run_id,
                temperature=temperature,
                base_seed=base_seed,
            )
        async with writer:
            with out.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(records)
        status = "ok" if all(r["ok"] for r in records) else "FAILED"
        print(f"  {model} · {prompt.id} · repeat {repeat} · {status}")

    tasks = [
        one(model, prompt, repeat)
        for model in models
        for prompt in prompts
        for repeat in range(repeats)
        if (model, prompt.id, repeat) not in done
    ]
    skipped = len(models) * len(prompts) * repeats - len(tasks)
    print(
        f"run {run_id}: {len(tasks)} conversations to run, {skipped} already in {out}"
    )
    print(
        f"  sampling: temperature {temperature}, base seed {base_seed} "
        f"(per-cell seed = digest of base seed, prompt id, repeat)"
    )
    await asyncio.gather(*tasks)
    print(f"wrote {written} turn records to {out}")
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--models",
        default="offline",
        help="comma-separated model names (default: offline)",
    )
    parser.add_argument(
        "--prompts", type=Path, default=DEFAULT_PROMPTS, help="prompt suite YAML"
    )
    parser.add_argument("--repeats", type=int, default=1, help="runs per cell")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSONL sink")
    parser.add_argument(
        "--categories", default="", help="comma-separated categories to keep"
    )
    parser.add_argument("--ids", default="", help="comma-separated prompt ids to keep")
    parser.add_argument("--limit", type=int, default=0, help="cap on prompts run")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="conversations in flight"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            f"sampling temperature, pinned for every cell "
            f"(default: {DEFAULT_TEMPERATURE}, the sweep default)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"base sampling seed; each cell draws a distinct seed derived "
            f"from (seed, prompt id, repeat) (default: {DEFAULT_SEED})"
        ),
    )
    return parser.parse_args(argv)


def _select(prompts: list[EvalPrompt], args: argparse.Namespace) -> list[EvalPrompt]:
    """Apply the category / id / limit filters to the suite."""
    categories = {c.strip() for c in str(args.categories).split(",") if c.strip()}
    ids = {i.strip() for i in str(args.ids).split(",") if i.strip()}
    chosen = [
        prompt
        for prompt in prompts
        if (not categories or prompt.category in categories)
        and (not ids or prompt.id in ids)
    ]
    limit = int(args.limit)
    return chosen[:limit] if limit > 0 else chosen


def main(argv: list[str] | None = None) -> int:
    """Run a sweep from the command line.

    Returns
    -------
    int
        Process exit status: ``0`` when at least one record was written or
        everything was already done, ``1`` when the selection was empty.
    """
    args = _parse_args(argv)
    models = [name.strip() for name in str(args.models).split(",") if name.strip()]
    prompts = _select(load_prompts(Path(args.prompts)), args)
    if not models or not prompts:
        print("nothing to run: empty model or prompt selection")
        return 1
    asyncio.run(
        sweep(
            models,
            prompts,
            int(args.repeats),
            Path(args.out),
            int(args.concurrency),
            temperature=float(args.temperature),
            base_seed=int(args.seed),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
