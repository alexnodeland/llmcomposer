"""Append-only recording of composing turns, so the data outlives the tab.

The studio is a measurement instrument; without a sink, every turn — the
score, the sampling settings, and above all the validator bounces — is
garbage-collected the moment the response is rendered. Set
``LLMCOMPOSER_RECORD_DIR`` and each finished turn is appended as one JSON
line to ``<dir>/turns-<YYYYMMDD>.jsonl``. Unset, the recorder is a no-op.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import logfire

from .models import TurnMeta

RECORD_DIR_ENV = "LLMCOMPOSER_RECORD_DIR"
"""Environment variable naming the directory turn records are appended to."""


class Recorder:
    """A JSONL sink for finished turns.

    Parameters
    ----------
    directory : str | os.PathLike[str] | None, optional
        Where to append records. ``None`` (the default) defers to
        ``LLMCOMPOSER_RECORD_DIR``, read afresh on every write so the sink
        can be switched on without rebuilding the app.
    """

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self._directory = Path(directory) if directory is not None else None

    @property
    def directory(self) -> Path | None:
        """The directory records are written to, or ``None`` when disabled."""
        if self._directory is not None:
            return self._directory
        configured = os.environ.get(RECORD_DIR_ENV, "").strip()
        return Path(configured) if configured else None

    @property
    def enabled(self) -> bool:
        """Whether this recorder currently writes anything."""
        return self.directory is not None

    def record(
        self,
        *,
        session_id: str,
        turn_index: int,
        user_message: str,
        reply: str,
        abc: str,
        meta: TurnMeta,
    ) -> Path | None:
        """Append one turn to today's JSONL file.

        Parameters
        ----------
        session_id : str
            Opaque id of the conversation the turn belongs to.
        turn_index : int
            Zero-based position of the turn within that conversation.
        user_message : str
            What the collaborator asked for.
        reply : str
            The composer's prose answer.
        abc : str
            The complete score the turn settled on.
        meta : TurnMeta
            Transparency data for the turn, including its bounces.

        Returns
        -------
        Path | None
            The file written to, or ``None`` when recording is disabled or
            the write failed. A recording failure never breaks a turn.
        """
        directory = self.directory
        if directory is None:
            return None
        now = datetime.now(UTC)
        record: dict[str, Any] = {
            "ts": now.isoformat(),
            "session_id": session_id,
            "turn_index": turn_index,
            "model": meta.model,
            "prompt_version": meta.prompt_version,
            "prompt_sha": meta.prompt_sha,
            "user_message": user_message,
            "reply": reply,
            "abc": abc,
            "bounces": [bounce.model_dump() for bounce in meta.bounces],
            "usage": {
                "requests": meta.requests,
                "input_tokens": meta.input_tokens,
                "output_tokens": meta.output_tokens,
            },
            "elapsed_ms": meta.elapsed_ms,
            "temperature": meta.temperature,
            "seed": meta.seed,
        }
        path = directory / f"turns-{now:%Y%m%d}.jsonl"
        line = json.dumps(record, ensure_ascii=False)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
        except OSError as exc:
            logfire.warn("turn recording failed: {error}", error=str(exc))
            return None
        return path
