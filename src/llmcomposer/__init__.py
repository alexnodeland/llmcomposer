"""llmcomposer — compose music with an LLM copilot.

A pydantic-ai agent writes and revises tunes in ABC notation through
conversation; a FastAPI app renders the score as sheet music and plays it.
"""

from .abc_notation import ABCValidationError, validate_abc
from .agent import DEFAULT_MODEL, CompositionDeps, composer_agent
from .app import create_app
from .models import ScoreUpdate
from .session import ComposerSession

__all__ = [
    "DEFAULT_MODEL",
    "ABCValidationError",
    "ComposerSession",
    "CompositionDeps",
    "ScoreUpdate",
    "composer_agent",
    "create_app",
    "validate_abc",
]
