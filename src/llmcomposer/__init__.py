"""llmcomposer — compose music with an LLM copilot.

A pydantic-ai agent writes and revises tunes in ABC notation through
conversation; a FastAPI app renders the score as sheet music and plays it.
"""

from .abc_notation import ABCErrorCode, ABCValidationError, validate_abc
from .agent import DEFAULT_MODEL, PROMPT_VERSION, CompositionDeps, composer_agent
from .app import create_app
from .descriptors import ScoreDescriptors, describe
from .models import Bounce, ScoreUpdate, TurnMeta
from .session import ComposerSession

__all__ = [
    "DEFAULT_MODEL",
    "PROMPT_VERSION",
    "ABCErrorCode",
    "ABCValidationError",
    "Bounce",
    "ComposerSession",
    "CompositionDeps",
    "ScoreDescriptors",
    "ScoreUpdate",
    "TurnMeta",
    "composer_agent",
    "create_app",
    "describe",
    "validate_abc",
]
