"""Shared test configuration: never let tests hit a real model API."""

from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False
