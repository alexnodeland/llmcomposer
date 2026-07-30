.PHONY: ci check lint lint-fix format format-check type-check test test-cov run run-offline docs docs-build clean

# CI — emulate the GitHub Actions pipeline locally. `make check` is the
# shorter everyday gate; `make ci` is byte-for-byte what CI enforces.
ci:
	@echo "=== Running CI Pipeline ==="
	@echo "\n--- Lint Check ---"
	uv run ruff check src tests
	@echo "\n--- Format Check ---"
	uv run ruff format --check src tests
	@echo "\n--- Type Check ---"
	uv run pyright src
	@echo "\n--- Tests ---"
	uv run pytest -v
	@echo "\n=== CI Pipeline Complete ==="

# The everyday pre-push gate: lint, types, tests.
check: lint type-check test

lint:
	uv run ruff check src tests

lint-fix:
	uv run ruff check --fix src tests

format:
	uv run ruff format src tests

format-check:
	uv run ruff format --check src tests

type-check:
	uv run pyright src

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=llmcomposer --cov-report=term --cov-report=html

# Run the studio against a real model (needs ANTHROPIC_API_KEY or
# LLMCOMPOSER_MODEL pointing at another provider).
run:
	uv run llmcomposer

# Run entirely offline — the built-in FunctionModel composes deterministic
# tunes, no credentials or network needed.
run-offline:
	LLMCOMPOSER_MODEL=offline uv run llmcomposer

docs:
	uv run --group docs mkdocs serve

docs-build:
	uv run --group docs mkdocs build --strict

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage site dist
