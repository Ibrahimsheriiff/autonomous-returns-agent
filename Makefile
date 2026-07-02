.PHONY: install format lint test check clean

install:
	uv sync

format:
	uv run ruff check --fix src tests
	uv run black src tests

lint:
	uv run ruff check src tests
	uv run black --check src tests

test:
	uv run pytest

check: lint test

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".uv-cache" -prune -exec rm -rf {} +
