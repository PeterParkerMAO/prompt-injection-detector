.PHONY: check test lint

check:
	@uv run python scripts/run_check.py

lint:
	@uv run ruff check .

test:
	@uv run pytest -q
