.PHONY: test lint demo run
test:
	uv run pytest -q
lint:
	uv run ruff check .
demo:
	uv run python -m pid.cli --synthetic --out /tmp/pid-demo
run:
	uv run python -m pid.cli --config configs/default.yaml
