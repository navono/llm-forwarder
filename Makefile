dev:
	uv run -m src.llm_forwarder

test-openai:
	uv run python -m pytest tests/test_openai.py -v

test-integration:
	uv run python -m pytest tests/integration/ -v

ruff-fix:
	ruff check --fix .

.PHONY: dev test-openai test-integration ruff-fix