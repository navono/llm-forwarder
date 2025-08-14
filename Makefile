dev:
	uv run -m packages.llm_forwarder

test-integration:
	uv run python -m pytest tests/integration/ -v

# uv run python -m pytest tests/integration/test_handler_index_tts.py -v -s

ruff-fix:
	ruff check --fix .

.PHONY: dev test-openai test-integration ruff-fix