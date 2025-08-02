dev:
	uv pip install -e .[test]

test:
	uv run pytest 