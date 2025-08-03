dev:
	uv pip install -e .[test]
test:
	uv run pytest 
build:
	uv build
