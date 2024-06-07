
format:
	ruff --fix-only .

lint:
	ruff check .

install:
	pip install -e .

test:
	pytest tool/tests