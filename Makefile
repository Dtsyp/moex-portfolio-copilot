.PHONY: eda eda-offline test lint

eda:
	python scripts/download_data.py
	python scripts/run_eda.py --offline

eda-offline:
	python scripts/download_data.py --offline
	python scripts/run_eda.py --offline

test:
	python -m pytest -q

lint:
	python -m ruff check .
	python -m ruff format --check .
