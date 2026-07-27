.PHONY: setup data test bench lint

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m noda.data.generate

test:
	pytest -q

lint:
	ruff check src tests

bench:
	@echo "placeholder: will call eval/benchmark.py to regenerate Experiments 1-4 (Day 7)"
	@echo "Day 1: only 'data' and 'test' are functional."
