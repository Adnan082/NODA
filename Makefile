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
	$(PYTHON) -m noda.eval.divergence
	$(PYTHON) -m noda.eval.benchmark
	$(PYTHON) -m noda.eval.calibration
	$(PYTHON) -m noda.eval.ood
