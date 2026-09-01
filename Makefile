PYTHON := .venv/bin/python
PYTEST  := $(PYTHON) -m pytest

.PHONY: test test-cov install

## Run all tests using the venv Python (avoids system pytest picking wrong interpreter)
test:
	$(PYTEST)

## Run tests with coverage report
test-cov:
	$(PYTEST) --cov=. --cov-report=term-missing

## Install all project deps into the venv
install:
	$(PYTHON) -m pip install -e ".[dev]"
