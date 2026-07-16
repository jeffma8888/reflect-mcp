# reflect-mcp -- dev tasks. Uses `uv` if present, else python venv + pip.
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help venv install test cov eval eval-live run clean

help:
	@echo "make install   - create .venv and install the package + dev deps"
	@echo "make test      - run the deterministic pytest suite (offline, no API keys)"
	@echo "make cov       - run tests with coverage"
	@echo "make eval      - run the eval harness OFFLINE against replay fixtures"
	@echo "make eval-live - run the eval against a real provider (needs REFLECT_PROVIDER + creds)"
	@echo "make run       - start the MCP server (needs [server] + a provider extra)"
	@echo "make clean     - remove venv + caches"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -e ".[dev,server]"

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=reflect_mcp --cov-report=term-missing

eval:
	$(PY) eval/run_eval.py

eval-live:
	$(PY) eval/run_eval.py --live

run:
	$(PY) -m reflect_mcp.server

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ .coverage
