SHELL := /bin/bash

VENV := .venv
VENV_BIN := $(VENV)/bin
PY := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
ATB := $(VENV_BIN)/atb

# Tunables (override on CLI, e.g., make backtest STRATEGY=bull DAYS=90)
PORT ?= 8090
STRATEGY ?= ml_basic
SYMBOL ?= BTCUSDT
TIMEFRAME ?= 1h
DAYS ?= 30

.PHONY: help venv install deps deps-server atb dashboards dashboard-monitoring \
        dashboard-backtesting live live-health backtest optimizer test lint fmt clean \
        rules setup-pre-commit

help:
	@echo "Common targets:"
	@echo "  make venv                 # create .venv"
	@echo "  make install              # install CLI (editable) and upgrade pip"
	@echo "  make deps                 # install dev deps (requirements.txt)"
	@echo "  make setup-pre-commit     # install pre-commit hooks"
	@echo "  make dashboards           # list dashboards"
	@echo "  make dashboard-monitoring # run monitoring dashboard on PORT=$(PORT)"
	@echo "  make live                 # run live trading (paper by default)"
	@echo "  make live-health          # run live trading with health server on PORT=$(PORT)"
	@echo "  make backtest             # run backtest for STRATEGY=$(STRATEGY)"
	@echo "  make optimizer            # run optimizer"
	@echo "  make test                 # run pytest with 4 workers"
	@echo "  make code-quality         # ruff/black/mypy/bandit"
	@echo "  make clean                # remove caches, build artifacts"

venv:
	test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	$(PY) -m pip install --upgrade pip
	$(PIP) install -e .

deps: venv
	$(PIP) install -r requirements.txt

deps-server: venv
	$(PIP) install -r requirements-server.txt

setup-pre-commit: deps
	$(PY) scripts/setup_pre_commit.py

# ------------- CLI wrappers (use venv bins directly, no need to activate) -------------
atb: install
	@$(ATB) --help

dashboards: install
	$(ATB) dashboards list

dashboard-monitoring: install
	$(ATB) dashboards run monitoring --port $(PORT)

dashboard-backtesting: install
	$(ATB) dashboards run backtesting --port 8001

live: install
	$(ATB) live $(STRATEGY)

live-health: install
	$(ATB) live-health $(STRATEGY)

backtest: install
	$(ATB) backtest $(STRATEGY) --symbol $(SYMBOL) --timeframe $(TIMEFRAME) --days $(DAYS)

optimizer: install
	$(ATB) optimizer --strategy $(STRATEGY) --days $(DAYS)

# ------------------------------ Quality / tests ---------------------------------------
test: deps
	$(VENV_BIN)/pytest -n 4

lint: venv
	$(VENV_BIN)/ruff check . || true

code-quality: venv
	$(VENV_BIN)/black .  && $(VENV_BIN)/ruff check .  && $(PY) bin/run_mypy.py && $(VENV_BIN)/bandit -c pyproject.toml -r src

# --------------------------------- Hygiene --------------------------------------------
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

# --------------------------------- Rules Generation --------------------------------------------
rules: deps
	$(PY) scripts/generate_rules.py
