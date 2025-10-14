SHELL := /bin/bash

# Tunables (override on CLI, e.g., make backtest STRATEGY=bull DAYS=90)
PORT ?= 8090
STRATEGY ?= ml_basic
SYMBOL ?= BTCUSDT
TIMEFRAME ?= 1h
DAYS ?= 30

.PHONY: help install deps deps-server atb dashboards dashboard-monitoring \
        dashboard-backtesting live live-health backtest optimizer test lint fmt clean \
        setup-pre-commit dev-setup migrate startup

help:
	@echo "Common targets:"
	@echo "  make dev-setup            # setup development environment with venv"
	@echo "  make install              # install CLI (editable) and upgrade pip"
	@echo "  make deps                 # install dev deps (requirements.txt)"
	@echo "  make deps-server          # install production deps (requirements-server.txt)"
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
	@echo "  make migrate              # run database migrations"
	@echo "  make startup              # run migrations and start application"

install:
	python -m pip install --upgrade pip
	pip install -e .

deps: install
	pip install -r requirements.txt

deps-server: install
	pip install -r requirements-server.txt

setup-pre-commit: deps
        pre-commit install

# ------------- CLI wrappers (direct execution) -------------
atb: install
	@atb --help

dashboards: install
	atb dashboards list

dashboard-monitoring: install
	atb dashboards run monitoring --port $(PORT)

dashboard-backtesting: install
	atb dashboards run backtesting --port 8001

live: install
	atb live $(STRATEGY)

live-health: install
	atb live-health $(STRATEGY)

backtest: install
	atb backtest $(STRATEGY) --symbol $(SYMBOL) --timeframe $(TIMEFRAME) --days $(DAYS)

optimizer: install
	atb optimizer --strategy $(STRATEGY) --days $(DAYS)

# ------------------------------ Quality / tests ---------------------------------------
test: deps
	pytest -n 4

lint:
	ruff check . || true

code-quality:
	black .  && ruff check .  && python bin/run_mypy.py && bandit -c pyproject.toml -r src

# --------------------------------- Hygiene --------------------------------------------
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

# --------------------------------- Development Setup --------------------------------------------
dev-setup:
        atb dev setup

# --------------------------------- Database Migrations --------------------------------------------
migrate: install
	atb db migrate

# --------------------------------- Rules Generation --------------------------------------------
