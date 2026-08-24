SHELL := /bin/bash

.PHONY: help install shim shim-check deps-dev deps-server clean build

help:
	@echo "AI Trading Bot - Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install         Install CLI in editable mode (pip install -e .)"
	@echo "  make deps-dev        Install development dependencies (includes install)"
	@echo "  make deps-server     Install server/production dependencies (includes install)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           Remove caches and build artifacts"
	@echo "  make build           Build distributable package"
	@echo ""
	@echo "Note: For project operations, use 'atb' commands:"
	@echo "  atb test unit        Run unit tests"
	@echo "  atb dev quality      Run code quality checks"
	@echo "  atb backtest         Run strategy backtests"
	@echo "  atb --help           Show all available commands"

install:
	pip install -e .
	$(MAKE) shim

# GH #1070: pip's editable install hardcodes the install-time checkout path, so a shared venv
# silently serves that checkout's code to every git worktree. The shim re-points src/cli at
# the checkout enclosing the cwd. Cheap, idempotent, no rebuild.
shim:
	python tools/install_worktree_shim.py

shim-check:
	python tools/install_worktree_shim.py --check

deps-dev: install
	pip install -r requirements.txt

deps-server: install
	pip install -r requirements-server.txt

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

build: install
	python -m build
