SHELL := /bin/bash

.PHONY: help install shim shim-check hooks hooks-check deps-dev deps-server clean build

help:
	@echo "AI Trading Bot - Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install         Install CLI in editable mode (pip install -e .)"
	@echo "  make hooks           Install tracked git hooks from .githooks/"
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
	$(MAKE) hooks

# GH #1070: pip's editable install hardcodes the install-time checkout path, so a shared venv
# silently serves that checkout's code to every git worktree. The shim re-points src/cli at
# the checkout enclosing the cwd. Cheap, idempotent, no rebuild.
shim:
	python tools/install_worktree_shim.py

shim-check:
	python tools/install_worktree_shim.py --check

# GH #1077: hooks used to live untracked in .git/hooks, so an inert pre-push hook went
# unreviewed for months. The source is tracked in .githooks/; this links it into place.
hooks:
	python tools/install_git_hooks.py

hooks-check:
	python tools/install_git_hooks.py --check

deps-dev: install
	pip install -r requirements.txt

deps-server: install
	pip install -r requirements-server.txt

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

build: install
	python -m build
