SHELL := /bin/bash

# Interpreter for every target below. Bare `python` does not exist outside an activated venv
# (macOS ships only python3), and the repair target for a missing hook cannot itself depend on
# one: GH #1077's inert hook died on `python: command not found` and still reported success.
# Order: the active venv, this checkout's .venv, the primary checkout's .venv (linked worktrees
# have none, and the shim must land in the shared venv's site-packages), then python3.
# Override with `make PY=/path/to/python <target>`.
ifndef PY
PY := $(shell \
	if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then echo "$$VIRTUAL_ENV/bin/python"; \
	elif [ -x .venv/bin/python ]; then echo "$$PWD/.venv/bin/python"; \
	else \
		common=$$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null); \
		if [ -n "$$common" ] && [ -x "$$(dirname "$$common")/.venv/bin/python" ]; then \
			echo "$$(dirname "$$common")/.venv/bin/python"; \
		else command -v python3 || command -v python; fi; \
	fi)
endif

.PHONY: help install shim shim-check hooks hooks-check merge-drivers merge-drivers-check install-check require-python deps-dev deps-server clean build

help:
	@echo "AI Trading Bot - Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install         Install CLI in editable mode (pip install -e .)"
	@echo "  make hooks           Install tracked git hooks from .githooks/"
	@echo "  make merge-drivers   Register git merge drivers for append-only files"
	@echo "  make install-check   Verify the shim, git hooks and merge driver are installed and current"
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

# Fails loudly rather than letting an empty $(PY) turn `$(PY) tools/x.py` into a confusing
# `tools/x.py: Permission denied`.
require-python:
	@[ -n "$(PY)" ] || { echo "make: no Python interpreter found (no active venv, no .venv, no python3 on PATH). Activate a venv or run: make PY=/path/to/python $(MAKECMDGOALS)" >&2; exit 1; }

install: require-python
	$(PY) -m pip install -e .
	$(MAKE) shim
	$(MAKE) hooks
	$(MAKE) merge-drivers

# GH #1070: pip's editable install hardcodes the install-time checkout path, so a shared venv
# silently serves that checkout's code to every git worktree. The shim re-points src/cli at
# the checkout enclosing the cwd. Cheap, idempotent, no rebuild.
shim: require-python
	$(PY) tools/install_worktree_shim.py

shim-check: require-python
	$(PY) tools/install_worktree_shim.py --check

# GH #1077: hooks used to live untracked in .git/hooks, so an inert pre-push hook went
# unreviewed for months. The source is tracked in .githooks/; this copies it into place.
hooks: require-python
	$(PY) tools/install_git_hooks.py

hooks-check: require-python
	$(PY) tools/install_git_hooks.py --check

# GH #1079: `merge.<name>.driver` cannot be version-controlled, so an unregistered driver
# makes .gitattributes silently inert and append-only files conflict by hand on every PR.
merge-drivers: require-python
	$(PY) tools/install_merge_drivers.py

merge-drivers-check: require-python
	$(PY) tools/install_merge_drivers.py --check

# GH #1147: all three install-required artifacts in one command, so a scheduled caller (the
# pre-push hook, a standup) has a single thing to run. Exits non-zero if any has drifted.
install-check: require-python
	$(PY) tools/check_install_state.py

deps-dev: install
	$(PY) -m pip install -r requirements.txt

deps-server: install
	$(PY) -m pip install -r requirements-server.txt

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

build: install
	$(PY) -m build
