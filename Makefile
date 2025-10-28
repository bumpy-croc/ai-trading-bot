SHELL := /bin/bash

.PHONY: help install deps-dev deps-prod clean build

help:
	@echo "AI Trading Bot - Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install CLI in editable mode (pip install -e .)"
	@echo "  make deps-dev      Install development dependencies (includes install)"
	@echo "  make deps-prod     Install production dependencies (includes install)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove caches and build artifacts"
	@echo "  make build         Build distributable package"
	@echo ""
	@echo "Note: Use 'atb --help' for all project operations (testing, quality checks, backtesting, etc.)"

install:
	pip install -e .

deps-dev: install
	pip install -r requirements.txt

deps-prod: install
	pip install -r requirements-server.txt

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

build: install
	python -m build
