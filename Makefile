# Try bash first, fall back to zsh (for macOS)
ifeq ($(shell which bash 2>/dev/null),)
SHELL := /bin/zsh
else
SHELL := /bin/bash
endif

# Project variables
PACKAGE      := senpai
DIST_NAME    := astro-senpai
VERSION      := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
# Export variables
EXPORT_BASE := dist/export
EXPORT_NAME := $(PACKAGE)-$(VERSION)
EXPORT_DIR  := $(EXPORT_BASE)/$(EXPORT_NAME)

.DEFAULT_GOAL := help

###################
# Setup           #
###################

.PHONY: check-uv sync

check-uv:
	@which uv > /dev/null 2>&1 || ( \
		echo "Error: uv is not installed. Please install it first:"; \
		echo "curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1 \
	)

sync: check-uv  ## Sync dependencies from pyproject.toml into virtual environment
	uv sync --all-extras

###################
# Main Operations #
###################

.PHONY: run

run: ## Run the API locally
	uv run python -m senpai.api.main

###################
# Testing         #
###################

.PHONY: test test-ci coverage

test: ## Run the full test suite
	uv run pytest -v

test-ci: ## Run the CI test subset (no astrometry.net / catalog deps)
	uv run pytest -v -m "not requires_astrometry and not requires_catalog"

coverage: ## Run tests with coverage + regenerate badges (tests.svg, coverage.svg)
	uv run pytest -v \
		--junitxml=reports/junit/junit.xml \
		--cov=$(PACKAGE) \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-report=html
	uv run genbadge tests -o tests.svg
	uv run genbadge coverage -i coverage.xml -o coverage.svg
	@echo "Coverage report: htmlcov/index.html"

###################
# Code Quality    #
###################

.PHONY: lint format

lint: ## Run linter (ruff)
	uv run ruff check $(PACKAGE)/ tests/

format: ## Format code and fix lint issues (ruff)
	uv run ruff format $(PACKAGE)/ tests/
	uv run ruff check --fix $(PACKAGE)/ tests/

###################
# Release         #
###################

.PHONY: version build check-clean-tree check-version-unpublished check-tag-free tag publish-test publish

version: ## Print the version from pyproject.toml
	@echo $(VERSION)

build: ## Build wheel and sdist (dist name: astro-senpai)
	rm -rf dist/
	uv build

check-clean-tree:
	@test -z "$$(git status --porcelain)" || { echo "ERROR: git tree is dirty; commit or stash first"; git status --short; exit 1; }

check-version-unpublished:
	@if curl -sf https://pypi.org/pypi/$(DIST_NAME)/$(VERSION)/json > /dev/null; then \
		echo "ERROR: $(DIST_NAME) $(VERSION) is already on PyPI; bump version in pyproject.toml first"; exit 1; \
	fi
	@echo "PyPI check OK: $(DIST_NAME) $(VERSION) not yet published"

check-tag-free:
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" > /dev/null; then \
		echo "ERROR: local tag v$(VERSION) already exists"; exit 1; \
	fi
	@if git ls-remote --exit-code --tags origin "v$(VERSION)" > /dev/null 2>&1; then \
		echo "ERROR: tag v$(VERSION) already exists on origin"; exit 1; \
	fi
	@echo "Tag check OK: v$(VERSION) is free"

tag: check-clean-tree check-tag-free ## git tag v<version> from pyproject.toml and push it
	git tag v$(VERSION)
	git push origin v$(VERSION)

publish-test: check-clean-tree test-ci build ## Test + build + upload to TestPyPI (token from ~/.pypirc)
	uvx twine upload --repository testpypi dist/*

# Publish to PyPI. Guarded: refuses on a dirty tree or an already-published
# version, always rebuilds from scratch, and runs the CI test subset first.
# PyPI uploads are irreversible per version - bump pyproject.toml first.
publish: check-clean-tree check-version-unpublished test-ci build ## Guarded upload to PyPI (token from ~/.pypirc)
	uvx twine upload dist/*

###################
# Export          #
###################

.PHONY: export export-zip

export: ## Export clean source snapshot (no git history)
	@echo "Exporting clean source snapshot..."
	rm -rf $(EXPORT_DIR)
	mkdir -p $(EXPORT_DIR)
	git archive --format=tar HEAD | tar -x -C $(EXPORT_DIR)
	@echo "Exported to $(EXPORT_DIR)"

export-zip: ## Export clean source snapshot as zip
	@echo "Exporting clean source snapshot (zip)..."
	mkdir -p $(EXPORT_BASE)
	git archive --format=zip -o $(EXPORT_BASE)/$(EXPORT_NAME).zip HEAD
	@echo "Exported to $(EXPORT_BASE)/$(EXPORT_NAME).zip"

###################
# Cleanup         #
###################

.PHONY: clean clean-all

clean: ## Remove build artifacts, caches, reports, and logs
	rm -rf build/ dist/ .eggs/ .pytest_cache/ .ruff_cache/ .coverage .coverage.* htmlcov/
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete
	rm -f coverage-*.xml junit-*.xml coverage.xml
	rm -rf reports/
	rm -rf logs/*.log logs/*.log.*

clean-all: clean ## clean + remove the virtual environment and lock file
	rm -rf .venv uv.lock

###################
# Help            #
###################

.PHONY: help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
