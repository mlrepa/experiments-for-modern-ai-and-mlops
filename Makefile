.PHONY: help install test lint format clean run-pipeline check-all
CODE_DIRS = src/ tests/

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial project setup (create venv, install deps, setup hooks)
	uv venv .venv --python 3.12
	@echo "Virtual environment created. Activate it with:"
	@echo "  source .venv/bin/activate  # On Unix/macOS"
	@echo "  .venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install"

install: ## Install dependencies and setup pre-commit hooks
	uv sync --group dev
	uv run pre-commit install
	@echo "Dependencies installed and pre-commit hooks configured!"

install-all: ## Install all optional dependency groups
	uv sync --all-groups
	uv run pre-commit install

sync: ## Sync dependencies (after updating pyproject.toml)
	uv sync --group dev

update: ## Update dependencies to their latest versions
	uv lock --upgrade
	uv sync --group dev

pre-commit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

format: ## Format code using Ruff formatter
	@echo "Formatting code..."
	uv run ruff format $(CODE_DIRS)

format-check: ## Check if code is formatted correctly (for CI)
	@echo "Checking code formatting..."
	uv run ruff format --check $(CODE_DIRS)

lint: ## Check for linting issues using Ruff (no fixes)
	@echo "Linting code..."
	uv run ruff check $(CODE_DIRS)

lint-fix: ## Attempt to automatically fix linting issues using Ruff
	@echo "Attempting to fix linting issues..."
	uv run ruff check --fix $(CODE_DIRS)

type-check: ## Run mypy type checking
	@echo "Running type checks..."
	uv run mypy $(CODE_DIRS) --config-file=pyproject.toml

security: ## Run security checks with bandit
	@echo "Running security checks..."
	uv run bandit -r src/ -f json -o security-report.json || true
	uv run bandit -r src/

autofix: lint-fix format ## Auto-fix linting issues and format code
	@echo "Code auto-fixing and formatting complete."

check-all: lint format-check type-check security test ## Run all code quality checks
	@echo "All checks complete."

clean: ## Clean up generated files and caches
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf security-report.json
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-data: ## Clean all files in data/ directory (except .gitignore)
	@echo "Cleaning data directory..."
	@if [ -d "data" ]; then \
		find data/ -type f ! -name ".gitignore" -delete; \
		echo "Data directory cleaned (kept .gitignore)"; \
	else \
		echo "No data directory found"; \
	fi

clean-models: ## Clean all files in models/ directory (except .gitignore)
	@echo "Cleaning models directory..."
	@if [ -d "models" ]; then \
		find models/ -type f ! -name ".gitignore" -delete; \
		echo "Models directory cleaned (kept .gitignore)"; \
	else \
		echo "No models directory found"; \
	fi

clean-all: clean clean-data clean-models ## Clean everything (dev files, data, and models)
	@echo "Complete cleanup finished!"

venv-info: ## Show virtual environment information
	@echo "Python executable: $$(which python)"
	@echo "Python version: $$(python --version)"
	@echo "Virtual environment: $$VIRTUAL_ENV"
	@echo "Installed packages:"
	@uv pip list


jupyter: ## Start Jupyter Lab
	uv run jupyter lab


# Development utilities
watch-tests: ## Run tests automatically when files change (requires entr)
	find src tests -name "*.py" | entr -c make test

watch-lint: ## Run linting automatically when files change (requires entr)
	find src tests -name "*.py" | entr -c make lint


.DEFAULT_GOAL := help
