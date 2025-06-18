.PHONY: install test lint format check-format check-types clean help

# Variables
PYTHON = python
PIP = pip
PYTEST = pytest
FLAKE8 = flake8
BLACK = black
ISORT = isort
MYPY = mypy

# Default target
help:
	@echo "Available targets:"
	@echo "  install     Install the package in development mode"
	@echo "  test        Run tests"
	@echo "  lint        Run linter"
	@echo "  format      Format code with Black and isort"
	@echo "  check-format Check if code is properly formatted"
	@echo "  check-types  Run type checking with mypy"
	@echo "  clean       Remove build artifacts and cache"

# Install the package in development mode
install:
	$(PIP) install -e ".[dev]"

# Run tests
test:
	$(PYTEST) tests/ -v --cov=agentres --cov-report=term-missing

# Run linter
lint:
	$(FLAKE8) src/agentres tests/

# Format code
format:
	$(BLACK) src/agentres tests/
	$(ISORT) src/agentres tests/

# Check code formatting
check-format:
	$(BLACK) --check src/agentres tests/
	$(ISORT) --check-only src/agentres tests/

# Run type checking
check-types:
	$(MYPY) src/agentres

# Clean build artifacts and cache
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
