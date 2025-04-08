.PHONY: setup install test lint clean benchmark

# Variables
VENV_NAME = stockPred_venv
PYTHON = python3
PIP = pip

setup:
	@echo "Setting up development environment..."
	$(PYTHON) -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV_NAME)/bin/activate && $(PIP) install -e .
	@echo "Development environment setup complete!"

install:
	@echo "Installing package..."
	$(PIP) install -e .
	@echo "Package installed successfully!"

test:
	@echo "Running tests..."
	pytest tests/ -v

lint:
	@echo "Running linters..."
	flake8 src/ tests/
	pylint src/ tests/
	-mypy src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

clean:
	@echo "Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

train:
	@echo "Training models..."
	$(PYTHON) src/train_models.py

dashboard:
	@echo "Starting dashboard..."
	$(PYTHON) src/dashboard/app.py

benchmark:
	@echo "Running performance benchmarks..."
	$(PYTHON) -m pytest -c pytest_benchmark.ini tests/test_benchmarks.py

help:
	@echo "Available commands:"
	@echo "  make setup     - Set up development environment"
	@echo "  make install   - Install package"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run linters"
	@echo "  make format    - Format code"
	@echo "  make clean     - Clean up build files"
	@echo "  make train     - Train models"
	@echo "  make dashboard - Start dashboard"
	@echo "  make benchmark - Run performance benchmarks" 