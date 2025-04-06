# Code Quality Guidelines

This document outlines the code quality standards and practices implemented in the stockPred project, with a focus on static type checking, linting, and code organization.

## Type Annotations

All code in the stockPred project uses explicit type annotations following Python's typing module. This allows for static type checking using mypy and improves code documentation and maintainability.

### Key Type Annotations

```python
# Basic type annotations
def calculate_returns(prices: List[float]) -> float:
    """Calculate the total return from a list of prices."""
    if not prices:
        return 0.0
    return (prices[-1] / prices[0]) - 1.0

# Complex type annotations
def process_trade_data(
    trades: List[Dict[str, Union[str, float, int, datetime]]],
    start_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process trade data into a DataFrame with summary statistics."""
    # Implementation details...
```

### Common Types Used

- `Optional[T]` - For values that could be None
- `Union[T1, T2, ...]` - For values that could be one of several types
- `Dict[K, V]` - For dictionary types with specific key and value types
- `List[T]` - For lists containing a specific type
- `Tuple[T1, T2, ...]` - For fixed-size tuples with specific types at each position
- `Callable[[Args], ReturnType]` - For function types

## External Libraries and Type Stubs

Some external libraries used in this project don't provide type annotations. We handle this in several ways:

### Libraries with Type Stubs

For libraries with available type stubs, we install the corresponding stub package:

```bash
pip install types-pandas
# Other stub packages as needed
```

### Libraries without Type Stubs

For libraries without type stubs (like 'ta' for technical analysis), we use `# type: ignore` comments:

```python
# Add type ignore comments for modules with missing stubs
from ta.trend import SMAIndicator, MACD  # type: ignore
from ta.momentum import RSIIndicator  # type: ignore
from ta.volatility import BollingerBands  # type: ignore
```

## Type Checking with mypy

We use mypy for static type checking. This helps catch potential type errors before runtime.

### Running mypy

```bash
# Check the entire source code
make lint

# Check a specific file
mypy src/data/stock_data.py
```

### Common mypy Issues and Solutions

| Issue | Solution |
|-------|----------|
| Incompatible return value | Ensure the function returns the correct type or use cast() |
| Untyped import | Add type stubs or use # type: ignore |
| Missing return type | Add -> ReturnType to function signature |
| Statement is unreachable | Restructure code flow or use a different approach |

## Code Organization and Package Structure

### Package Structure

The project is structured as a proper Python package:

- Every directory contains an `__init__.py` file
- A `setup.py` file defines the package for installation
- Code is organized into logical modules

### Environment Variables

We use a `.env` file to set important environment variables:

```
# Path to project root directory 
PYTHONPATH=/Users/username/Projects/stockPred
```

This ensures consistent import behavior regardless of the execution context.

## Linting and Formatting

We use several tools to maintain code quality:

### Pylint

Pylint checks for coding standards and potential errors.

```bash
# Run pylint
make lint
```

### Flake8

Flake8 enforces PEP 8 style guide rules.

```bash
# Run flake8
flake8 src tests
```

### Black

Black automatically formats code to a consistent style.

```bash
# Format all code
make format

# Check if formatting is needed
black --check src tests
```

### isort

isort organizes imports according to best practices.

```bash
# Sort imports
isort src tests
```

## Continuous Integration

We recommend setting up continuous integration to run these checks automatically:

```yaml
# Example GitHub Actions workflow
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install -r requirements-dev.txt
      - name: Lint
        run: make lint
      - name: Test
        run: make test
```

## Best Practices

1. **Always add type annotations** to function parameters and return values
2. **Use Optional[T]** for values that might be None
3. **Be explicit about None returns** with -> Optional[T] or -> None
4. **Import from typing** at the top of each file, before third-party imports
5. **Check your code** with `make lint` before committing
6. **Format your code** with `make format` to maintain consistent style
7. **Add docstrings** with description, Args, and Returns sections

Following these guidelines ensures the codebase remains clean, type-safe, and maintainable. 