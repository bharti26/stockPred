import pytest


@pytest.fixture
def benchmark():
    """Fixture for benchmarking functions."""

    class Benchmark:
        def __call__(self, func, *args, **kwargs):
            """Run the function and return its result."""
            return func(*args, **kwargs)

        def pedantic(self, func, *args, **kwargs):
            """Run the function with pedantic settings."""
            return func(*args, **kwargs)

    return Benchmark()
