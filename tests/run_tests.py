import os
import sys

import pytest

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests():
    """Run all test suites using pytest"""
    # Run pytest on the tests directory
    result = pytest.main(["-v", os.path.dirname(os.path.abspath(__file__))])
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
