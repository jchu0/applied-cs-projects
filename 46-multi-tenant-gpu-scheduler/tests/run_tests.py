#!/usr/bin/env python3
"""Test runner script for GPU scheduler tests."""

import sys
import os
import pytest
import coverage

def run_tests():
    """Run all tests with coverage reporting."""
    # Start coverage
    cov = coverage.Coverage(source=['../src'])
    cov.start()

    # Run pytest
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(test_dir, '..', 'src'))

    # Test arguments
    args = [
        '-v',  # Verbose
        '--tb=short',  # Short traceback
        '--color=yes',  # Colored output
        test_dir,  # Test directory
    ]

    # Add any additional arguments from command line
    args.extend(sys.argv[1:])

    # Run tests
    exit_code = pytest.main(args)

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    print("\n" + "="*60)
    print("Coverage Report")
    print("="*60)
    cov.report()

    # Generate HTML coverage report
    cov.html_report(directory='htmlcov')
    print("\nDetailed HTML coverage report generated in 'htmlcov' directory")

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())