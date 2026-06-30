"""Test suite for the ML Compiler project.

This test suite provides comprehensive testing for the ML compiler,
including unit tests, integration tests, and performance benchmarks.

Test files:
- test_cuda_codegen.py: CUDA code generation correctness tests
- test_ir_operations_actual.py: IR operations and transformations tests
- test_memory_planning.py: Memory planning and allocation tests
- test_optimization_passes_actual.py: Optimization passes tests
"""

# Note: The original test_* files use mock classes that don't exist in the
# actual implementation. The new *_actual.py tests use the real implementation.
