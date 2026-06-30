"""Pytest configuration and fixtures for RAG baseline tests."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Configure pytest-asyncio if available
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test."
    )


# Provide asyncio event loop fixture for async tests
try:
    import pytest_asyncio
    # pytest-asyncio is installed, use its loop fixture
except ImportError:
    # pytest-asyncio is not installed, provide a fallback using asyncio.run
    import asyncio
    import pytest

    @pytest.fixture
    def event_loop():
        """Create an instance of the default event loop for each test case."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    # Create a custom asyncio marker handler
    def pytest_collection_modifyitems(items):
        """Handle async tests without pytest-asyncio."""
        for item in items:
            if asyncio.iscoroutinefunction(item.obj):
                # Wrap async test in asyncio.run
                original = item.obj

                def make_sync_test(coro_func):
                    def sync_test(*args, **kwargs):
                        return asyncio.run(coro_func(*args, **kwargs))
                    sync_test.__name__ = coro_func.__name__
                    return sync_test

                item.obj = make_sync_test(original)
