"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(autouse=True)
def cleanup_writers():
    """Reset all TBWriter caches before and after each test.

    This ensures tests don't interfere with each other through shared writer state.
    """
    from printboard.writer import TBWriter

    TBWriter.reset()
    yield
    TBWriter.reset()
