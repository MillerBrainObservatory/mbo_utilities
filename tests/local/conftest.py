# local test configuration
# these tests require local data and are not run in CI

import pytest


def pytest_configure(config):
    """register custom markers."""
    config.addinivalue_line("markers", "local: tests requiring local data")


def pytest_collection_modifyitems(config, items):
    """add local marker to all tests in this directory."""
    for item in items:
        if "local" in str(item.fspath):
            item.add_marker(pytest.mark.local)
