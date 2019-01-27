import pytest


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_test_marker = pytest.mark.skip(reason="not running integration tests")
    for test in items:
        if "integration" in test.keywords:
            test.add_marker(skip_test_marker)
