import pytest
from src.logger import Logger

@pytest.fixture
def logger():
    return Logger()

def test_info_log(logger):
    logger.info("This is an info message.")
    log_content = logger.read_log_file()
    assert "INFO" in log_content
    assert "This is an info message." in log_content

def test_warning_log(logger):
    logger.warning("This is a warning message.")
    log_content = logger.read_log_file()
    assert "WARNING" in log_content
    assert "This is a warning message." in log_content

def test_error_log(logger):
    logger.error("This is an error message.")
    log_content = logger.read_log_file()
    assert "ERROR" in log_content
    assert "This is an error message." in log_content

def test_debug_log(logger):
    logger.debug("This is a debug message.")
    log_content = logger.read_log_file()
    assert "DEBUG" in log_content
    assert "This is a debug message." in log_content

def test_exception_log(logger):
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        logger.exception("This is an exception message.")
    log_content = logger.read_log_file()
    assert "ERROR" in log_content
    assert "This is an exception message." in log_content
    assert "Test exception" in log_content 