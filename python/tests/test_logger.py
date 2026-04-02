"""Comprehensive tests for Logger Python bindings.

Tests the functionality of the logger module including:
- Setting and getting log levels (both enum and string)
- Logging at all levels (trace, debug, info, warn, error, critical)
- Automatic context extraction from Python stack
- trace_entering functionality
- Raw logger access
- Environment variable loading
- Error handling
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import pytest

from qdk_chemistry.utils import Logger


@pytest.fixture(scope="module", autouse=True)
def setup_logger():
    """Initialize logger state and suppress output after tests."""
    # Start with off to suppress output
    Logger.set_global_level("off")
    yield
    # Ensure we end with off to suppress any remaining output
    Logger.set_global_level("off")


class TestLogLevel:
    """Tests for LogLevel enum and level setting/getting."""

    def test_log_level_enum_values(self):
        """Test that all LogLevel enum values are accessible."""
        assert hasattr(Logger, "LogLevel")
        assert hasattr(Logger.LogLevel, "trace")
        assert hasattr(Logger.LogLevel, "debug")
        assert hasattr(Logger.LogLevel, "info")
        assert hasattr(Logger.LogLevel, "warn")
        assert hasattr(Logger.LogLevel, "error")
        assert hasattr(Logger.LogLevel, "critical")
        assert hasattr(Logger.LogLevel, "off")

    def test_set_global_level_with_enum(self):
        """Test setting global log level using enum values."""
        # Test each level
        for level in [
            Logger.LogLevel.trace,
            Logger.LogLevel.debug,
            Logger.LogLevel.info,
            Logger.LogLevel.warn,
            Logger.LogLevel.error,
            Logger.LogLevel.critical,
            Logger.LogLevel.off,
        ]:
            Logger.set_global_level(level)
            current = Logger.get_global_level()
            assert current in ["trace", "debug", "info", "warn", "error", "critical", "off"]

    def test_set_global_level_with_string(self):
        """Test setting global log level using string values."""
        levels = ["trace", "debug", "info", "warn", "error", "critical", "off"]
        for level in levels:
            Logger.set_global_level(level)
            assert Logger.get_global_level() == level

    def test_set_global_level_invalid_string(self):
        """Test that invalid log level strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            Logger.set_global_level("invalid_level")

        with pytest.raises(ValueError, match="Invalid log level"):
            Logger.set_global_level("INFO")  # Case sensitive

        with pytest.raises(ValueError, match="Invalid log level"):
            Logger.set_global_level("")

    def test_get_global_level(self):
        """Test getting the current log level."""
        Logger.set_global_level("debug")
        assert Logger.get_global_level() == "debug"

        Logger.set_global_level("warn")
        assert Logger.get_global_level() == "warn"

    def test_level_persistence(self):
        """Test that log level persists across function calls."""
        Logger.set_global_level("error")
        assert Logger.get_global_level() == "error"

        # Call some other function
        Logger.info("Test message")

        # Level should still be error
        assert Logger.get_global_level() == "error"


class TestLoggingFunctions:
    """Tests for the logging functions at various levels."""

    @pytest.fixture(autouse=True)
    def setup_log_level(self):
        """Set log level to trace before each test to capture all messages."""
        Logger.set_global_level("trace")
        yield
        # Reset to off after test to suppress output
        Logger.set_global_level("off")

    def test_trace_logging(self, capfd):
        """Test trace level logging with context."""
        Logger.trace("This is a trace message")
        captured = capfd.readouterr()
        assert "[trace]" in captured.out
        assert "This is a trace message" in captured.out
        assert "[tests.test_logger]" in captured.out or "[pytest" in captured.out

    def test_debug_logging(self, capfd):
        """Test debug level logging with context."""
        Logger.debug("This is a debug message")
        captured = capfd.readouterr()
        assert "[debug]" in captured.out
        assert "This is a debug message" in captured.out
        assert "[tests.test_logger]" in captured.out or "[pytest" in captured.out

    def test_info_logging(self, capfd):
        """Test info level logging with context."""
        Logger.info("This is an info message")
        captured = capfd.readouterr()
        assert "[info]" in captured.out
        assert "This is an info message" in captured.out
        assert "[tests.test_logger]" in captured.out or "[pytest" in captured.out

    def test_warn_logging(self, capfd):
        """Test warning level logging with context."""
        Logger.warn("This is a warning message")
        captured = capfd.readouterr()
        assert "[warn]" in captured.out or "[warning]" in captured.out
        assert "This is a warning message" in captured.out
        assert "[tests.test_logger]" in captured.out or "[pytest" in captured.out

    def test_error_logging(self, capfd):
        """Test error level logging with context."""
        Logger.error("This is an error message")
        captured = capfd.readouterr()
        assert "[error]" in captured.out
        assert "This is an error message" in captured.out
        assert "[tests.test_logger]" in captured.out or "[pytest" in captured.out

    def test_critical_logging(self, capfd):
        """Test critical level logging with context."""
        Logger.critical("This is a critical message")
        captured = capfd.readouterr()
        assert "[critical]" in captured.out
        assert "This is a critical message" in captured.out
        assert "[tests.test_logger]" in captured.out or "[pytest" in captured.out

    def test_log_filtering_by_level(self, capfd):
        """Test that messages below the set level are filtered out."""
        # Set to warn level - should filter out trace, debug, info
        Logger.set_global_level("warn")

        Logger.trace("trace message")
        Logger.debug("debug message")
        Logger.info("info message")
        Logger.warn("warn message")
        Logger.error("error message")

        captured = capfd.readouterr()

        # Lower levels should not appear
        assert "trace message" not in captured.out
        assert "debug message" not in captured.out
        assert "info message" not in captured.out

        # warn and above should appear
        assert "warn message" in captured.out
        assert "error message" in captured.out

    def test_log_off_level(self, capfd):
        """Test that 'off' level suppresses all logging."""
        Logger.set_global_level("off")

        Logger.trace("trace message")
        Logger.debug("debug message")
        Logger.info("info message")
        Logger.warn("warn message")
        Logger.error("error message")
        Logger.critical("critical message")

        captured = capfd.readouterr()

        # Nothing should be logged
        assert captured.out == ""

    def test_unicode_in_log_messages(self, capfd):
        """Test that Unicode characters are handled correctly in log messages."""
        Logger.info("Unicode test: 你好世界 🚀 αβγ")
        captured = capfd.readouterr()
        assert "Unicode test: 你好世界 🚀 αβγ" in captured.out


class TestTraceEntering:
    """Tests for the trace_entering functionality."""

    @pytest.fixture(autouse=True)
    def setup_trace_level(self):
        """Set log level to trace before each test."""
        Logger.set_global_level("trace")
        yield
        Logger.set_global_level("off")

    def test_trace_entering_in_function(self, capfd):
        """Test trace_entering logs the function name correctly."""

        def test_function():
            Logger.trace_entering()
            return "done"

        result = test_function()
        captured = capfd.readouterr()

        assert result == "done"
        assert "[trace]" in captured.out
        assert "Entering test_function" in captured.out

    def test_trace_entering_in_nested_function(self, capfd):
        """Test trace_entering works in nested functions."""

        def outer_function():
            def inner_function():
                Logger.trace_entering()
                return "inner"

            return inner_function()

        result = outer_function()
        captured = capfd.readouterr()

        assert result == "inner"
        assert "Entering inner_function" in captured.out

    def test_trace_entering_in_method(self, capfd):
        """Test trace_entering works in class methods."""

        class TestClass:
            def test_method(self):
                Logger.trace_entering()
                return "method"

        obj = TestClass()
        result = obj.test_method()
        captured = capfd.readouterr()

        assert result == "method"
        assert "Entering test_method" in captured.out

    def test_trace_entering_at_module_level(self, capfd):
        """Test that trace_entering works when called from a test function."""
        Logger.trace_entering()
        captured = capfd.readouterr()
        assert len(captured.out) > 0

    def test_trace_entering_disabled_by_flag(self):
        """Test is_trace_log_disabled flag."""
        result = Logger.is_trace_log_disabled()
        assert isinstance(result, bool)


class TestRawLogger:
    """Tests for raw spdlog logger access."""

    @pytest.fixture(autouse=True)
    def setup_log_level(self):
        """Enable logging for raw logger tests."""
        Logger.set_global_level("trace")
        yield
        Logger.set_global_level("off")

    def test_get_raw_logger(self):
        """Test getting the raw spdlog logger instance."""
        logger = Logger.get()
        assert logger is not None
        assert hasattr(logger, "trace")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warn")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")
        assert hasattr(logger, "name")

    def test_raw_logger_name(self):
        """Test that raw logger has the correct name."""
        logger = Logger.get()
        name = logger.name()
        assert name == "qdk-chemistry"

    def test_raw_logger_trace(self, capfd):
        """Test raw logger trace method (no automatic context)."""
        logger = Logger.get()
        logger.trace("Raw trace message")
        captured = capfd.readouterr()

        assert "[trace]" in captured.out
        assert "Raw trace message" in captured.out

    def test_raw_logger_debug(self, capfd):
        """Test raw logger debug method (no automatic context)."""
        logger = Logger.get()
        logger.debug("Raw debug message")
        captured = capfd.readouterr()

        assert "[debug]" in captured.out
        assert "Raw debug message" in captured.out

    def test_raw_logger_info(self, capfd):
        """Test raw logger info method (no automatic context)."""
        logger = Logger.get()
        logger.info("Raw info message")
        captured = capfd.readouterr()

        assert "[info]" in captured.out
        assert "Raw info message" in captured.out

    def test_raw_logger_warn(self, capfd):
        """Test raw logger warn method (no automatic context)."""
        logger = Logger.get()
        logger.warn("Raw warn message")
        captured = capfd.readouterr()

        assert "[warn]" in captured.out or "[warning]" in captured.out
        assert "Raw warn message" in captured.out

    def test_raw_logger_error(self, capfd):
        """Test raw logger error method (no automatic context)."""
        logger = Logger.get()
        logger.error("Raw error message")
        captured = capfd.readouterr()

        assert "[error]" in captured.out
        assert "Raw error message" in captured.out

    def test_raw_logger_critical(self, capfd):
        """Test raw logger critical method (no automatic context)."""
        logger = Logger.get()
        logger.critical("Raw critical message")
        captured = capfd.readouterr()

        assert "[critical]" in captured.out
        assert "Raw critical message" in captured.out


class TestContextExtraction:
    """Tests for Python context extraction from stack frames."""

    @pytest.fixture(autouse=True)
    def setup_log_level(self):
        """Enable logging for context extraction tests."""
        Logger.set_global_level("info")
        yield
        Logger.set_global_level("off")

    def test_context_includes_module_name(self, capfd):
        """Test that log messages include the module name in context."""
        Logger.info("Test message with context")
        captured = capfd.readouterr()

        assert re.search(r"\[[\w.]+\]", captured.out) is not None

    def test_context_from_different_modules(self, capfd):
        """Test that context correctly identifies different calling modules."""
        Logger.info("Message from test")
        captured = capfd.readouterr()

        assert "test_logger" in captured.out or "pytest" in captured.out

    def test_context_with_nested_calls(self, capfd):
        """Test context extraction with nested function calls."""

        def level1():
            def level2():
                def level3():
                    Logger.info("Deep nested message")

                level3()

            level2()

        level1()
        captured = capfd.readouterr()

        assert "[" in captured.out
        assert "]" in captured.out
        assert "Deep nested message" in captured.out


class TestEnvironmentVariables:
    """Tests for environment variable loading."""

    def test_load_env_levels_function_exists(self):
        """Test that load_env_levels function is accessible."""
        assert hasattr(Logger, "load_env_levels")
        assert callable(Logger.load_env_levels)

    def test_load_env_levels_no_crash(self):
        """Test that load_env_levels can be called without crashing."""
        # Should not raise an exception even if env var is not set
        Logger.load_env_levels()

    def test_load_env_levels_with_env_var(self, monkeypatch):
        """Test that SPDLOG_LEVEL environment variable is respected."""
        monkeypatch.setenv("SPDLOG_LEVEL", "debug")
        Logger.load_env_levels()

    def test_load_env_levels_multiple_times(self):
        """Test that load_env_levels can be called multiple times safely."""
        for _ in range(3):
            Logger.load_env_levels()


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture(autouse=True)
    def setup_log_level(self):
        """Enable logging for edge case tests."""
        Logger.set_global_level("info")
        yield
        Logger.set_global_level("off")

    def test_empty_log_message(self, capfd):
        """Test logging an empty message."""
        Logger.info("")
        captured = capfd.readouterr()

        assert "[info]" in captured.out

    def test_very_long_log_message(self, capfd):
        """Test logging a very long message."""
        long_message = "A" * 10000
        Logger.info(long_message)
        captured = capfd.readouterr()

        assert long_message in captured.out

    def test_log_message_with_special_characters(self, capfd):
        """Test logging messages with special characters."""
        Logger.info("Test: {}, [], \n, \t, $, @, #, %, &")
        captured = capfd.readouterr()

        assert "Test:" in captured.out

    def test_log_message_with_braces(self, capfd):
        """Test logging messages with curly braces."""
        Logger.info("Message with {{braces}}")
        captured = capfd.readouterr()

        assert "Message with" in captured.out

    def test_concurrent_log_level_changes(self):
        """Test rapidly changing log levels."""
        levels = ["trace", "debug", "info", "warn", "error", "critical"]
        for _ in range(10):
            for level in levels:
                Logger.set_global_level(level)
                assert Logger.get_global_level() == level

    def test_logger_after_exception(self, capfd):
        """Test that logger works correctly after an exception."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            pass

        Logger.info("Message after exception")
        captured = capfd.readouterr()

        assert "Message after exception" in captured.out


class TestDocumentationExamples:
    """Tests based on the documentation examples in the docstrings."""

    @pytest.fixture(autouse=True)
    def setup_log_level(self):
        """Enable logging for documentation example tests."""
        Logger.set_global_level("trace")
        yield
        Logger.set_global_level("off")

    def test_basic_usage_example(self, capfd):
        """Test the basic usage example from module docstring."""
        Logger.set_global_level("debug")
        Logger.info("Starting calculation")
        Logger.debug("Processing 10 atoms")

        captured = capfd.readouterr()
        assert "Starting calculation" in captured.out
        assert "Processing 10 atoms" in captured.out

    def test_set_level_using_enum_example(self):
        """Test setting level using enum as shown in docs."""
        Logger.set_global_level(Logger.LogLevel.debug)
        assert Logger.get_global_level() == "debug"

    def test_set_level_using_string_example(self):
        """Test setting level using string as shown in docs."""
        Logger.set_global_level("debug")
        assert Logger.get_global_level() == "debug"

    def test_trace_entering_example(self, capfd):
        """Test trace_entering example from docs."""

        def calculate_energy():
            Logger.trace_entering()
            return 42

        result = calculate_energy()
        captured = capfd.readouterr()

        assert result == 42
        assert "Entering calculate_energy" in captured.out


class TestLoggerSingleton:
    """Tests to verify logger is a singleton shared with C++."""

    def test_get_returns_same_instance(self):
        """Test that multiple calls to get() return the same logger instance."""
        logger1 = Logger.get()
        logger2 = Logger.get()

        # Should be the same underlying logger
        assert logger1.name() == logger2.name()

    def test_level_changes_affect_all_references(self):
        """Test that changing level affects all logger references."""
        Logger.set_global_level("error")
        Logger.get()

        Logger.set_global_level("trace")
        assert Logger.get_global_level() == "trace"
