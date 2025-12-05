// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <qdk/chemistry/utils/logger.hpp>

namespace py = pybind11;

// Extract script name from file path (without extension)
std::string extract_script_name(const std::string& filepath) {
  size_t last_slash = filepath.find_last_of("/\\");
  std::string filename = (last_slash != std::string::npos)
                             ? filepath.substr(last_slash + 1)
                             : filepath;

  size_t last_dot = filename.find_last_of('.');
  if (last_dot != std::string::npos) {
    filename = filename.substr(0, last_dot);
  }
  return filename;
}

// Helper to get Python caller context (just module path, like C++ version)
std::string get_python_context() {
  try {
    py::module_ inspect = py::module_::import("inspect");
    py::object stack = inspect.attr("stack")();

    // Walk up the stack to find the actual caller
    // Skip frames that are internal (pybind, builtins, etc.)
    size_t stack_len = py::len(stack);
    for (size_t i = 0; i < stack_len; ++i) {
      py::object frame_info = stack[py::int_(i)];
      py::object frame = frame_info.attr("frame");
      py::object globals = frame.attr("f_globals");

      std::string module_name = "unknown";
      if (globals.contains("__name__")) {
        module_name = globals["__name__"].cast<std::string>();
      }

      // Skip internal modules - we want the actual user/library code
      // Skip if module starts with underscore (like _core) or is a built-in
      // But allow __main__ - we'll handle it specially
      if (module_name.empty() || module_name == "unknown" ||
          (module_name.rfind("_", 0) == 0 && module_name != "__main__") ||
          module_name == "builtins") {
        continue;
      }

      // Handle __main__ - extract script name from filename
      if (module_name == "__main__") {
        std::string filepath = frame_info.attr("filename").cast<std::string>();
        return extract_script_name(filepath);
      }

      // Return just the module path (like C++ version)
      return module_name;
    }
  } catch (...) {
    // If inspection fails, return unknown
  }
  return "python.unknown";
}

// Helper to get Python caller context WITH function name (for trace_entering)
std::pair<std::string, std::string> get_python_context_with_function() {
  try {
    py::module_ inspect = py::module_::import("inspect");
    py::object stack = inspect.attr("stack")();

    size_t stack_len = py::len(stack);
    for (size_t i = 0; i < stack_len; ++i) {
      py::object frame_info = stack[py::int_(i)];
      py::object frame = frame_info.attr("frame");
      py::object globals = frame.attr("f_globals");

      std::string module_name = "unknown";
      if (globals.contains("__name__")) {
        module_name = globals["__name__"].cast<std::string>();
      }

      if (module_name.empty() || module_name == "unknown" ||
          (module_name.rfind("_", 0) == 0 && module_name != "__main__") ||
          module_name == "builtins") {
        continue;
      }

      // Handle __main__ - extract script name from filename
      if (module_name == "__main__") {
        std::string filepath = frame_info.attr("filename").cast<std::string>();
        module_name = extract_script_name(filepath);
      }

      std::string func_name = frame_info.attr("function").cast<std::string>();
      return {module_name, func_name};
    }
  } catch (...) {
  }
  return {"python.unknown", "unknown"};
}

void bind_logger(py::module& m) {
  auto logger_module = m.def_submodule("Logger", R"(
Logging utilities for QDK Chemistry.

This module provides a unified logging interface that shares a single global
logger instance with the C++ backend. All log messages from both Python and
C++ code appear in the same output stream with consistent formatting.

Typical usage:

.. code-block:: python

    from qdk_chemistry.utils import Logger

    # Set the global log level
    Logger.set_global_level("debug")

    # Log messages with automatic context (module path)
    Logger.info("Starting calculation")
    Logger.debug(f"Processing {num_atoms} atoms")

    # Log function entry (like C++ QDK_LOG_TRACE_ENTERING)
    def my_function():
        Logger.trace_entering()  # Logs: [module.path] Entering my_function
        ...

Output format::

    [2025-12-03 10:30:00.123456] [info] [qdk_chemistry.algorithms.scf] Starting calculation

See Also:
    :class:`LogLevel`: Enumeration of available log levels

)");

  // Bind the LogLevel enum
  py::enum_<qdk::chemistry::utils::LogLevel>(logger_module, "LogLevel", R"(
Log level enumeration for controlling logging verbosity.

Levels are ordered from most verbose (trace) to least verbose (critical).
Setting a level filters out all messages below that level.

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.set_global_level(Logger.LogLevel.debug)
    >>> # Or using string:
    >>> Logger.set_global_level("debug")

)")
      .value("trace", qdk::chemistry::utils::LogLevel::trace,
             "Most verbose logging, for detailed tracing")
      .value("debug", qdk::chemistry::utils::LogLevel::debug,
             "Debug information for development")
      .value("info", qdk::chemistry::utils::LogLevel::info,
             "General informational messages")
      .value("warn", qdk::chemistry::utils::LogLevel::warn,
             "Warning messages for potential issues")
      .value("error", qdk::chemistry::utils::LogLevel::error,
             "Error messages for failures")
      .value("critical", qdk::chemistry::utils::LogLevel::critical,
             "Critical errors requiring immediate attention")
      .value("off", qdk::chemistry::utils::LogLevel::off,
             "Disable all logging");

  // Set global level - affects both C++ and Python
  logger_module.def("set_global_level",
                    &qdk::chemistry::utils::Logger::set_global_level,
                    py::arg("level"),
                    R"(
Set the global logging level using a LogLevel enum value.

Changes the logging level for both Python and C++ code. Messages below
this level will be suppressed.

Args:
    level (LogLevel): The minimum log level to output

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.set_global_level(Logger.LogLevel.debug)

See Also:
    :meth:`get_global_level`: Get the current log level

)");

  logger_module.def(
      "set_global_level",
      [](const std::string& level) {
        if (level == "trace") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::trace);
        } else if (level == "debug") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::debug);
        } else if (level == "info") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::info);
        } else if (level == "warn") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::warn);
        } else if (level == "error") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::error);
        } else if (level == "critical") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::critical);
        } else if (level == "off") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::off);
        } else {
          throw std::invalid_argument("Invalid log level: " + level);
        }
      },
      py::arg("level"),
      R"(
Set the global logging level using a string.

Convenience overload that accepts level names as strings.

Args:
    level (str): Log level name. Valid values: "trace", "debug", "info",
        "warn", "error", "critical", "off"

Raises:
    ValueError: If an invalid level string is provided

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.set_global_level("debug")
    >>> Logger.set_global_level("info")

)");

  logger_module.def(
      "get_global_level",
      []() -> std::string {
        auto level = qdk::chemistry::utils::Logger::get_global_level();
        switch (level) {
          case qdk::chemistry::utils::LogLevel::trace:
            return "trace";
          case qdk::chemistry::utils::LogLevel::debug:
            return "debug";
          case qdk::chemistry::utils::LogLevel::info:
            return "info";
          case qdk::chemistry::utils::LogLevel::warn:
            return "warn";
          case qdk::chemistry::utils::LogLevel::error:
            return "error";
          case qdk::chemistry::utils::LogLevel::critical:
            return "critical";
          case qdk::chemistry::utils::LogLevel::off:
            return "off";
          default:
            return "unknown";
        }
      },
      R"(
Get the current global logging level.

Returns:
    str: The current log level as a string ("trace", "debug", "info",
        "warn", "error", "critical", or "off")

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.get_global_level()
    'info'
    >>> Logger.set_global_level("debug")
    >>> Logger.get_global_level()
    'debug'

)");

  logger_module.def(
      "load_env_levels", []() { spdlog::cfg::load_env_levels(); },
      R"(
Load logging levels from environment variables.

Reads the SPDLOG_LEVEL environment variable to configure logging.
This allows runtime configuration without code changes.

Examples:
    Set environment variable before running Python::

        $ export SPDLOG_LEVEL=debug
        $ python my_script.py

    Then in Python::

        >>> from qdk_chemistry.utils import Logger
        >>> Logger.load_env_levels()  // Applies SPDLOG_LEVEL=debug

)");

  // Get the single global logger (same instance as C++ uses)
  logger_module.def(
      "get", []() { return qdk::chemistry::utils::Logger::get(); },
      R"(
Get the raw spdlog logger instance.

Returns the underlying spdlog logger shared with C++ code. This is
useful for advanced usage or direct spdlog API access.

Note:
    For most use cases, prefer the convenience methods like
    :meth:`info`, :meth:`debug`, etc., which automatically include
    source context.

Returns:
    SpdLogger: The global spdlog logger instance

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> raw_logger = Logger.get()
    >>> raw_logger.info("Raw message without context prefix")

)");

  // Convenience logging functions with automatic Python context
  logger_module.def(
      "trace",
      [](const std::string& msg) {
        auto ctx = get_python_context();
        qdk::chemistry::utils::Logger::get()->trace("[{}] {}", ctx, msg);
      },
      py::arg("msg"),
      R"(
Log a trace-level message with automatic context.

Trace is the most verbose level, used for detailed debugging and
tracing program flow.

Args:
    msg (str): The message to log

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.set_global_level("trace")
    >>> Logger.trace("Entering loop iteration 5")
    [2025-12-03 10:30:00.123456] [trace] [my_module] Entering loop iteration 5

)");

  logger_module.def(
      "debug",
      [](const std::string& msg) {
        auto ctx = get_python_context();
        qdk::chemistry::utils::Logger::get()->debug("[{}] {}", ctx, msg);
      },
      py::arg("msg"),
      R"(
Log a debug-level message with automatic context.

Debug messages provide information useful during development and
debugging.

Args:
    msg (str): The message to log

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.set_global_level("debug")
    >>> Logger.debug(f"Processing {n} atoms")
    [2025-12-03 10:30:00.123456] [debug] [my_module] Processing 10 atoms

)");

  logger_module.def(
      "info",
      [](const std::string& msg) {
        auto ctx = get_python_context();
        qdk::chemistry::utils::Logger::get()->info("[{}] {}", ctx, msg);
      },
      py::arg("msg"),
      R"(
Log an info-level message with automatic context.

Info messages provide general information about program progress.

Args:
    msg (str): The message to log

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.info("Calculation completed successfully")
    [2025-12-03 10:30:00.123456] [info] [my_module] Calculation completed successfully

)");

  logger_module.def(
      "warn",
      [](const std::string& msg) {
        auto ctx = get_python_context();
        qdk::chemistry::utils::Logger::get()->warn("[{}] {}", ctx, msg);
      },
      py::arg("msg"),
      R"(
Log a warning-level message with automatic context.

Warnings indicate potential issues that don't prevent execution but
may require attention.

Args:
    msg (str): The message to log

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.warn("Convergence threshold not met, using best result")

)");

  logger_module.def(
      "error",
      [](const std::string& msg) {
        auto ctx = get_python_context();
        qdk::chemistry::utils::Logger::get()->error("[{}] {}", ctx, msg);
      },
      py::arg("msg"),
      R"(
Log an error-level message with automatic context.

Error messages indicate failures that may affect results but don't
necessarily terminate the program.

Args:
    msg (str): The message to log

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.error("Failed to read configuration file")

)");

  logger_module.def(
      "critical",
      [](const std::string& msg) {
        auto ctx = get_python_context();
        qdk::chemistry::utils::Logger::get()->critical("[{}] {}", ctx, msg);
      },
      py::arg("msg"),
      R"(
Log a critical-level message with automatic context.

Critical messages indicate severe errors that likely require
immediate attention or program termination.

Args:
    msg (str): The message to log

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.critical("Memory allocation failed, cannot continue")

)");

  // Trace entering - like C++ QDK_LOG_TRACE_ENTERING macro
  logger_module.def(
      "trace_entering",
      []() {
        auto [module_name, func_name] = get_python_context_with_function();
        // Warn if called at module level - <module> is not a real function
        if (func_name == "<module>") {
          qdk::chemistry::utils::Logger::get()->warn(
              "[{}] Logger.trace_entering() called at module level; "
              "should only be used inside functions",
              module_name);
          return;
        }
        qdk::chemistry::utils::Logger::get()->trace("[{}] Entering {}",
                                                    module_name, func_name);
      },
      R"(
Log function entry at trace level with automatic context.

This is the Python equivalent of the C++ ``QDK_LOG_TRACE_ENTERING()`` macro.
It automatically detects the calling function name and logs an "Entering"
message, useful for tracing function call flow during debugging.

Note:
    This should only be called from within a function. If called at module
    level (outside any function), a warning is logged instead.

Examples:
    >>> from qdk_chemistry.utils import Logger
    >>> Logger.set_global_level("trace")
    >>>
    >>> def calculate_energy():
    ...     Logger.trace_entering()  # Logs: [my_module] Entering calculate_energy
    ...     # ... function implementation
    ...
    >>> calculate_energy()
    [2025-12-03 10:30:00.123456] [trace] [my_module] Entering calculate_energy

See Also:
    :meth:`trace`: For custom trace messages

)");

  // Bind spdlog logger class with essential methods (for direct access)
  py::class_<spdlog::logger, std::shared_ptr<spdlog::logger>>(logger_module,
                                                              "SpdLogger",
                                                              R"(
Raw spdlog logger interface.

This class provides direct access to the underlying spdlog logger.
Messages logged through this interface do not include automatic
source context prefixes.

Note:
    For most use cases, prefer the module-level functions like
    ``Logger.info()``, ``Logger.debug()``, etc., which automatically
    include source context.

)")
      .def(
          "trace",
          [](spdlog::logger& logger, const std::string& msg) {
            logger.trace(msg);
          },
          py::arg("msg"), "Log a raw trace message without context prefix")
      .def(
          "debug",
          [](spdlog::logger& logger, const std::string& msg) {
            logger.debug(msg);
          },
          py::arg("msg"), "Log a raw debug message without context prefix")
      .def(
          "info",
          [](spdlog::logger& logger, const std::string& msg) {
            logger.info(msg);
          },
          py::arg("msg"), "Log a raw info message without context prefix")
      .def(
          "warn",
          [](spdlog::logger& logger, const std::string& msg) {
            logger.warn(msg);
          },
          py::arg("msg"), "Log a raw warning message without context prefix")
      .def(
          "error",
          [](spdlog::logger& logger, const std::string& msg) {
            logger.error(msg);
          },
          py::arg("msg"), "Log a raw error message without context prefix")
      .def(
          "critical",
          [](spdlog::logger& logger, const std::string& msg) {
            logger.critical(msg);
          },
          py::arg("msg"), "Log a raw critical message without context prefix")
      .def("name", &spdlog::logger::name,
           "Get the logger's name (returns 'qdk-chemistry')");
}
