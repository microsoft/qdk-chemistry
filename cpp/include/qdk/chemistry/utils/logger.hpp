// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <source_location>  // Added for automatic source-based naming
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace qdk::chemistry::utils {

/**
 * @enum LogLevel
 * @brief Log level enumeration for QDK Chemistry logging
 */
enum class LogLevel {
  trace,     ///< Most verbose logging
  debug,     ///< Debug information
  info,      ///< General information
  warn,      ///< Warning messages
  error,     ///< Error messages
  critical,  ///< Critical errors
  off        ///< Disable logging
};

/**
 * @class Logger
 * @brief Centralized logging utility wrapper around spdlog for QDK Chemistry
 *
 * This class provides a consistent interface for logging throughout the QDK
 * Chemistry library, wrapping the spdlog library with project-specific defaults
 * and conventions. It supports named logger instances, global log level
 * control, and provides convenience functions for common logging operations.
 *
 * Features
 * --------
 * - Named logger instances with automatic creation based on source location
 * - Global log level control across all loggers
 * - Consistent formatting and output configuration
 * - Thread-safe operation (inherited from spdlog)
 * - Convenience macros for easy logger access
 *
 * Usage Example
 * -------------
 * ```cpp
 * #include <qdk/chemistry/utils/logger.hpp>
 *
 * // Get a logger for your component (automatically named from source location)
 * auto logger = qdk::chemistry::utils::Logger::get();
 * logger->info("Starting calculation");
 * logger->warn("Parameter {} may be out of range: {}", param_name, value);
 *
 * // Or use the convenience macro
 * auto logger2 = QDK_LOGGER();
 * logger2->debug("Debug information: {}", data);
 *
 * // Log function entry with convenience macro
 * void myFunction() {
 *   LOG_TRACE_ENTERING();  // Automatically logs "Entering myFunction"
 *   // ... function implementation
 * }
 *
 * // Control global logging
 * Logger::set_global_level(LogLevel::debug);  // Enable debug output
 * Logger::disable_all();                      // Disable all logging
 * ```
 *
 * Thread Safety
 * -------------
 * This class is thread-safe as it wraps spdlog, which provides thread-safe
 * logging operations. Multiple threads can safely use the same logger instance
 * or create different logger instances simultaneously.
 */
class Logger {
 public:
  /**
   * @brief Get or create a logger instance based on source location
   *
   * Returns an existing logger derived from the calling context (via
   * std::source_location). Loggers are automatically named based
   * on the class and method from which they are called.
   *
   * New loggers are created with default settings:
   * - Colored console output
   * - Inherit global log level
   * - Consistent timestamped format
   *
   * @param location Automatically provided source location of the caller
   * @return Shared pointer to the logger instance
   */
  static std::shared_ptr<spdlog::logger> get(
      const std::source_location& location = std::source_location::current());

  /**
   * @brief Set the global log level for all loggers
   *
   * Changes the logging level for all existing and future logger instances.
   * Messages below this level will be suppressed.
   *
   * @param level The minimum log level to output
   */
  static void set_global_level(LogLevel level);

  /**
   * @brief Disable all logging output
   *
   * Convenience function to completely disable all logging by setting
   * the global level to 'off'. This is equivalent to calling
   * set_global_level(LogLevel::off).
   *
   * @note This is useful for performance-critical sections or when
   *       running in silent mode
   */
  static void disable_all();
};

/**
 * @brief Convert a filesystem path to colon-separated string starting from a
 * given segment
 *
 * @param file_path Full path to the file
 * @param start_segment The segment from which to start (default "qdk")
 * @return Colon-separated string
 */
inline std::string path_to_colon_string(
    const std::string& file_path, const std::string& start_segment = "qdk");

/**
 * @brief Extract a clean method/function name from a mangled function signature
 *
 * This function processes C++ function signatures (as provided by
 * std::source_location::function_name()) and extracts a human-readable
 * method or function name suitable for logging purposes.
 *
 * Processing Steps:
 * 1. Removes lambda expressions (everything after "::<lambda")
 * 2. Strips function parameter lists (everything after the last '(')
 * 3. Extracts the final component after the last "::" (method/function name)
 * 4. Detects constructors and appends " constructor" for clarity
 *
 * Examples:
 * - "MyClass::myMethod(int, bool)" → "myMethod"
 * - "namespace::MyClass::MyClass()" → "MyClass constructor"
 * - "globalFunction(std::string)" → "globalFunction"
 * - "MyClass::operator[](size_t)" → "operator[]"
 * - "lambda expressions" → strips lambda suffix
 *
 * @param func_name The raw function name from
 * std::source_location::function_name()
 * @return Clean, human-readable method/function name suitable for logging
 *
 * @note This function handles various C++ constructs including:
 *       - Namespaced functions and methods
 *       - Constructor detection and labeling
 *       - Lambda expression cleanup
 *       - Operator overloading
 *       - Template instantiations (basic handling)
 */
std::string extract_method_name(std::string_view func_name);

/**
 * @brief Logs a standardized trace message for entering a function
 *
 * Automatically uses std::source_location to determine the calling
 * function and logs a message in the form:
 *     "Entering <function_name>"
 *
 * This function is typically called via the QDK_LOG_TRACE_ENTERING() macro
 * for convenience.
 *
 * Example:
 * ```cpp
 * void compute() {
 *   qdk::chemistry::utils::log_trace_entering();  // Direct call
 *   // or preferably:
 *   QDK_LOG_TRACE_ENTERING();  // Via convenience macro
 *   // ...
 * }
 * ```
 *
 * @param location Automatically provided by std::source_location::current()
 */
void log_trace_entering(
    const std::source_location& location = std::source_location::current());

}  // namespace qdk::chemistry::utils

/**
 * @def QDK_LOGGER()
 * @brief Convenience macro to get a logger instance
 *
 * Provides a shorter syntax for obtaining logger instances.
 * Equivalent to calling qdk::chemistry::utils::Logger::get().
 * The logger is automatically named based on the caller's source location.
 *
 * @return Shared pointer to the logger instance
 *
 * Example:
 * ```cpp
 * auto logger = QDK_LOGGER();
 * logger->info("This is an info message");
 * ```
 */
#define QDK_LOGGER() qdk::chemistry::utils::Logger::get()

/**
 * @def QDK_LOG_TRACE_ENTERING()
 * @brief Convenience macro to log function entry
 *
 * Provides a shorter syntax for logging function entry.
 * Equivalent to calling qdk::chemistry::utils::log_trace_entering().
 * Automatically captures the current function name and logs an
 * "Entering <function_name>" message at trace level.
 *
 * Example:
 * ```cpp
 * void myFunction() {
 *   LOG_TRACE_ENTERING();  // Logs "Entering myFunction"
 *   // ... function implementation
 * }
 * ```
 */
#define QDK_LOG_TRACE_ENTERING() qdk::chemistry::utils::log_trace_entering()
