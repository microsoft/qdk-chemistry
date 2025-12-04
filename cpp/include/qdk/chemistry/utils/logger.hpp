// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <source_location>
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
 * and conventions. It uses a single global logger instance for efficiency,
 * while still providing per-file/function context in log messages.
 *
 * Features
 * --------
 * - Single global logger instance (efficient, no per-file logger creation)
 * - Automatic source context (file path and method name) in log messages
 * - Global log level control
 * - Consistent formatting and output configuration
 * - Thread-safe operation (inherited from spdlog)
 * - Convenience macros for easy logging with automatic context
 *
 * Usage Example
 * -------------
 * ```cpp
 * #include <qdk/chemistry/utils/logger.hpp>
 *
 * // Use the QDK_LOGGER() macro - context is automatic
 * QDK_LOGGER().info("Starting calculation with {} atoms", num_atoms);
 * QDK_LOGGER().warn("Parameter {} may be out of range: {}", param_name,
 * value); QDK_LOGGER().debug("Debug information: {}", data);
 *
 * // Log function entry with convenience macro
 * void myFunction() {
 *   QDK_LOG_TRACE_ENTERING();  // Automatically logs "Entering myFunction"
 *   // ... function implementation
 * }
 *
 * // Control global logging
 * Logger::set_global_level(LogLevel::debug);  // Enable debug output
 * Logger::disable_all();                      // Disable all logging
 * ```
 *
 * Output Format
 * -------------
 * ```

 * [2025-12-03 10:30:00.123456] [info] [qdk:chemistry:scf:run] Message
 *
 * Thread Safety
 * -------------
 * This class is thread-safe as it wraps spdlog, which provides thread-safe
 * logging operations. The single global logger instance is initialized once
 * using std::call_once for thread-safe initialization.
 */
class Logger {
 public:
  /**
   * @brief Get the global logger instance
   *
   * Returns the single global logger instance for QDK Chemistry.
   * The logger is lazily initialized on first call and reused thereafter.
   *
   * New logger is created with default settings:
   * - Colored console output
   * - Inherit global log level
   * - Consistent timestamped format
   *
   * @return Shared pointer to the global logger instance
   */
  static std::shared_ptr<spdlog::logger> get();

  /**
   * @brief Set the global log level for all loggers
   *
   * Changes the logging level for the global logger instance.
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

  /**
   * @brief Get the current global log level
   *
   * Returns the current global logging level. This uses mutex protection
   * to ensure thread safety.
   *
   * @return The current global log level
   */
  static spdlog::level::level_enum get_global_level();

  /**
   * @brief Get a formatted source context string for the given location
   *
   * Returns a string like "qdk:chemistry:utils:logger:method_name" that
   * identifies the source file and method. This is automatically used by
   * the ContextLogger to prefix log messages with per-file context.
   *
   * @param location Source location (defaults to caller's location)
   * @return Formatted context string
   */
  static std::string get_source_context(
      const std::source_location& location = std::source_location::current());
};

/**
 * @brief Convert a filesystem path to colon-separated string starting from a
 * given segment
 *
 * @param file_path Full path to the file
 * @param start_segment The segment from which to start (default "qdk")
 * @return Colon-separated string
 */
std::string path_to_colon_string(const std::string& file_path,
                                 const std::string& start_segment = "qdk");

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
 *     "[context] Entering method_name"
 *
 * This function is typically called via the QDK_LOG_TRACE_ENTERING() macro
 * for convenience.
 *
 * Example:
 * ```cpp
 * void compute() {
 *   QDK_LOG_TRACE_ENTERING();  // Logs "[qdk:chemistry:...:compute] Entering"
 *   // ...
 * }
 * ```
 *
 * @param location Automatically provided by std::source_location::current()
 */
void log_trace_entering(
    const std::source_location& location = std::source_location::current());

/**
 * @class ContextLogger
 * @brief A logger wrapper that automatically prepends source context to
 * messages
 *
 * This class wraps the global logger and automatically includes file/method
 * context in every log message. It provides the same interface as
 * spdlog::logger (trace, debug, info, warn, error, critical) but prepends
 * source location info.
 *
 * Use via the QDK_LOGGER() macro which captures the call site's source
 * location.
 *
 * Example:
 * ```cpp
 * QDK_LOGGER().info("Message with {} args", 2);
 * // Output: [timestamp] [info] [file:method] Message with 2 args
 * ```
 */
class ContextLogger {
 public:
  explicit ContextLogger(const std::source_location& loc)
      : context_(Logger::get_source_context(loc)) {}

  template <typename... Args>
  void trace(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get()->trace("[{}] {}", context_,
                         fmt::format(fmt, std::forward<Args>(args)...));
  }

  void trace(const std::string& msg) {
    Logger::get()->trace("[{}] {}", context_, msg);
  }

  template <typename... Args>
  void debug(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get()->debug("[{}] {}", context_,
                         fmt::format(fmt, std::forward<Args>(args)...));
  }

  void debug(const std::string& msg) {
    Logger::get()->debug("[{}] {}", context_, msg);
  }

  template <typename... Args>
  void info(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get()->info("[{}] {}", context_,
                        fmt::format(fmt, std::forward<Args>(args)...));
  }

  void info(const std::string& msg) {
    Logger::get()->info("[{}] {}", context_, msg);
  }

  template <typename... Args>
  void warn(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get()->warn("[{}] {}", context_,
                        fmt::format(fmt, std::forward<Args>(args)...));
  }

  void warn(const std::string& msg) {
    Logger::get()->warn("[{}] {}", context_, msg);
  }

  template <typename... Args>
  void error(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get()->error("[{}] {}", context_,
                         fmt::format(fmt, std::forward<Args>(args)...));
  }

  void error(const std::string& msg) {
    Logger::get()->error("[{}] {}", context_, msg);
  }

  template <typename... Args>
  void critical(fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get()->critical("[{}] {}", context_,
                            fmt::format(fmt, std::forward<Args>(args)...));
  }

  void critical(const std::string& msg) {
    Logger::get()->critical("[{}] {}", context_, msg);
  }

 private:
  std::string context_;
};

}  // namespace qdk::chemistry::utils

// =============================================================================
// Convenience macros
// =============================================================================

/**
 * @def QDK_LOGGER()
 * @brief Get a context-aware logger instance
 *
 * Returns a ContextLogger that automatically prepends source file and method
 * information to all log messages. Use with . to call logging methods.
 *
 * Example:
 * ```cpp
 * QDK_LOGGER().info("Starting calculation");
 * QDK_LOGGER().debug("Value is {}", value);
 * QDK_LOGGER().trace("Entering loop iteration {}", i);
 * ```
 *
 * Output:
 * ```

 * [2025-12-03 10:30:00.123456] [info] [qdk:chemistry:scf:run] Starting
 calculation
 *
 */
#define QDK_LOGGER() \
  qdk::chemistry::utils::ContextLogger(std::source_location::current())

/**
 * @def QDK_RAW_LOGGER()
 * @brief Get the raw spdlog logger instance (without automatic context)
 *
 * Use this when you need direct access to the underlying spdlog logger,
 * for example when you want to log without the source context prefix.
 *
 * @return Shared pointer to the global logger instance
 *
 * Example:
 * ```cpp
 * QDK_RAW_LOGGER()->info("Raw message without context prefix");
 * ```
 */
#define QDK_RAW_LOGGER() qdk::chemistry::utils::Logger::get()

/**
 * @def QDK_LOG_TRACE_ENTERING()
 * @brief Log function entry at trace level
 *
 * Logs an "Entering" message with automatic source context.
 * Useful for tracing function call flow during debugging.
 *
 * Example:
 * ```cpp
 * void myFunction() {
 *   QDK_LOG_TRACE_ENTERING();  // Logs "[qdk:...:myFunction] Entering"
 *   // ... function implementation
 * }
 * ```
 */
#define QDK_LOG_TRACE_ENTERING() qdk::chemistry::utils::log_trace_entering()
