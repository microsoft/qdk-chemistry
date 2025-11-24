// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <spdlog/sinks/stdout_color_sinks.h>

#include <mutex>
#include <qdk/chemistry/utils/logger.hpp>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace qdk::chemistry::utils {

// Track our own global level to avoid spdlog::get_level() issues
static spdlog::level::level_enum g_global_level = spdlog::level::info;
static std::mutex g_level_mutex;

inline std::string path_to_colon_string(const std::string& file_path,
                                        const std::string& start_segment) {
  size_t pos = file_path.find(start_segment);
  if (pos == std::string::npos) return "";  // start segment not found

  std::string relevant_path = file_path.substr(pos);

  std::vector<std::string> parts;
  std::string temp;
  std::istringstream stream(relevant_path);

  // Split by '/'
  while (std::getline(stream, temp, '/')) {
    if (!temp.empty()) {
      parts.push_back(temp);
    }
  }

  // Remove file extension from last part
  if (!parts.empty()) {
    std::string& last = parts.back();
    size_t dot_pos = last.find_last_of('.');
    if (dot_pos != std::string::npos) {
      last = last.substr(0, dot_pos);
    }
  }

  // Join with ':'
  std::ostringstream result;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) result << ":";
    result << parts[i];
  }

  return result.str();
}

inline std::string extract_method_name(std::string_view func_name) {
  std::string full_name = std::string(func_name);
  // Handle lambdas: strip everything starting with '::<lambda'
  size_t lambda_pos = full_name.find("::<lambda");
  if (lambda_pos != std::string::npos) {
    full_name = full_name.substr(0, lambda_pos);
  }

  // Strip argument list
  size_t paren_pos = full_name.rfind('(');
  if (paren_pos != std::string::npos) {
    full_name = full_name.substr(0, paren_pos);
  }

  // Find last '::' to get function name
  size_t last_colons = full_name.rfind("::");
  std::string name = (last_colons != std::string::npos)
                         ? full_name.substr(last_colons + 2)
                         : full_name;

  // Detect constructor: check if the function name matches the preceding class
  // name
  if (last_colons != std::string::npos) {
    // Find class name (the component before last '::')
    size_t second_last_colons = full_name.rfind("::", last_colons - 1);
    std::string class_name =
        (second_last_colons != std::string::npos)
            ? full_name.substr(second_last_colons + 2,
                               last_colons - second_last_colons - 2)
            : full_name.substr(0, last_colons);

    if (class_name == name) {
      name += " constructor";
    }
  }

  return name;
}

static spdlog::level::level_enum to_spdlog_level(LogLevel level) {
  switch (level) {
    case LogLevel::trace:
      return spdlog::level::trace;
    case LogLevel::debug:
      return spdlog::level::debug;
    case LogLevel::info:
      return spdlog::level::info;
    case LogLevel::warn:
      return spdlog::level::warn;
    case LogLevel::error:
      return spdlog::level::err;
    case LogLevel::critical:
      return spdlog::level::critical;
    case LogLevel::off:
      return spdlog::level::off;
  }
  return spdlog::level::info;  // fallback
}

std::shared_ptr<spdlog::logger> Logger::get(
    const std::source_location& location) {
  // Use the file path (or fallback) from source_location as logger name
  std::string logger_name = path_to_colon_string(location.file_name());
  if (logger_name.empty()) {
    logger_name = "unknown_function";
  }

  // Try to get or create logger by function name
  auto logger = spdlog::get(logger_name);
  if (!logger) {
    try {
      logger = spdlog::stdout_color_mt(logger_name);
    } catch (const spdlog::spdlog_ex&) {
      // Handle race condition between threads
      logger = spdlog::get(logger_name);
    }
  }

  // Always apply QDK Chemistry default logging configuration
  // Use our tracked global level instead of spdlog::get_level()
  {
    std::lock_guard<std::mutex> lock(g_level_mutex);
    logger->set_level(g_global_level);
  }

  // Output format:
  // [2025-11-04 17:54:10.939593] [source_location.method_name] [trace] Entering
  // ...
  logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [%n] [%^%l%$] %v");

  return logger;
}

void Logger::set_global_level(LogLevel level) {
  auto spdlog_level = to_spdlog_level(level);

  // Update both our tracked level and spdlog's global level
  {
    std::lock_guard<std::mutex> lock(g_level_mutex);
    g_global_level = spdlog_level;
  }

  spdlog::set_level(spdlog_level);
}

void Logger::disable_all() {
  {
    std::lock_guard<std::mutex> lock(g_level_mutex);
    g_global_level = spdlog::level::off;
  }

  spdlog::set_level(spdlog::level::off);
}

spdlog::level::level_enum Logger::get_global_level() {
  std::lock_guard<std::mutex> lock(g_level_mutex);
  return g_global_level;
}

void log_trace_entering(const std::source_location& location) {
  auto logger = Logger::get(location);
  logger->trace("Entering {}", extract_method_name(location.function_name()));
}

}  // namespace qdk::chemistry::utils
