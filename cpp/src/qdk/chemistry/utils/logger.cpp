// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <spdlog/sinks/stdout_color_sinks.h>

#include <mutex>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace qdk::chemistry::utils {

// Track our own global level to avoid spdlog::get_level() issues
// Map compile-time level to spdlog level
static constexpr spdlog::level::level_enum default_level_from_config() {
  switch (QDK_CHEMISTRY_LOG_LEVEL) {
    case 0:
      return spdlog::level::trace;
    case 1:
      return spdlog::level::debug;
    case 2:
      return spdlog::level::info;
    case 3:
      return spdlog::level::warn;
    case 4:
      return spdlog::level::err;
    case 5:
      return spdlog::level::critical;
    case 6:
      return spdlog::level::off;
    default:
      return spdlog::level::info;
  }
}

static spdlog::level::level_enum g_global_level = default_level_from_config();
static std::mutex g_level_mutex;

// Single global logger instance
static std::shared_ptr<spdlog::logger> g_logger;
static std::once_flag g_logger_init_flag;

// Internal helper - convert path to colon-separated context string
static std::string path_to_colon_string(
    const std::string& file_path, const std::string& start_segment = "qdk") {
  // Look for the segment as a directory (with / before and after)
  // This prevents matching "qdk_chem" when looking for "qdk"
  std::string segment_pattern = "/" + start_segment + "/";
  size_t pos = file_path.find(segment_pattern);

  if (pos == std::string::npos) {
    // Also try at the start of the path (no leading /)
    if (file_path.rfind(start_segment + "/", 0) == 0) {
      pos = 0;
    } else {
      return "";  // start segment not found
    }
  } else {
    // Skip the leading /
    pos += 1;
  }

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

// Internal helper - extract clean method name from function signature
static std::string extract_method_name(std::string_view func_name) {
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

static LogLevel from_spdlog_level(spdlog::level::level_enum level) {
  switch (level) {
    case spdlog::level::trace:
      return LogLevel::trace;
    case spdlog::level::debug:
      return LogLevel::debug;
    case spdlog::level::info:
      return LogLevel::info;
    case spdlog::level::warn:
      return LogLevel::warn;
    case spdlog::level::err:
      return LogLevel::error;
    case spdlog::level::critical:
      return LogLevel::critical;
    case spdlog::level::off:
      return LogLevel::off;
    default:
      return LogLevel::info;
  }
}

static void init_global_logger() {
  try {
    g_logger = spdlog::stdout_color_mt("qdk-chemistry");
  } catch (const spdlog::spdlog_ex&) {
    g_logger = spdlog::get("qdk-chemistry");
  }

  if (g_logger) {
    std::lock_guard<std::mutex> lock(g_level_mutex);
    g_logger->set_level(g_global_level);
    // Pattern: [timestamp] [colored_level] message
    // The file context and method are added by ContextLogger in the message
    g_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [%^%l%$] %v");
  }
}

std::shared_ptr<spdlog::logger> Logger::get() {
  // Initialize the single global logger once
  std::call_once(g_logger_init_flag, init_global_logger);

  // Update level if it changed (thread-safe check)
  if (g_logger) {
    std::lock_guard<std::mutex> lock(g_level_mutex);
    if (g_logger->level() != g_global_level) {
      g_logger->set_level(g_global_level);
    }
  }

  return g_logger;
}

std::string Logger::get_source_context(const std::source_location& location) {
  std::string file_id = path_to_colon_string(location.file_name(), "qdk");

  if (file_id.empty()) {
    return "unknown";
  }
  return file_id;
}

void Logger::set_global_level(LogLevel level) {
  auto spdlog_level = to_spdlog_level(level);

  // Update both our tracked level and spdlog's global level
  {
    std::lock_guard<std::mutex> lock(g_level_mutex);
    g_global_level = spdlog_level;
    if (g_logger) {
      g_logger->set_level(spdlog_level);
    }
  }

  spdlog::set_level(spdlog_level);
}

LogLevel Logger::get_global_level() {
  std::lock_guard<std::mutex> lock(g_level_mutex);
  return from_spdlog_level(g_global_level);
}

void log_trace_entering(const std::source_location& location) {
  auto logger = Logger::get();
  std::string file_ctx = path_to_colon_string(location.file_name(), "qdk");
  std::string method = extract_method_name(location.function_name());

  if (file_ctx.empty()) {
    file_ctx = "unknown";
  }
  if (method.empty()) {
    method = "unknown";
  }

  logger->trace("[{}] Entering {}", file_ctx, method);
}

}  // namespace qdk::chemistry::utils
