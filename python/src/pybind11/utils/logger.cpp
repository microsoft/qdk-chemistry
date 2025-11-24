// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <qdk/chemistry/utils/logger.hpp>

namespace py = pybind11;

void bind_logger(py::module& m) {
  auto logger_module = m.def_submodule("Logger", "Logging utilities");

  // Bind the LogLevel enum
  py::enum_<qdk::chemistry::utils::LogLevel>(logger_module, "LogLevel")
      .value("trace", qdk::chemistry::utils::LogLevel::trace)
      .value("debug", qdk::chemistry::utils::LogLevel::debug)
      .value("info", qdk::chemistry::utils::LogLevel::info)
      .value("warn", qdk::chemistry::utils::LogLevel::warn)
      .value("error", qdk::chemistry::utils::LogLevel::error)
      .value("critical", qdk::chemistry::utils::LogLevel::critical)
      .value("off", qdk::chemistry::utils::LogLevel::off);

  // Single set_global_level binding - this affects ALL loggers (C++ and Python)
  logger_module.def("set_global_level",
                    &qdk::chemistry::utils::Logger::set_global_level,
                    "Set global logging level for all loggers (accepts "
                    "LogLevel enum or string)");

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
      "Set global logging level with string");

  // Bind get_global_level for consistency
  logger_module.def(
      "get_global_level",
      []() -> std::string {
        auto level = qdk::chemistry::utils::Logger::get_global_level();
        switch (level) {
          case spdlog::level::trace:
            return "trace";
          case spdlog::level::debug:
            return "debug";
          case spdlog::level::info:
            return "info";
          case spdlog::level::warn:
            return "warn";
          case spdlog::level::err:
            return "error";
          case spdlog::level::critical:
            return "critical";
          case spdlog::level::off:
            return "off";
          default:
            return "unknown";
        }
      },
      "Get the current global logging level as string");

  // Bind utility functions
  logger_module.def(
      "load_env_levels", []() { spdlog::cfg::load_env_levels(); },
      "Load logging levels from environment variables");

  logger_module.def("disable_all", &qdk::chemistry::utils::Logger::disable_all,
                    "Disable all logging");

  // Bind function to create logger by name (Python-specific)
  // This uses the same mutex-protected global level as C++ Logger::get()
  logger_module.def(
      "QDK_LOGGER",
      [](const std::string& name) {
        auto logger = spdlog::get(name);
        if (!logger) {
          try {
            logger = spdlog::stdout_color_mt(name);
            logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [%n] [%^%l%$] %v");
          } catch (const spdlog::spdlog_ex&) {
            logger = spdlog::get(name);
          }
        }

        // IMPORTANT: Use the same mutex-protected global level as C++
        // Logger::get()
        if (logger) {
          // Get the current global level using the same mutex protection as C++
          auto global_level = qdk::chemistry::utils::Logger::get_global_level();
          logger->set_level(global_level);
        }

        return logger;
      },
      "Get or create a named logger instance (inherits mutex-protected global "
      "log level)");

  // Bind spdlog logger class with essential methods
  py::class_<spdlog::logger, std::shared_ptr<spdlog::logger>>(logger_module,
                                                              "SpdLogger")
      .def("trace", [](spdlog::logger& logger,
                       const std::string& msg) { logger.trace(msg); })
      .def("debug", [](spdlog::logger& logger,
                       const std::string& msg) { logger.debug(msg); })
      .def("info", [](spdlog::logger& logger,
                      const std::string& msg) { logger.info(msg); })
      .def("warn", [](spdlog::logger& logger,
                      const std::string& msg) { logger.warn(msg); })
      .def("error", [](spdlog::logger& logger,
                       const std::string& msg) { logger.error(msg); })
      .def("critical", [](spdlog::logger& logger,
                          const std::string& msg) { logger.critical(msg); })
      .def("name", &spdlog::logger::name)
      .def("set_level", [](spdlog::logger& logger, const std::string& level) {
        // This is for setting individual logger level (overrides global)
        if (level == "trace")
          logger.set_level(spdlog::level::trace);
        else if (level == "debug")
          logger.set_level(spdlog::level::debug);
        else if (level == "info")
          logger.set_level(spdlog::level::info);
        else if (level == "warn")
          logger.set_level(spdlog::level::warn);
        else if (level == "error")
          logger.set_level(spdlog::level::err);
        else if (level == "critical")
          logger.set_level(spdlog::level::critical);
        else if (level == "off")
          logger.set_level(spdlog::level::off);
        else
          throw std::invalid_argument("Invalid log level: " + level);
      });
}
