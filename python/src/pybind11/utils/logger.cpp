// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <spdlog/cfg/env.h>

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

  // Use the LogLevel enum directly
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
        } else if (level == "off") {
          qdk::chemistry::utils::Logger::set_global_level(
              qdk::chemistry::utils::LogLevel::off);
        } else {
          throw std::invalid_argument("Invalid log level: " + level);
        }
      },
      "Set global logging level");

  // Also bind the direct enum version
  logger_module.def("set_global_level",
                    &qdk::chemistry::utils::Logger::set_global_level,
                    "Set global logging level with LogLevel enum");

  logger_module.def(
      "load_env_levels", []() { spdlog::cfg::load_env_levels(); },
      "Load logging levels from environment variables");

  logger_module.def("disable_all", &qdk::chemistry::utils::Logger::disable_all,
                    "Disable all logging");
}
