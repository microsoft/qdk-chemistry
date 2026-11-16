// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>

#include <algorithm>
#include <memory>
#include <qdk/chemistry/algorithms/jk_builder.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace qdk::chemistry;

// The plugins of the test add-on (CMakeLists.txt) run on the first use of an
// algorithm factory, which must be the first in the process.
TEST(PluginTest, PluginsLoadAfterBuiltInImplementations) {
  std::ostringstream messages;
  auto logger = utils::Logger::get();
  struct LoggerGuard {
    std::shared_ptr<spdlog::logger> logger;
    std::vector<spdlog::sink_ptr> sinks;
    utils::LogLevel level;
    ~LoggerGuard() {
      logger->sinks() = std::move(sinks);
      utils::Logger::set_global_level(level);
    }
  } guard{logger, logger->sinks(), utils::Logger::get_global_level()};
  logger->sinks().push_back(
      std::make_shared<spdlog::sinks::ostream_sink_mt>(messages));
  utils::Logger::set_global_level(utils::LogLevel::warn);

  // Creating the SCF solver initializes the J/K builder registry first.
  EXPECT_NE(algorithms::ScfSolverFactory::create(), nullptr);
  const auto available = algorithms::JKBuilderFactory::available();
  EXPECT_NE(std::find(available.begin(), available.end(), "test_addon"),
            available.end());

  const std::string log = messages.str();
  EXPECT_NE(log.find("Failed to load QDK/Chemistry plugin "
                     "'qdk_test_addon::register_duplicate': "),
            std::string::npos);
  EXPECT_NE(log.find("already exists in registry"), std::string::npos);
  EXPECT_NE(log.find("Failed to load QDK/Chemistry plugin "
                     "'qdk_test_addon::throw_non_standard'"),
            std::string::npos);
  EXPECT_EQ(log.find("'qdk_test_addon::register_plugin'"), std::string::npos);
}
