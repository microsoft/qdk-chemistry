// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/utils/logger.hpp>

TEST(LoggerStartupTest, StartupLoggerUsesItsConfiguredFlushPolicy) {
  auto logger = qdk::chemistry::utils::Logger::get();

  ASSERT_NE(logger, nullptr);
  EXPECT_EQ(logger->flush_level(), logger->level());
}
