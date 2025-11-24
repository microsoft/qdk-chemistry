// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <qdk/chemistry/utils/logger.hpp>

using namespace qdk::chemistry::utils;

class LoggerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset spdlog state for each test
    spdlog::drop_all();
    spdlog::set_level(spdlog::level::trace);
  }

  void TearDown() override {
    // Clean up after each test
    spdlog::drop_all();
  }
};

TEST_F(LoggerTest, GetLoggerWithSourceLocation) {
  auto logger = Logger::get();
  ASSERT_NE(logger, nullptr);
  // Logger name should be based on source location, not empty
  EXPECT_FALSE(logger->name().empty());
}

TEST_F(LoggerTest, GetLoggerReturnsConsistentLogger) {
  auto logger1 = Logger::get();
  auto logger2 = Logger::get();

  // Should return the same logger instance when called from same location
  EXPECT_EQ(logger1, logger2);
}

TEST_F(LoggerTest, MacroWorks) {
  auto logger = QDK_LOGGER();
  ASSERT_NE(logger, nullptr);
  // Name should be based on source location
  EXPECT_FALSE(logger->name().empty());
}

TEST_F(LoggerTest, LoggingLevelsWork) {
  auto logger = Logger::get();

  // These should not throw and should work without error
  EXPECT_NO_THROW({
    logger->trace("This is a trace message");
    logger->debug("This is a debug message");
    logger->info("This is an info message");
    logger->warn("This is a warning message");
    logger->error("This is an error message");
    logger->critical("This is a critical message");
  });
}

TEST_F(LoggerTest, GlobalLevelControl) {
  auto logger = Logger::get();

  // Set to info level
  Logger::set_global_level(LogLevel::info);
  EXPECT_NO_THROW({
    logger->info("This should appear");
    logger->debug("This should be suppressed");
  });

  // Disable all logging
  Logger::disable_all();
  EXPECT_NO_THROW({ logger->critical("This should be suppressed"); });
}

TEST_F(LoggerTest, FormattedLogging) {
  auto logger = Logger::get();

  EXPECT_NO_THROW({
    logger->info("Testing formatted message: value = {}", 42);
    logger->warn("Multiple values: {} and {}", "hello", 3.14);
  });
}

TEST_F(LoggerTest, LogTraceEnteringFunction) {
  // Enable all levels for this test
  Logger::set_global_level(LogLevel::trace);

  // Test that log_trace_entering doesn't throw
  EXPECT_NO_THROW({ log_trace_entering(); });
}

void test_function() { log_trace_entering(); }

TEST_F(LoggerTest, LogTraceEnteringInFunction) {
  // Enable all levels for this test
  Logger::set_global_level(LogLevel::trace);

  // Test that log_trace_entering works in a named function
  EXPECT_NO_THROW({ test_function(); });
}

// Simple manual test that actually shows output
TEST_F(LoggerTest, ManualOutputTest) {
  // Enable all levels for this test
  Logger::set_global_level(LogLevel::trace);

  auto logger = Logger::get();

  std::cout << "\n=== Manual Logger Test Output ===" << std::endl;
  logger->trace("TRACE: This is a trace message");
  logger->debug("DEBUG: This is a debug message");
  logger->info("INFO: Starting calculation for component XYZ");
  logger->warn("WARN: Parameter value {} may be out of range", 999);
  logger->error("ERROR: Failed to load configuration file");
  logger->critical("CRITICAL: System failure detected");

  std::cout << "\n=== Testing log_trace_entering ===" << std::endl;
  log_trace_entering();
  test_function();

  std::cout << "=== End Manual Logger Test Output ===\n" << std::endl;
}
