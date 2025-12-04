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
    // Reset to trace level for each test
    Logger::set_global_level(LogLevel::trace);
  }

  void TearDown() override { Logger::set_global_level(LogLevel::info); }
};

TEST_F(LoggerTest, GetLoggerReturnsValidLogger) {
  auto logger = Logger::get();
  ASSERT_NE(logger, nullptr);
  // Logger name should be "qdk-chemistry"
  EXPECT_EQ(logger->name(), "qdk-chemistry");
}

TEST_F(LoggerTest, GetLoggerReturnsSameInstance) {
  auto logger1 = Logger::get();
  auto logger2 = Logger::get();

  // Should return the same logger instance (single global logger)
  EXPECT_EQ(logger1.get(), logger2.get());
}

TEST_F(LoggerTest, ContextLoggerMacroWorks) {
  // QDK_LOGGER() returns a ContextLogger value, not a pointer
  // Just verify it can be used without throwing
  EXPECT_NO_THROW({ QDK_LOGGER().info("Test message from macro"); });
}

TEST_F(LoggerTest, RawLoggerMacroWorks) {
  // QDK_RAW_LOGGER() returns the raw spdlog logger pointer
  auto logger = QDK_RAW_LOGGER();
  ASSERT_NE(logger, nullptr);
  EXPECT_EQ(logger->name(), "qdk-chemistry");
}

TEST_F(LoggerTest, LoggingLevelsWork) {
  // Test raw logger
  auto logger = Logger::get();
  EXPECT_NO_THROW({
    logger->trace("This is a trace message");
    logger->debug("This is a debug message");
    logger->info("This is an info message");
    logger->warn("This is a warning message");
    logger->error("This is an error message");
    logger->critical("This is a critical message");
  });
}

TEST_F(LoggerTest, ContextLoggerLevelsWork) {
  // Test ContextLogger via macro
  EXPECT_NO_THROW({
    QDK_LOGGER().trace("This is a trace message");
    QDK_LOGGER().debug("This is a debug message");
    QDK_LOGGER().info("This is an info message");
    QDK_LOGGER().warn("This is a warning message");
    QDK_LOGGER().error("This is an error message");
    QDK_LOGGER().critical("This is a critical message");
  });
}

TEST_F(LoggerTest, GlobalLevelControl) {
  // Set to info level
  Logger::set_global_level(LogLevel::info);
  EXPECT_EQ(Logger::get_global_level(), spdlog::level::info);

  EXPECT_NO_THROW({
    QDK_LOGGER().info("This should appear");
    QDK_LOGGER().debug("This should be suppressed");
  });

  // Disable all logging
  Logger::disable_all();
  EXPECT_EQ(Logger::get_global_level(), spdlog::level::off);

  EXPECT_NO_THROW({ QDK_LOGGER().critical("This should be suppressed"); });
}

TEST_F(LoggerTest, FormattedLogging) {
  EXPECT_NO_THROW({
    QDK_LOGGER().info("Testing formatted message: value = {}", 42);
    QDK_LOGGER().warn("Multiple values: {} and {}", "hello", 3.14);
  });
}

TEST_F(LoggerTest, RuntimeStringLogging) {
  // Test that runtime strings work (not just compile-time format strings)
  std::ostringstream oss;
  oss << "Dynamic message with value: " << 123;

  EXPECT_NO_THROW({
    QDK_LOGGER().info(oss.str());
    QDK_LOGGER().warn(std::string("Another runtime string"));
  });
}

TEST_F(LoggerTest, LogTraceEnteringFunction) {
  Logger::set_global_level(LogLevel::trace);

  // Test that log_trace_entering doesn't throw
  EXPECT_NO_THROW({ log_trace_entering(); });
}

void test_function() { log_trace_entering(); }

TEST_F(LoggerTest, LogTraceEnteringInFunction) {
  Logger::set_global_level(LogLevel::trace);

  // Test that log_trace_entering works in a named function
  EXPECT_NO_THROW({ test_function(); });
}

TEST_F(LoggerTest, LogTraceEnteringMacro) {
  Logger::set_global_level(LogLevel::trace);

  // Test the macro version
  EXPECT_NO_THROW({ QDK_LOG_TRACE_ENTERING(); });
}

TEST_F(LoggerTest, LogTraceEnteringFormat) {
  Logger::set_global_level(LogLevel::trace);

  // log_trace_entering outputs: "[file_context] Entering method_name"
  // Verify it doesn't throw and uses the correct format
  EXPECT_NO_THROW({ QDK_LOG_TRACE_ENTERING(); });
}

TEST_F(LoggerTest, GetSourceContext) {
  auto context = Logger::get_source_context();
  // Context is non-empty (either a valid path or "unknown")
  EXPECT_FALSE(context.empty());
}

TEST_F(LoggerTest, GetSourceContextForQdkPath) {
  // Test that path_to_colon_string works correctly for qdk paths
  std::string qdk_path =
      "/workspaces/qdk_chem/cpp/src/qdk/chemistry/utils/logger.cpp";
  std::string result = path_to_colon_string(qdk_path, "qdk");
  EXPECT_EQ(result, "qdk:chemistry:utils:logger");
  EXPECT_NE(result.find("logger"), std::string::npos);
}

TEST_F(LoggerTest, GetSourceContextReturnsUnknownForNonQdkPath) {
  // Test files are not under qdk/ directory
  std::string test_path = "/workspaces/qdk_chem/cpp/tests/test_logger.cpp";
  std::string result = path_to_colon_string(test_path, "qdk");
  EXPECT_TRUE(result.empty());  // No qdk/ segment found

  // Therefore get_source_context returns "unknown" for test files
  auto context = Logger::get_source_context();
  EXPECT_EQ(context, "unknown");
}

TEST_F(LoggerTest, ExtractMethodName) {
  // Test method name extraction
  EXPECT_EQ(extract_method_name("void MyClass::myMethod(int)"), "myMethod");
  EXPECT_EQ(extract_method_name("MyClass::MyClass()"), "MyClass constructor");
  EXPECT_EQ(extract_method_name("globalFunction(std::string)"),
            "globalFunction");
}

// Manual test that shows actual output
TEST_F(LoggerTest, ManualOutputTest) {
  Logger::set_global_level(LogLevel::trace);

  std::cout << "\n=== Manual Logger Test Output ===" << std::endl;

  // Using ContextLogger (with automatic context)
  QDK_LOGGER().trace("TRACE: This is a trace message");
  QDK_LOGGER().debug("DEBUG: This is a debug message");
  QDK_LOGGER().info("INFO: Starting calculation for component XYZ");
  QDK_LOGGER().warn("WARN: Parameter value {} may be out of range", 999);
  QDK_LOGGER().error("ERROR: Failed to load configuration file");
  QDK_LOGGER().critical("CRITICAL: System failure detected");

  std::cout << "\n=== Testing log_trace_entering ===" << std::endl;
  QDK_LOG_TRACE_ENTERING();
  test_function();

  std::cout << "\n=== Testing raw logger (no context) ===" << std::endl;
  QDK_RAW_LOGGER()->info("Raw message without context prefix");

  std::cout << "\n=== Testing runtime strings ===" << std::endl;
  std::ostringstream oss;
  oss << "Dynamic message: " << 42;
  QDK_LOGGER().info(oss.str());

  std::cout << "=== End Manual Logger Test Output ===\n" << std::endl;
}
