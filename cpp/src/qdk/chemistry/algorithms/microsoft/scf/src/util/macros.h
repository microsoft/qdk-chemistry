// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <spdlog/spdlog.h>

#include <exception>

#include "util/timer.h"

/**
 * @brief Macro to time a code block execution
 * @param call Code to execute
 * @param name Name for the timer
 */
#define TIMEIT(call, name) \
  {                        \
    AutoTimer timer(name); \
    call;                  \
  }

/**
 * @brief Verify an expression and abort if false
 * @param expr Expression to verify
 */
#define VERIFY(expr)                                                      \
  if (!static_cast<bool>(expr)) {                                         \
    spdlog::critical("{}:{}: Verifying '{}' failed.", __FILE__, __LINE__, \
                     #expr);                                              \
    std::abort();                                                         \
  }

/**
 * @brief Verify input and raise InputError exception if false
 * @param expr Expression to verify
 * @param msg Error message
 */
#define VERIFY_INPUT(expr, msg) \
  if (!static_cast<bool>(expr)) \
    throw std::invalid_argument(std::string("InputError: " + std::string(msg)));
