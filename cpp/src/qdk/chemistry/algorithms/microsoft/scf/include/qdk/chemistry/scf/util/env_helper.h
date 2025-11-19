// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace qdk::chemistry::scf::env {

/**
 * @brief Get environment variable value with type conversion
 * @tparam T Type to convert the environment variable to
 * @param key Environment variable name
 * @param default_value Default value if variable not set
 * @return Environment variable value or default
 */
template <class T>
T get(const std::string& key, const T& default_value = {}) {
  char* var = std::getenv(key.c_str());
  if (var == nullptr) {
    return default_value;
  } else {
    T value;
    std::istringstream ss(var);
    ss >> value;
    return value;
  }
}

}  // namespace qdk::chemistry::scf::env
