// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace qdk::chemistry::utils {

/**
 * @brief Convert a PascalCase/camelCase string to snake_case at runtime
 *
 * This function inserts an underscore before each uppercase letter
 * (except at position 0) and converts all letters to lowercase.
 *
 * @param input Input string in PascalCase or camelCase
 * @return std::string containing the snake_case version
 *
 * Examples:
 * - "Ansatz" -> "ansatz"
 * - "ConfigurationSet" -> "configuration_set"
 * - "StabilityResult" -> "stability_result"
 */
inline std::string to_snake_case(const char* input) {
  std::string result;
  for (std::size_t i = 0; input[i] != '\0'; ++i) {
    char c = input[i];
    if (c >= 'A' && c <= 'Z') {
      if (i > 0) {
        result += '_';
      }
      result += static_cast<char>(c + 32);
    } else {
      result += c;
    }
  }
  return result;
}

}  // namespace qdk::chemistry::utils
