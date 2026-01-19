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

/**
 * @def DATACLASS_TO_SNAKE_CASE
 * @brief Macro to generate snake_case data type name from a class name
 *
 * This macro converts a PascalCase class name to a snake_case string.
 * The conversion happens once per call site due to the static local variable.
 *
 * @param ClassName The class name in PascalCase (e.g., ConfigurationSet)
 * @return A const char* containing the snake_case version, (e.g.
 * configuration_set)
 *
 * Usage:
 * @code
 * class ConfigurationSet : public DataClass {
 *   std::string get_data_type_name() const override {
 *     return DATACLASS_TO_SNAKE_CASE(ConfigurationSet);
 *   }
 *   // Returns "configuration_set"
 * };
 * @endcode
 */
#define DATACLASS_TO_SNAKE_CASE(ClassName)                \
  ([]() -> const char* {                                  \
    static const std::string result =                     \
        qdk::chemistry::utils::to_snake_case(#ClassName); \
    return result.c_str();                                \
  }())

}  // namespace qdk::chemistry::utils
