// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <stdexcept>
#include <string>

namespace qdk::chemistry::data {

/**
 * @brief Utility for handling data structure type in filenames
 */
class DataTypeFilename {
 public:
  /**
   * @brief Validate filename has the correct data type suffix for writing
   * @param filename Filename to validate (e.g., "example.structure.json")
   * @param data_type Expected data structure type (e.g., "structure")
   * @return The original filename if valid
   * @throws std::invalid_argument if filename doesn't have correct data type
   * suffix
   */
  static std::string validate_write_suffix(const std::string &filename,
                                           const std::string &data_type) {
    // Find the last dot (extension)
    size_t last_dot = filename.find_last_of('.');
    if (last_dot == std::string::npos) {
      // Check if filename ends with just the data type
      if (filename.length() >= data_type.length() + 1 &&
          filename.substr(filename.length() - data_type.length() - 1) ==
              "." + data_type) {
        return filename;
      }
      throw std::invalid_argument("Invalid filename: Filename '" + filename +
                                  "' must have '." + data_type + "' suffix");
    }

    std::string base = filename.substr(0, last_dot);
    const std::string expected_suffix = "." + data_type;
    if (base.length() < expected_suffix.length() ||
        base.compare(base.length() - expected_suffix.length(),
                     expected_suffix.length(), expected_suffix) != 0) {
      throw std::invalid_argument("Invalid filename: Filename '" + filename +
                                  "' must have '." + data_type +
                                  ".' before the file extension");
    }

    return filename;
  }

  /**
   * @brief Validate filename has the correct data type suffix for reading
   * @param filename Filename to validate (e.g., "example.structure.json")
   * @param data_type Expected data structure type (e.g., "structure")
   * @return The original filename if valid
   * @throws std::invalid_argument if filename doesn't have correct data type
   */
  static std::string validate_read_suffix(const std::string &filename,
                                          const std::string &data_type) {
    // Find the last dot (extension)
    size_t last_dot = filename.find_last_of('.');
    if (last_dot == std::string::npos) {
      // Check if filename ends with just the data type
      if (filename.length() >= data_type.length() + 1 &&
          filename.substr(filename.length() - data_type.length() - 1) ==
              "." + data_type) {
        return filename;
      }
      throw std::invalid_argument("Invalid filename: Filename '" + filename +
                                  "' must have '." + data_type + "' suffix");
    }

    std::string base = filename.substr(0, last_dot);
    const std::string expected_suffix = "." + data_type;
    if (base.length() < expected_suffix.length() ||
        base.compare(base.length() - expected_suffix.length(),
                     expected_suffix.length(), expected_suffix) != 0) {
      throw std::invalid_argument("Invalid filename: Filename '" + filename +
                                  "' must have '." + data_type +
                                  ".' before the file extension");
    }

    return filename;
  }
};

}  // namespace qdk::chemistry::data
