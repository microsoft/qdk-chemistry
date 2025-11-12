// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "json_serialization.hpp"

#include <H5Cpp.h>

#include <stdexcept>
#include <string>
#include <tuple>

namespace qdk::chemistry::data {

nlohmann::json matrix_to_json(const Eigen::MatrixXd& matrix) {
  nlohmann::json j = nlohmann::json::array();
  for (int row = 0; row < matrix.rows(); ++row) {
    nlohmann::json row_array = nlohmann::json::array();
    for (int col = 0; col < matrix.cols(); ++col) {
      row_array.push_back(matrix(row, col));
    }
    j.push_back(row_array);
  }
  return j;
}

nlohmann::json vector_to_json(const Eigen::VectorXd& vector) {
  nlohmann::json j = nlohmann::json::array();
  for (int i = 0; i < vector.size(); ++i) {
    j.push_back(vector(i));
  }
  return j;
}

Eigen::MatrixXd json_to_matrix(const nlohmann::json& j) {
  if (!j.is_array() || j.empty()) {
    throw std::invalid_argument(
        "JSON must be a non-empty array for matrix conversion");
  }

  const int rows = static_cast<int>(j.size());
  const int cols = static_cast<int>(j[0].size());

  Eigen::MatrixXd matrix(rows, cols);
  for (int row = 0; row < rows; ++row) {
    if (!j[row].is_array() || static_cast<int>(j[row].size()) != cols) {
      throw std::invalid_argument(
          "All rows must have the same length for matrix conversion");
    }
    for (int col = 0; col < cols; ++col) {
      matrix(row, col) = j[row][col].get<double>();
    }
  }
  return matrix;
}

Eigen::VectorXd json_to_vector(const nlohmann::json& j) {
  if (!j.is_array()) {
    throw std::invalid_argument("JSON must be an array for vector conversion");
  }

  const int size = static_cast<int>(j.size());
  Eigen::VectorXd vector(size);
  for (int i = 0; i < size; ++i) {
    vector(i) = j[i].get<double>();
  }
  return vector;
}

std::tuple<int, int, int> parse_version_string(
    const std::string& version_string) {
  // Expected format: "major.minor.patch"
  std::size_t first_dot = version_string.find('.');
  std::size_t second_dot = version_string.find('.', first_dot + 1);

  if (first_dot == std::string::npos || second_dot == std::string::npos) {
    throw std::runtime_error(
        "Invalid version string format. Expected 'major.minor.patch', got: " +
        version_string);
  }

  try {
    int major = std::stoi(version_string.substr(0, first_dot));
    int minor = std::stoi(
        version_string.substr(first_dot + 1, second_dot - first_dot - 1));
    int patch = std::stoi(version_string.substr(second_dot + 1));

    return std::make_tuple(major, minor, patch);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Invalid version string format. Expected 'major.minor.patch', got: " +
        version_string);
  }
}

void validate_serialization_version(const std::string& expected_version,
                                    const std::string& found_version) {
  if (expected_version == found_version) {
    return;  // Exact match is always valid
  }

  auto [expected_major, expected_minor, expected_patch] =
      parse_version_string(expected_version);
  auto [found_major, found_minor, found_patch] =
      parse_version_string(found_version);

  // Major version must match exactly
  if (expected_major != found_major) {
    throw std::runtime_error(
        "Serialization version major mismatch. Expected: " + expected_version +
        ", Found: " + found_version +
        ". Major version differences are not compatible.");
  }

  // Minor version must match exactly
  if (expected_minor != found_minor) {
    throw std::runtime_error(
        "Serialization version minor mismatch. Expected: " + expected_version +
        ", Found: " + found_version +
        ". Minor version differences are not compatible.");
  }

  // Patch version differences are allowed (backward compatibility)
  // No need to check patch version
}

}  // namespace qdk::chemistry::data
