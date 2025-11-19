// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @file json_serialization.hpp
 * @brief JSON serialization helpers for various data types
 *
 * This file contains helper functions for JSON serialization/deserialization
 * of Eigen matrices/vectors and STL containers, as well as version validation.
 */

/**
 * @brief Validate serialization version compatibility
 * @param expected_version The version string this code expects (e.g., "0.1.0")
 * @param found_version The version string found in the serialized data
 * @throws std::runtime_error if major or minor version mismatch
 */
void validate_serialization_version(const std::string& expected_version,
                                    const std::string& found_version);

/**
 * @brief Parse a semantic version string into major, minor, patch components
 * @param version_string Version string in format "major.minor.patch"
 * @return Tuple of (major, minor, patch) as integers
 * @throws std::runtime_error if version string format is invalid
 */
std::tuple<int, int, int> parse_version_string(
    const std::string& version_string);

/**
 * @brief Convert Eigen matrix to JSON array
 * @param matrix Matrix to convert
 * @return JSON array representation
 */
nlohmann::json matrix_to_json(const Eigen::MatrixXd& matrix);

/**
 * @brief Convert Eigen vector to JSON array
 * @param vector Vector to convert
 * @return JSON array representation
 */
nlohmann::json vector_to_json(const Eigen::VectorXd& vector);

/**
 * @brief Convert JSON array to Eigen matrix
 * @param j JSON array to convert
 * @return Eigen matrix
 */
Eigen::MatrixXd json_to_matrix(const nlohmann::json& j);

/**
 * @brief Convert JSON array to Eigen vector
 * @param j JSON array to convert
 * @return Eigen vector
 */
Eigen::VectorXd json_to_vector(const nlohmann::json& j);

/**
 * @brief Convert std::vector to JSON array
 * @tparam T Element type
 * @param vector Vector to convert
 * @return JSON array representation
 */
template <typename T>
nlohmann::json vector_to_json(const std::vector<T>& vector);

/**
 * @brief Convert JSON array to std::vector
 * @tparam T Element type
 * @param j JSON array to convert
 * @return std::vector
 */
template <typename T>
std::vector<T> json_to_vector(const nlohmann::json& j);

template <typename T>
nlohmann::json vector_to_json(const std::vector<T>& vector) {
  nlohmann::json j = nlohmann::json::array();
  for (const auto& element : vector) {
    j.push_back(element);
  }
  return j;
}

template <typename T>
std::vector<T> json_to_vector(const nlohmann::json& j) {
  std::vector<T> vector;
  vector.reserve(j.size());
  for (const auto& element : j) {
    vector.push_back(element.get<T>());
  }
  return vector;
}

}  // namespace qdk::chemistry::data
