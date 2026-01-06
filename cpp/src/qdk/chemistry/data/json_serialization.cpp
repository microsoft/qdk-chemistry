// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "json_serialization.hpp"

#include <H5Cpp.h>

#include <Eigen/Dense>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>

namespace qdk::chemistry::data {

nlohmann::json matrix_to_json(const Eigen::MatrixXd& matrix) {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j = nlohmann::json::array();
  for (int i = 0; i < vector.size(); ++i) {
    j.push_back(vector(i));
  }
  return j;
}

nlohmann::json vector_variant_to_json(const VectorVariant& vec_var,
                                      bool is_complex) {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j_vec;
  if (is_complex) {
    const auto& vec_c = std::get<Eigen::VectorXcd>(vec_var);
    for (int i = 0; i < vec_c.size(); ++i) {
      j_vec.push_back({vec_c(i).real(), vec_c(i).imag()});
    }
    return j_vec;
  } else {
    const auto& vec_r = std::get<Eigen::VectorXd>(vec_var);
    for (int i = 0; i < vec_r.size(); ++i) {
      j_vec.push_back(vec_r(i));
    }
  }
  return j_vec;
}

nlohmann::json matrix_variant_to_json(const MatrixVariant& mat_var,
                                      bool is_complex) {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j_mat = nlohmann::json::array();
  if (is_complex) {
    const auto& mat_c = std::get<Eigen::MatrixXcd>(mat_var);
    for (int row = 0; row < mat_c.rows(); ++row) {
      nlohmann::json row_array = nlohmann::json::array();
      for (int col = 0; col < mat_c.cols(); ++col) {
        row_array.push_back({mat_c(row, col).real(), mat_c(row, col).imag()});
      }
      j_mat.push_back(row_array);
    }
  } else {
    const auto& mat_r = std::get<Eigen::MatrixXd>(mat_var);
    for (int row = 0; row < mat_r.rows(); ++row) {
      nlohmann::json row_array = nlohmann::json::array();
      for (int col = 0; col < mat_r.cols(); ++col) {
        row_array.push_back(mat_r(row, col));
      }
      j_mat.push_back(row_array);
    }
  }
  return j_mat;
}

VectorVariant json_to_vector_variant(const nlohmann::json& j_vec,
                                     bool is_complex = false) {
  QDK_LOG_TRACE_ENTERING();
  VectorVariant vec_var;
  if (is_complex) {
    if (!j_vec.is_array() || j_vec.empty() || !j_vec[0].is_array()) {
      throw std::runtime_error(
          "Invalid complex format: expected array of [real, imag] pairs");
    }
    Eigen::VectorXcd vec(j_vec.size());
    for (size_t i = 0; i < j_vec.size(); ++i) {
      if (j_vec[i].size() != 2) {
        throw std::runtime_error(
            "Invalid complex format: expected array of [real, imag] pairs");
      }
      vec(i) = std::complex<double>(j_vec[i][0], j_vec[i][1]);
    }
    vec_var = vec;
  } else {
    if (!j_vec.is_array()) {
      throw std::runtime_error("Invalid format: expected array of numbers");
    }
    Eigen::VectorXd vec(j_vec.size());
    for (size_t i = 0; i < j_vec.size(); ++i) {
      vec(i) = j_vec[i];
    }
    vec_var = vec;
  }
  return vec_var;
}

MatrixVariant json_to_matrix_variant(const nlohmann::json& j_mat,
                                     bool is_complex = false) {
  QDK_LOG_TRACE_ENTERING();
  if (!j_mat.is_array() || j_mat.empty()) {
    throw std::runtime_error(
        "Invalid format: expected non-empty array for matrix");
  }

  MatrixVariant mat_var;
  const int rows = static_cast<int>(j_mat.size());

  if (is_complex) {
    if (!j_mat[0].is_array() || j_mat[0].empty() || !j_mat[0][0].is_array()) {
      throw std::runtime_error(
          "Invalid complex matrix format: expected array of rows, each "
          "containing [real, imag] pairs");
    }
    const int cols = static_cast<int>(j_mat[0].size());
    Eigen::MatrixXcd mat(rows, cols);

    for (int row = 0; row < rows; ++row) {
      if (!j_mat[row].is_array() ||
          static_cast<int>(j_mat[row].size()) != cols) {
        throw std::runtime_error(
            "Invalid complex matrix format: all rows must have the same "
            "length");
      }
      for (int col = 0; col < cols; ++col) {
        if (!j_mat[row][col].is_array() || j_mat[row][col].size() != 2) {
          throw std::runtime_error(
              "Invalid complex matrix format: expected [real, imag] pairs");
        }
        mat(row, col) =
            std::complex<double>(j_mat[row][col][0], j_mat[row][col][1]);
      }
    }
    mat_var = mat;
  } else {
    if (!j_mat[0].is_array()) {
      throw std::runtime_error("Invalid matrix format: expected array of rows");
    }
    const int cols = static_cast<int>(j_mat[0].size());
    Eigen::MatrixXd mat(rows, cols);

    for (int row = 0; row < rows; ++row) {
      if (!j_mat[row].is_array() ||
          static_cast<int>(j_mat[row].size()) != cols) {
        throw std::runtime_error(
            "Invalid matrix format: all rows must have the same length");
      }
      for (int col = 0; col < cols; ++col) {
        mat(row, col) = j_mat[row][col].get<double>();
      }
    }
    mat_var = mat;
  }
  return mat_var;
}

Eigen::MatrixXd json_to_matrix(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
