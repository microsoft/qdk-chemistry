// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

std::string StabilityResult::get_summary() const {
  QDK_LOG_TRACE_ENTERING();
  if (internal_eigenvalues_.size() == 0 && external_eigenvalues_.size() == 0) {
    return "StabilityResult(empty)";
  }

  std::ostringstream oss;
  oss << "StabilityResult(";

  if (is_stable()) {
    oss << "stable, ";
  } else {
    oss << "unstable, ";
  }

  oss << "internal: " << internal_eigenvalues_.size()
      << " eigenvalues, external: " << external_eigenvalues_.size()
      << " eigenvalues";

  if (!_is_valid()) {
    oss << " [INVALID]";
  }

  oss << ")";

  if (internal_eigenvalues_.size() > 0) {
    oss << ", smallest internal eigenvalue: " << std::setprecision(6)
        << get_smallest_internal_eigenvalue();
  }
  if (external_eigenvalues_.size() > 0) {
    oss << ", smallest external eigenvalue: " << std::setprecision(6)
        << get_smallest_external_eigenvalue();
  }
  return oss.str();
}

bool StabilityResult::empty() const {
  QDK_LOG_TRACE_ENTERING();
  return !has_internal_result() && !has_external_result();
}

bool StabilityResult::has_internal_result() const {
  QDK_LOG_TRACE_ENTERING();
  return internal_eigenvalues_.size() > 0;
}

bool StabilityResult::has_external_result() const {
  QDK_LOG_TRACE_ENTERING();
  return external_eigenvalues_.size() > 0;
}

double StabilityResult::get_smallest_internal_eigenvalue() const {
  QDK_LOG_TRACE_ENTERING();
  if (internal_eigenvalues_.size() == 0) {
    throw std::runtime_error("No internal eigenvalues available");
  }
  return internal_eigenvalues_.minCoeff();
}

double StabilityResult::get_smallest_external_eigenvalue() const {
  QDK_LOG_TRACE_ENTERING();
  if (external_eigenvalues_.size() == 0) {
    throw std::runtime_error("No external eigenvalues available");
  }
  return external_eigenvalues_.minCoeff();
}

double StabilityResult::get_smallest_eigenvalue() const {
  QDK_LOG_TRACE_ENTERING();
  if (internal_eigenvalues_.size() == 0 && external_eigenvalues_.size() == 0) {
    throw std::runtime_error("No eigenvalues available");
  }

  double smallest = std::numeric_limits<double>::max();

  if (internal_eigenvalues_.size() > 0) {
    smallest = std::min(smallest, internal_eigenvalues_.minCoeff());
  }

  if (external_eigenvalues_.size() > 0) {
    smallest = std::min(smallest, external_eigenvalues_.minCoeff());
  }

  return smallest;
}

std::pair<double, Eigen::VectorXd>
StabilityResult::get_smallest_internal_eigenvalue_and_vector() const {
  QDK_LOG_TRACE_ENTERING();
  if (internal_eigenvalues_.size() == 0) {
    throw std::runtime_error("No internal eigenvalues available");
  }

  Eigen::Index min_index;
  double min_eigenvalue = internal_eigenvalues_.minCoeff(&min_index);
  Eigen::VectorXd min_eigenvector = internal_eigenvectors_.col(min_index);

  return std::make_pair(min_eigenvalue, min_eigenvector);
}

std::pair<double, Eigen::VectorXd>
StabilityResult::get_smallest_external_eigenvalue_and_vector() const {
  QDK_LOG_TRACE_ENTERING();
  if (external_eigenvalues_.size() == 0) {
    throw std::runtime_error("No external eigenvalues available");
  }

  Eigen::Index min_index;
  double min_eigenvalue = external_eigenvalues_.minCoeff(&min_index);
  Eigen::VectorXd min_eigenvector = external_eigenvectors_.col(min_index);

  return std::make_pair(min_eigenvalue, min_eigenvector);
}

std::pair<double, Eigen::VectorXd>
StabilityResult::get_smallest_eigenvalue_and_vector() const {
  QDK_LOG_TRACE_ENTERING();
  if (internal_eigenvalues_.size() == 0 && external_eigenvalues_.size() == 0) {
    throw std::runtime_error("No eigenvalues available");
  }

  double smallest = std::numeric_limits<double>::max();
  Eigen::VectorXd smallest_vector;
  bool found = false;

  if (internal_eigenvalues_.size() > 0) {
    Eigen::Index min_index;
    double min_internal = internal_eigenvalues_.minCoeff(&min_index);
    if (min_internal < smallest) {
      smallest = min_internal;
      smallest_vector = internal_eigenvectors_.col(min_index);
      found = true;
    }
  }

  if (external_eigenvalues_.size() > 0) {
    Eigen::Index min_index;
    double min_external = external_eigenvalues_.minCoeff(&min_index);
    if (min_external < smallest) {
      smallest = min_external;
      smallest_vector = external_eigenvectors_.col(min_index);
      found = true;
    }
  }

  return std::make_pair(smallest, smallest_vector);
}

// === File I/O implementations ===

void StabilityResult::to_file(const std::string& filename,
                              const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(StabilityResult));

  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

nlohmann::json StabilityResult::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["serialization_version"] = SERIALIZATION_VERSION;
  j["type"] = "StabilityResult";
  j["internal_stable"] = internal_stable_;
  j["external_stable"] = external_stable_;

  if (internal_eigenvalues_.size() > 0) {
    j["internal_eigenvalues"] = vector_to_json(internal_eigenvalues_);
  }
  if (internal_eigenvectors_.size() > 0) {
    j["internal_eigenvectors"] = matrix_to_json(internal_eigenvectors_);
  }
  if (external_eigenvalues_.size() > 0) {
    j["external_eigenvalues"] = vector_to_json(external_eigenvalues_);
  }
  if (external_eigenvectors_.size() > 0) {
    j["external_eigenvectors"] = matrix_to_json(external_eigenvectors_);
  }

  return j;
}

void StabilityResult::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(StabilityResult));
  _to_json_file(filename);
}
void StabilityResult::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  // Save metadata
  auto attr_space = H5::DataSpace(H5S_SCALAR);
  auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);

  auto version_attr =
      group.createAttribute("serialization_version", str_type, attr_space);
  std::string version_str(SERIALIZATION_VERSION);
  version_attr.write(str_type, version_str);

  auto type_attr = group.createAttribute("type", str_type, attr_space);
  std::string type_str = "StabilityResult";
  type_attr.write(str_type, type_str);

  // Save stability flags
  auto bool_attr_space = H5::DataSpace(H5S_SCALAR);
  auto internal_attr = group.createAttribute(
      "internal_stable", H5::PredType::NATIVE_HBOOL, bool_attr_space);
  hbool_t internal_flag = internal_stable_ ? 1 : 0;
  internal_attr.write(H5::PredType::NATIVE_HBOOL, &internal_flag);

  auto external_attr = group.createAttribute(
      "external_stable", H5::PredType::NATIVE_HBOOL, bool_attr_space);
  hbool_t external_flag = external_stable_ ? 1 : 0;
  external_attr.write(H5::PredType::NATIVE_HBOOL, &external_flag);

  // Save eigenvalues and eigenvectors if they exist
  if (internal_eigenvalues_.size() > 0) {
    save_vector_to_group(group, "internal_eigenvalues", internal_eigenvalues_);
  }
  if (internal_eigenvectors_.size() > 0) {
    save_matrix_to_group(group, "internal_eigenvectors",
                         internal_eigenvectors_);
  }
  if (external_eigenvalues_.size() > 0) {
    save_vector_to_group(group, "external_eigenvalues", external_eigenvalues_);
  }
  if (external_eigenvectors_.size() > 0) {
    save_matrix_to_group(group, "external_eigenvectors",
                         external_eigenvectors_);
  }
}

void StabilityResult::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(StabilityResult));
  _to_hdf5_file(filename);
}
std::shared_ptr<StabilityResult> StabilityResult::from_file(
    const std::string& filename, const std::string& type) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_read_suffix(filename, "stability_result");

  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

std::shared_ptr<StabilityResult> StabilityResult::from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_read_suffix(filename, "stability_result");
  return _from_json_file(filename);
}
std::shared_ptr<StabilityResult> StabilityResult::from_json(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  // Validate version
  if (j.contains("serialization_version")) {
    validate_serialization_version(
        SERIALIZATION_VERSION, j["serialization_version"].get<std::string>());
  }

  // Validate type
  if (j.contains("type") && j["type"].get<std::string>() != "StabilityResult") {
    throw std::runtime_error("Invalid type in JSON data");
  }

  auto result = std::make_shared<StabilityResult>();

  // Load stability flags
  if (j.contains("internal_stable")) {
    result->internal_stable_ = j["internal_stable"].get<bool>();
  }
  if (j.contains("external_stable")) {
    result->external_stable_ = j["external_stable"].get<bool>();
  }

  // Load eigenvalues and eigenvectors
  if (j.contains("internal_eigenvalues")) {
    result->internal_eigenvalues_ = json_to_vector(j["internal_eigenvalues"]);
  }
  if (j.contains("internal_eigenvectors")) {
    result->internal_eigenvectors_ = json_to_matrix(j["internal_eigenvectors"]);
  }
  if (j.contains("external_eigenvalues")) {
    result->external_eigenvalues_ = json_to_vector(j["external_eigenvalues"]);
  }
  if (j.contains("external_eigenvectors")) {
    result->external_eigenvectors_ = json_to_matrix(j["external_eigenvectors"]);
  }

  // Validate the loaded result
  if (!result->_is_valid()) {
    throw std::runtime_error(
        "Invalid StabilityResult from JSON: eigenvector dimensions must match "
        "eigenvalue sizes");
  }

  return result;
}

std::shared_ptr<StabilityResult> StabilityResult::from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_read_suffix(filename, "stability_result");
  return _from_hdf5_file(filename);
}
std::shared_ptr<StabilityResult> StabilityResult::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  // Validate version if present
  if (group.attrExists("serialization_version")) {
    auto attr = group.openAttribute("serialization_version");
    auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    std::string version;
    attr.read(str_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);
  }

  // Validate type if present
  if (group.attrExists("type")) {
    auto attr = group.openAttribute("type");
    auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    std::string type;
    attr.read(str_type, type);
    if (type != "StabilityResult") {
      throw std::runtime_error("Invalid type in HDF5 data");
    }
  }

  auto result = std::make_shared<StabilityResult>();

  // Load stability flags
  if (group.attrExists("internal_stable")) {
    auto attr = group.openAttribute("internal_stable");
    hbool_t flag;
    attr.read(H5::PredType::NATIVE_HBOOL, &flag);
    result->internal_stable_ = (flag != 0);
  }
  if (group.attrExists("external_stable")) {
    auto attr = group.openAttribute("external_stable");
    hbool_t flag;
    attr.read(H5::PredType::NATIVE_HBOOL, &flag);
    result->external_stable_ = (flag != 0);
  }

  // Load eigenvalues and eigenvectors if they exist
  if (dataset_exists_in_group(group, "internal_eigenvalues")) {
    result->internal_eigenvalues_ =
        load_vector_from_group(group, "internal_eigenvalues");
  }
  if (dataset_exists_in_group(group, "internal_eigenvectors")) {
    result->internal_eigenvectors_ =
        load_matrix_from_group(group, "internal_eigenvectors");
  }
  if (dataset_exists_in_group(group, "external_eigenvalues")) {
    result->external_eigenvalues_ =
        load_vector_from_group(group, "external_eigenvalues");
  }
  if (dataset_exists_in_group(group, "external_eigenvectors")) {
    result->external_eigenvectors_ =
        load_matrix_from_group(group, "external_eigenvectors");
  }

  // Validate the loaded result
  if (!result->_is_valid()) {
    throw std::runtime_error(
        "Invalid StabilityResult from HDF5: eigenvector dimensions must match "
        "eigenvalue sizes");
  }

  return result;
}

// === Private helper methods ===

bool StabilityResult::_is_valid() const {
  QDK_LOG_TRACE_ENTERING();
  // Check that eigenvectors dimensions match eigenvalues if both are present
  if (internal_eigenvalues_.size() > 0 && internal_eigenvectors_.size() > 0) {
    if (internal_eigenvectors_.cols() != internal_eigenvalues_.size()) {
      return false;
    }
  }

  if (external_eigenvalues_.size() > 0 && external_eigenvectors_.size() > 0) {
    if (external_eigenvectors_.cols() != external_eigenvalues_.size()) {
      return false;
    }
  }

  return true;
}

void StabilityResult::_to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  file << to_json().dump(2);
  file.close();

  if (file.fail()) {
    throw std::runtime_error("Failed to write to file: " + filename);
  }
}

void StabilityResult::_to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

std::shared_ptr<StabilityResult> StabilityResult::_from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open StabilityResult JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }

  nlohmann::json j;
  file >> j;
  return from_json(j);
}

std::shared_ptr<StabilityResult> StabilityResult::_from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  // Disable HDF5 automatic error printing to stderr unless verbose mode
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }

  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open StabilityResult HDF5 file '" +
                             filename + "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    return from_hdf5(file);
  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "Unable to read StabilityResult data from HDF5 file '" + filename +
        "'. " + "HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
