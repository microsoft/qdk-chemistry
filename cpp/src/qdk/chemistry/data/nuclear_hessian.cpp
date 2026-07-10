// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <fstream>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

NuclearHessian::NuclearHessian(std::shared_ptr<Structure> structure,
                               const Eigen::MatrixXd& hessian_matrix)
    : structure_(std::move(structure)), matrix_(hessian_matrix) {
  if (!_is_valid()) {
    throw std::invalid_argument(
        "NuclearHessian requires a non-null structure and a 3*N by 3*N matrix");
  }
}

const std::shared_ptr<Structure> NuclearHessian::get_structure() const {
  if (!structure_) {
    throw std::runtime_error("No structure is associated with this Hessian");
  }
  return structure_;
}

Eigen::Matrix3d NuclearHessian::get_atom_pair_block(
    size_t row_atom_index, size_t column_atom_index) const {
  const auto num_atoms = get_structure()->get_num_atoms();
  if (row_atom_index >= num_atoms) {
    throw std::out_of_range("Row atom index " + std::to_string(row_atom_index) +
                            " is out of range for " +
                            std::to_string(num_atoms) + " atoms");
  }
  if (column_atom_index >= num_atoms) {
    throw std::out_of_range(
        "Column atom index " + std::to_string(column_atom_index) +
        " is out of range for " + std::to_string(num_atoms) + " atoms");
  }

  const auto row_offset = static_cast<Eigen::Index>(3 * row_atom_index);
  const auto column_offset = static_cast<Eigen::Index>(3 * column_atom_index);
  return matrix_.block<3, 3>(row_offset, column_offset);
}

std::string NuclearHessian::get_summary() const {
  std::ostringstream oss;
  oss << "NuclearHessian(" << get_structure()->get_num_atoms() << " atoms, "
      << matrix_.rows() << "x" << matrix_.cols() << ")";
  return oss.str();
}

void NuclearHessian::to_file(const std::string& filename,
                             const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

nlohmann::json NuclearHessian::to_json() const {
  nlohmann::json j;
  j["serialization_version"] = SERIALIZATION_VERSION;
  j["type"] = "NuclearHessian";
  j["structure"] = get_structure()->to_json();
  j["matrix"] = matrix_to_json(matrix_);
  j["units"] = "hartree/bohr^2";
  return j;
}

void NuclearHessian::to_json_file(const std::string& filename) const {
  auto validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(NuclearHessian));
  _to_json_file(validated_filename);
}

void NuclearHessian::to_hdf5(H5::Group& group) const {
  auto scalar_space = H5::DataSpace(H5S_SCALAR);
  auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);

  auto version_attr =
      group.createAttribute("serialization_version", str_type, scalar_space);
  std::string version_str(SERIALIZATION_VERSION);
  version_attr.write(str_type, version_str);

  auto type_attr = group.createAttribute("type", str_type, scalar_space);
  std::string type_str = "NuclearHessian";
  type_attr.write(str_type, type_str);

  auto units_attr = group.createAttribute("units", str_type, scalar_space);
  std::string units_str = "hartree/bohr^2";
  units_attr.write(str_type, units_str);

  auto structure_group = group.createGroup("structure");
  get_structure()->to_hdf5(structure_group);
  save_matrix_to_group(group, "matrix", matrix_);
}

void NuclearHessian::to_hdf5_file(const std::string& filename) const {
  auto validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(NuclearHessian));
  _to_hdf5_file(validated_filename);
}

std::shared_ptr<NuclearHessian> NuclearHessian::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported file type: " + type +
                              ". Supported types are: json, hdf5");
}

std::shared_ptr<NuclearHessian> NuclearHessian::from_json_file(
    const std::string& filename) {
  auto validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "nuclear_hessian");
  return _from_json_file(validated_filename);
}

std::shared_ptr<NuclearHessian> NuclearHessian::from_json(
    const nlohmann::json& j) {
  if (j.contains("serialization_version")) {
    validate_serialization_version(
        SERIALIZATION_VERSION, j["serialization_version"].get<std::string>());
  }
  if (j.contains("type") && j["type"].get<std::string>() != "NuclearHessian") {
    throw std::runtime_error("Invalid type in JSON data");
  }
  if (!j.contains("structure") || !j.contains("matrix")) {
    throw std::runtime_error(
        "NuclearHessian JSON requires structure and matrix fields");
  }

  return std::make_shared<NuclearHessian>(Structure::from_json(j["structure"]),
                                          json_to_matrix(j["matrix"]));
}

std::shared_ptr<NuclearHessian> NuclearHessian::from_hdf5_file(
    const std::string& filename) {
  auto validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "nuclear_hessian");
  return _from_hdf5_file(validated_filename);
}

std::shared_ptr<NuclearHessian> NuclearHessian::from_hdf5(H5::Group& group) {
  if (group.attrExists("serialization_version")) {
    auto attr = group.openAttribute("serialization_version");
    auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    std::string version;
    attr.read(str_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);
  }
  if (group.attrExists("type")) {
    auto attr = group.openAttribute("type");
    auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    std::string type;
    attr.read(str_type, type);
    if (type != "NuclearHessian") {
      throw std::runtime_error("Invalid type in HDF5 data");
    }
  }
  auto structure_group = group.openGroup("structure");
  return std::make_shared<NuclearHessian>(
      Structure::from_hdf5(structure_group),
      load_matrix_from_group(group, "matrix"));
}

bool NuclearHessian::_is_valid() const {
  if (!structure_ || matrix_.rows() != matrix_.cols()) {
    return false;
  }
  const auto expected =
      static_cast<Eigen::Index>(3 * structure_->get_num_atoms());
  return matrix_.rows() == expected && matrix_.cols() == expected;
}

void NuclearHessian::_to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  file << to_json().dump(2);
}

void NuclearHessian::_to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

std::shared_ptr<NuclearHessian> NuclearHessian::_from_json_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open NuclearHessian JSON file: " +
                             filename);
  }
  nlohmann::json j;
  file >> j;
  return from_json(j);
}

std::shared_ptr<NuclearHessian> NuclearHessian::_from_hdf5_file(
    const std::string& filename) {
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

void NuclearHessian::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, get_structure()->content_hash());
  hash_value(ctx, matrix_);
}

}  // namespace qdk::chemistry::data
