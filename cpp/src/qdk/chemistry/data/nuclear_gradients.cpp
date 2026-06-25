// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <fstream>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

NuclearGradients::NuclearGradients(std::shared_ptr<Structure> structure,
                                   const Eigen::VectorXd& gradient_values)
    : structure_(std::move(structure)), values_(gradient_values) {
  if (!_is_valid()) {
    throw std::invalid_argument(
        "NuclearGradients requires a non-null structure and 3*N values");
  }
}

const std::shared_ptr<Structure> NuclearGradients::get_structure() const {
  if (!structure_) {
    throw std::runtime_error("No structure is associated with these gradients");
  }
  return structure_;
}

Eigen::MatrixXd NuclearGradients::as_matrix() const {
  const auto num_atoms = get_structure()->get_num_atoms();
  Eigen::MatrixXd matrix(num_atoms, 3);
  for (Eigen::Index atom = 0; atom < static_cast<Eigen::Index>(num_atoms);
       ++atom) {
    matrix(atom, 0) = values_(3 * atom);
    matrix(atom, 1) = values_(3 * atom + 1);
    matrix(atom, 2) = values_(3 * atom + 2);
  }
  return matrix;
}

std::string NuclearGradients::get_summary() const {
  std::ostringstream oss;
  oss << "NuclearGradients(" << get_structure()->get_num_atoms()
      << " atoms, norm=" << values_.norm() << ")";
  return oss.str();
}

void NuclearGradients::to_file(const std::string& filename,
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

nlohmann::json NuclearGradients::to_json() const {
  nlohmann::json j;
  j["serialization_version"] = SERIALIZATION_VERSION;
  j["type"] = "NuclearGradients";
  j["structure"] = get_structure()->to_json();
  j["values"] = vector_to_json(values_);
  j["units"] = "hartree/bohr";
  return j;
}

void NuclearGradients::to_json_file(const std::string& filename) const {
  auto validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(NuclearGradients));
  _to_json_file(validated_filename);
}

void NuclearGradients::to_hdf5(H5::Group& group) const {
  auto scalar_space = H5::DataSpace(H5S_SCALAR);
  auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);

  auto version_attr =
      group.createAttribute("serialization_version", str_type, scalar_space);
  std::string version_str(SERIALIZATION_VERSION);
  version_attr.write(str_type, version_str);

  auto type_attr = group.createAttribute("type", str_type, scalar_space);
  std::string type_str = "NuclearGradients";
  type_attr.write(str_type, type_str);

  auto units_attr = group.createAttribute("units", str_type, scalar_space);
  std::string units_str = "hartree/bohr";
  units_attr.write(str_type, units_str);

  auto structure_group = group.createGroup("structure");
  get_structure()->to_hdf5(structure_group);
  save_vector_to_group(group, "values", values_);
}

void NuclearGradients::to_hdf5_file(const std::string& filename) const {
  auto validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(NuclearGradients));
  _to_hdf5_file(validated_filename);
}

std::shared_ptr<NuclearGradients> NuclearGradients::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported file type: " + type +
                              ". Supported types are: json, hdf5");
}

std::shared_ptr<NuclearGradients> NuclearGradients::from_json_file(
    const std::string& filename) {
  auto validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "nuclear_gradients");
  return _from_json_file(validated_filename);
}

std::shared_ptr<NuclearGradients> NuclearGradients::from_json(
    const nlohmann::json& j) {
  if (j.contains("serialization_version")) {
    validate_serialization_version(
        SERIALIZATION_VERSION, j["serialization_version"].get<std::string>());
  }
  if (j.contains("type") &&
      j["type"].get<std::string>() != "NuclearGradients") {
    throw std::runtime_error("Invalid type in JSON data");
  }
  if (!j.contains("structure") || !j.contains("values")) {
    throw std::runtime_error(
        "NuclearGradients JSON requires structure and values fields");
  }

  return std::make_shared<NuclearGradients>(
      Structure::from_json(j["structure"]), json_to_vector(j["values"]));
}

std::shared_ptr<NuclearGradients> NuclearGradients::from_hdf5_file(
    const std::string& filename) {
  auto validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "nuclear_gradients");
  return _from_hdf5_file(validated_filename);
}

std::shared_ptr<NuclearGradients> NuclearGradients::from_hdf5(
    H5::Group& group) {
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
    if (type != "NuclearGradients") {
      throw std::runtime_error("Invalid type in HDF5 data");
    }
  }
  auto structure_group = group.openGroup("structure");
  return std::make_shared<NuclearGradients>(
      Structure::from_hdf5(structure_group),
      load_vector_from_group(group, "values"));
}

bool NuclearGradients::_is_valid() const {
  return structure_ != nullptr &&
         values_.size() ==
             static_cast<Eigen::Index>(3 * structure_->get_num_atoms());
}

void NuclearGradients::_to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  file << to_json().dump(2);
}

void NuclearGradients::_to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

std::shared_ptr<NuclearGradients> NuclearGradients::_from_json_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open NuclearGradients JSON file: " +
                             filename);
  }
  nlohmann::json j;
  file >> j;
  return from_json(j);
}

std::shared_ptr<NuclearGradients> NuclearGradients::_from_hdf5_file(
    const std::string& filename) {
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

void NuclearGradients::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, get_structure()->content_hash());
  hash_value(ctx, values_);
}

}  // namespace qdk::chemistry::data
