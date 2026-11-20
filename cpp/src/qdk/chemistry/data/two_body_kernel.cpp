// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <cmath>
#include <fstream>
#include <qdk/chemistry/data/two_body_kernel.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

namespace detail {

/// Returns omega if it is a valid range-separation parameter
double checked_omega(double omega) {
  if (!std::isfinite(omega) || omega <= 0.0) {
    throw std::invalid_argument(
        "The range-separation parameter omega must be finite and positive");
  }
  return omega;
}

/// Writes a string attribute of an HDF5 group
void write_string_attribute(H5::Group& group, const std::string& name,
                            const std::string& value) {
  const auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
  group.createAttribute(name, str_type, H5::DataSpace(H5S_SCALAR))
      .write(str_type, value);
}

/// Reads a string attribute of an HDF5 group
std::string read_string_attribute(H5::Group& group, const std::string& name) {
  const auto str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
  std::string value;
  group.openAttribute(name).read(str_type, value);
  return value;
}

/// Writes the serialization version, type, and omega, if positive, of a
/// kernel to an HDF5 group
void write_kernel(H5::Group& group, const std::string& version,
                  const std::string& type, double omega = 0.0) {
  write_string_attribute(group, "serialization_version", version);
  write_string_attribute(group, "type", type);
  if (omega > 0.0) {
    group
        .createAttribute("omega", H5::PredType::NATIVE_DOUBLE,
                         H5::DataSpace(H5S_SCALAR))
        .write(H5::PredType::NATIVE_DOUBLE, &omega);
  }
}

/// A short summary of an attenuated Coulomb kernel
std::string omega_summary(const std::string& type, double omega) {
  std::ostringstream oss;
  oss << type << "(omega=" << omega << ")";
  return oss.str();
}

}  // namespace detail

void TwoBodyKernel::to_file(const std::string& filename,
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

void TwoBodyKernel::to_json_file(const std::string& filename) const {
  std::ofstream file(
      DataTypeFilename::validate_write_suffix(filename, data_type_name()));
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  file << to_json().dump(2);
}

void TwoBodyKernel::to_hdf5_file(const std::string& filename) const {
  H5::H5File file(
      DataTypeFilename::validate_write_suffix(filename, data_type_name()),
      H5F_ACC_TRUNC);
  to_hdf5(file);
}

std::shared_ptr<TwoBodyKernel> TwoBodyKernel::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported file type: " + type +
                              ". Supported types are: json, hdf5");
}

std::shared_ptr<TwoBodyKernel> TwoBodyKernel::from_json_file(
    const std::string& filename) {
  std::ifstream file(
      DataTypeFilename::validate_read_suffix(filename, data_type_name()));
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open TwoBodyKernel JSON file: " +
                             filename);
  }
  nlohmann::json j;
  file >> j;
  return from_json(j);
}

std::shared_ptr<TwoBodyKernel> TwoBodyKernel::from_json(
    const nlohmann::json& j) {
  if (j.contains("serialization_version")) {
    validate_serialization_version(
        SERIALIZATION_VERSION, j["serialization_version"].get<std::string>());
  }
  const auto type = j.at("type").get<std::string>();
  if (type == "CoulombKernel") {
    return std::make_shared<CoulombKernel>();
  } else if (type == "ErfCoulombKernel") {
    return std::make_shared<ErfCoulombKernel>(j.at("omega").get<double>());
  } else if (type == "ErfcCoulombKernel") {
    return std::make_shared<ErfcCoulombKernel>(j.at("omega").get<double>());
  }
  throw std::runtime_error("Unknown two-body kernel type: " + type);
}

std::shared_ptr<TwoBodyKernel> TwoBodyKernel::from_hdf5_file(
    const std::string& filename) {
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }
  H5::H5File file(
      DataTypeFilename::validate_read_suffix(filename, data_type_name()),
      H5F_ACC_RDONLY);
  return from_hdf5(file);
}

std::shared_ptr<TwoBodyKernel> TwoBodyKernel::from_hdf5(H5::Group& group) {
  if (group.attrExists("serialization_version")) {
    validate_serialization_version(
        SERIALIZATION_VERSION,
        detail::read_string_attribute(group, "serialization_version"));
  }
  const auto type = detail::read_string_attribute(group, "type");
  auto omega = [&group] {
    double value = 0.0;
    group.openAttribute("omega").read(H5::PredType::NATIVE_DOUBLE, &value);
    return value;
  };
  if (type == "CoulombKernel") {
    return std::make_shared<CoulombKernel>();
  } else if (type == "ErfCoulombKernel") {
    return std::make_shared<ErfCoulombKernel>(omega());
  } else if (type == "ErfcCoulombKernel") {
    return std::make_shared<ErfcCoulombKernel>(omega());
  }
  throw std::runtime_error("Unknown two-body kernel type: " + type);
}

std::string CoulombKernel::get_summary() const { return "CoulombKernel(1/r)"; }

nlohmann::json CoulombKernel::to_json() const {
  return {{"serialization_version", SERIALIZATION_VERSION},
          {"type", "CoulombKernel"}};
}

void CoulombKernel::to_hdf5(H5::Group& group) const {
  detail::write_kernel(group, SERIALIZATION_VERSION, "CoulombKernel");
}

void CoulombKernel::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, "CoulombKernel");
}

ErfCoulombKernel::ErfCoulombKernel(double omega)
    : _omega(detail::checked_omega(omega)) {}

std::string ErfCoulombKernel::get_summary() const {
  return detail::omega_summary("ErfCoulombKernel", _omega);
}

nlohmann::json ErfCoulombKernel::to_json() const {
  return {{"serialization_version", SERIALIZATION_VERSION},
          {"type", "ErfCoulombKernel"},
          {"omega", _omega}};
}

void ErfCoulombKernel::to_hdf5(H5::Group& group) const {
  detail::write_kernel(group, SERIALIZATION_VERSION, "ErfCoulombKernel",
                       _omega);
}

void ErfCoulombKernel::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, "ErfCoulombKernel");
  hash_value(ctx, _omega);
}

ErfcCoulombKernel::ErfcCoulombKernel(double omega)
    : _omega(detail::checked_omega(omega)) {}

std::string ErfcCoulombKernel::get_summary() const {
  return detail::omega_summary("ErfcCoulombKernel", _omega);
}

nlohmann::json ErfcCoulombKernel::to_json() const {
  return {{"serialization_version", SERIALIZATION_VERSION},
          {"type", "ErfcCoulombKernel"},
          {"omega", _omega}};
}

void ErfcCoulombKernel::to_hdf5(H5::Group& group) const {
  detail::write_kernel(group, SERIALIZATION_VERSION, "ErfcCoulombKernel",
                       _omega);
}

void ErfcCoulombKernel::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, "ErfcCoulombKernel");
  hash_value(ctx, _omega);
}

}  // namespace qdk::chemistry::data
