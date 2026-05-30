// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <fstream>
#include <qdk/chemistry/data/orbital_containers/basis_coefficients.hpp>
#include <sstream>
#include <stdexcept>

namespace qdk::chemistry::data {

namespace {

constexpr const char* kHdf5JsonDataset = "basis_coefficients_json";

void write_json_string(H5::Group& group, const std::string& payload) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  auto dataset = group.createDataSet(kHdf5JsonDataset, str_type, scalar_space);
  dataset.write(payload, str_type);
}

std::string read_json_string(H5::Group& group) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  auto dataset = group.openDataSet(kHdf5JsonDataset);
  std::string payload;
  dataset.read(payload, str_type);
  return payload;
}

}  // namespace

BasisCoefficients::BasisCoefficients(std::shared_ptr<const Sbt> coefficients)
    : _coefficients(std::move(coefficients)) {
  if (!_coefficients) {
    throw std::invalid_argument(
        "BasisCoefficients requires a non-null symmetry-blocked tensor.");
  }
}

std::string BasisCoefficients::get_summary() const {
  std::ostringstream oss;
  oss << "BasisCoefficients(" << _coefficients->get_summary() << ")";
  return oss.str();
}

nlohmann::json BasisCoefficients::to_json() const {
  return nlohmann::json{{"type", "basis_coefficients"},
                        {"tensor", _coefficients->to_json()}};
}

void BasisCoefficients::to_json_file(const std::string& filename) const {
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  out << to_json().dump(2);
}

void BasisCoefficients::to_hdf5(H5::Group& group) const {
  write_json_string(group, to_json().dump());
}

void BasisCoefficients::to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

void BasisCoefficients::to_file(const std::string& filename,
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

std::shared_ptr<BasisCoefficients> BasisCoefficients::from_json(
    const nlohmann::json& j) {
  return std::make_shared<BasisCoefficients>(Sbt::from_json(j.at("tensor")));
}

std::shared_ptr<BasisCoefficients> BasisCoefficients::from_json_file(
    const std::string& filename) {
  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  nlohmann::json j;
  in >> j;
  return from_json(j);
}

std::shared_ptr<BasisCoefficients> BasisCoefficients::from_hdf5(
    H5::Group& group) {
  return from_json(nlohmann::json::parse(read_json_string(group)));
}

std::shared_ptr<BasisCoefficients> BasisCoefficients::from_hdf5_file(
    const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

std::shared_ptr<BasisCoefficients> BasisCoefficients::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported file type: " + type +
                              ". Supported types are: json, hdf5");
}

}  // namespace qdk::chemistry::data
