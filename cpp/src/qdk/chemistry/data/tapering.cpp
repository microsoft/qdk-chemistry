// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/tapering.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

TaperingSpecification::TaperingSpecification(
    std::vector<std::size_t> qubit_indices, std::vector<int> eigenvalues)
    : qubit_indices_(std::move(qubit_indices)),
      eigenvalues_(std::move(eigenvalues)) {
  if (qubit_indices_.size() != eigenvalues_.size()) {
    throw std::invalid_argument(
        "qubit_indices length must match eigenvalues length");
  }

  auto sorted = qubit_indices_;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    throw std::invalid_argument("qubit_indices must not contain duplicates");
  }

  for (auto value : eigenvalues_) {
    if (value != 1 && value != -1) {
      throw std::invalid_argument("Eigenvalue must be +1 or -1, got " +
                                  std::to_string(value));
    }
  }
}

TaperingSpecification TaperingSpecification::symmetry_conserving_bravyi_kitaev(
    std::size_t num_modes, std::size_t n_alpha, std::size_t n_beta) {
  if (num_modes < 4 || num_modes % 2 != 0) {
    throw std::invalid_argument(
        "Symmetry-conserving Bravyi-Kitaev requires an even num_modes >= 4, "
        "got " +
        std::to_string(num_modes));
  }
  if (n_alpha > num_modes / 2) {
    throw std::invalid_argument("n_alpha (" + std::to_string(n_alpha) +
                                ") exceeds spatial orbitals (" +
                                std::to_string(num_modes / 2) + ")");
  }
  if (n_beta > num_modes / 2) {
    throw std::invalid_argument("n_beta (" + std::to_string(n_beta) +
                                ") exceeds spatial orbitals (" +
                                std::to_string(num_modes / 2) + ")");
  }

  int ev_total = ((n_alpha + n_beta) % 2 == 0) ? 1 : -1;
  int ev_alpha = (n_alpha % 2 == 0) ? 1 : -1;

  return TaperingSpecification({num_modes / 2 - 1, num_modes - 1},
                               {ev_alpha, ev_total});
}

TaperingSpecification TaperingSpecification::parity_two_qubit_reduction(
    std::size_t num_modes, std::size_t n_alpha, std::size_t n_beta) {
  return symmetry_conserving_bravyi_kitaev(num_modes, n_alpha, n_beta);
}

std::string TaperingSpecification::get_summary() const {
  std::ostringstream ss;
  ss << "TaperingSpecification\n"
     << "  Tapered qubits: " << qubit_indices_.size();
  return ss.str();
}

bool TaperingSpecification::operator==(
    const TaperingSpecification& other) const {
  return qubit_indices_ == other.qubit_indices_ &&
         eigenvalues_ == other.eigenvalues_;
}

nlohmann::json TaperingSpecification::to_json() const {
  return nlohmann::json{{"version", SERIALIZATION_VERSION},
                        {"qubit_indices", qubit_indices_},
                        {"eigenvalues", eigenvalues_}};
}

TaperingSpecification TaperingSpecification::from_json(
    const nlohmann::json& data) {
  if (!data.contains("qubit_indices") || !data.contains("eigenvalues")) {
    throw std::runtime_error(
        "TaperingSpecification JSON missing required field");
  }
  return TaperingSpecification(
      data.at("qubit_indices").get<std::vector<std::size_t>>(),
      data.at("eigenvalues").get<std::vector<int>>());
}

void TaperingSpecification::to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for writing: " + filename);
  }
  file << to_json().dump(2);
}

TaperingSpecification TaperingSpecification::from_json_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for reading: " + filename);
  }
  nlohmann::json data;
  file >> data;
  return from_json(data);
}

void TaperingSpecification::to_hdf5(H5::Group& group) const {
  hsize_t index_dims[] = {qubit_indices_.size()};
  H5::DataSpace index_space(1, index_dims);
  std::vector<unsigned long long> qubits(qubit_indices_.begin(),
                                         qubit_indices_.end());
  group.createDataSet("qubit_indices", H5::PredType::NATIVE_ULLONG, index_space)
      .write(qubits.data(), H5::PredType::NATIVE_ULLONG);

  hsize_t eigen_dims[] = {eigenvalues_.size()};
  H5::DataSpace eigen_space(1, eigen_dims);
  group.createDataSet("eigenvalues", H5::PredType::NATIVE_INT, eigen_space)
      .write(eigenvalues_.data(), H5::PredType::NATIVE_INT);

  group
      .createAttribute("version", H5::StrType(0, H5T_VARIABLE), H5::DataSpace())
      .write(H5::StrType(0, H5T_VARIABLE), std::string(SERIALIZATION_VERSION));
}

TaperingSpecification TaperingSpecification::from_hdf5(H5::Group& group) {
  auto qubit_dataset = group.openDataSet("qubit_indices");
  auto qubit_space = qubit_dataset.getSpace();
  hsize_t qubit_dims[1] = {0};
  qubit_space.getSimpleExtentDims(qubit_dims);
  std::vector<unsigned long long> qubit_buffer(qubit_dims[0]);
  qubit_dataset.read(qubit_buffer.data(), H5::PredType::NATIVE_ULLONG);
  std::vector<std::size_t> qubit_indices(qubit_buffer.begin(),
                                         qubit_buffer.end());

  auto eigen_dataset = group.openDataSet("eigenvalues");
  auto eigen_space = eigen_dataset.getSpace();
  hsize_t eigen_dims[1] = {0};
  eigen_space.getSimpleExtentDims(eigen_dims);
  std::vector<int> eigenvalues(eigen_dims[0]);
  eigen_dataset.read(eigenvalues.data(), H5::PredType::NATIVE_INT);

  return TaperingSpecification(std::move(qubit_indices),
                               std::move(eigenvalues));
}

void TaperingSpecification::to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  H5::Group root = file.openGroup("/");
  to_hdf5(root);
}

TaperingSpecification TaperingSpecification::from_hdf5_file(
    const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  H5::Group root = file.openGroup("/");
  return from_hdf5(root);
}

void TaperingSpecification::to_file(const std::string& filename,
                                    const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5" || type == "h5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported format type: " + type);
  }
}

TaperingSpecification TaperingSpecification::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  }
  if (type == "hdf5" || type == "h5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported format type: " + type);
}

}  // namespace qdk::chemistry::data
