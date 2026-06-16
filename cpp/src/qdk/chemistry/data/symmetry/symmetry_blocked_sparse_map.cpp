// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <qdk/chemistry/data/symmetry/symmetry_blocked_sparse_map.hpp>

#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

namespace detail {

static void write_sparse_map_string_dataset(H5::Group& group,
                                            const std::string& name,
                                            const std::string& payload) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  auto dataset = group.createDataSet(name, str_type, scalar_space);
  dataset.write(payload, str_type);
}

static std::string read_sparse_map_string_dataset(H5::Group& group,
                                                  const std::string& name) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  auto dataset = group.openDataSet(name);
  std::string payload;
  dataset.read(payload, str_type);
  return payload;
}

}  // namespace detail

template <std::size_t Rank, class Scalar>
std::shared_ptr<SymmetryBlockedSparseMap<Rank, Scalar>>
SymmetryBlockedSparseMap<Rank, Scalar>::from_json(const nlohmann::json& j) {
  if (!j.contains("version")) {
    throw std::runtime_error(
        "SymmetryBlockedSparseMap JSON missing required 'version' field.");
  }
  validate_serialization_version(SERIALIZATION_VERSION,
                                 j.at("version").template get<std::string>());

  auto symmetries = Base::_symmetries_from_json(j);
  auto extents = Base::_extents_from_json(j);

  BlockMap blocks;
  for (const auto& entry : j.at("blocks")) {
    auto sparse_block = std::make_shared<SparseBlock>();
    for (const auto& e : entry.at("entries")) {
      IndexTuple idx{};
      for (std::size_t i = 0; i < Rank; ++i) {
        idx[i] = e.at(i).template get<unsigned>();
      }
      (*sparse_block)[idx] = e.at(Rank).template get<Scalar>();
    }
    auto const_block = std::const_pointer_cast<const SparseBlock>(sparse_block);
    for (const auto& key_json : entry.at("keys")) {
      std::vector<SymmetryLabel> labels;
      for (const auto& label_json : key_json) {
        labels.push_back(SymmetryLabel::from_json(label_json));
      }
      blocks.emplace(detail::make_labels<Rank>(labels), const_block);
    }
  }
  return std::make_shared<SymmetryBlockedSparseMap>(
      std::move(symmetries), std::move(extents), std::move(blocks));
}

template <std::size_t Rank, class Scalar>
std::shared_ptr<SymmetryBlockedSparseMap<Rank, Scalar>>
SymmetryBlockedSparseMap<Rank, Scalar>::from_json_file(
    const std::string& filename) {
  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  nlohmann::json j;
  in >> j;
  return from_json(j);
}

template <std::size_t Rank, class Scalar>
void SymmetryBlockedSparseMap<Rank, Scalar>::to_hdf5(H5::Group& group) const {
  detail::write_sparse_map_string_dataset(
      group, "symmetry_blocked_sparse_map_metadata", to_json().dump());
}

template <std::size_t Rank, class Scalar>
void SymmetryBlockedSparseMap<Rank, Scalar>::to_hdf5_file(
    const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

template <std::size_t Rank, class Scalar>
std::shared_ptr<SymmetryBlockedSparseMap<Rank, Scalar>>
SymmetryBlockedSparseMap<Rank, Scalar>::from_hdf5(H5::Group& group) {
  if (!group.nameExists("symmetry_blocked_sparse_map_metadata")) {
    throw std::runtime_error(
        "SymmetryBlockedSparseMap HDF5 metadata dataset not found.");
  }
  auto metadata = nlohmann::json::parse(detail::read_sparse_map_string_dataset(
      group, "symmetry_blocked_sparse_map_metadata"));
  return from_json(metadata);
}

template <std::size_t Rank, class Scalar>
std::shared_ptr<SymmetryBlockedSparseMap<Rank, Scalar>>
SymmetryBlockedSparseMap<Rank, Scalar>::from_hdf5_file(
    const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

template class SymmetryBlockedSparseMap<4, double>;

}  // namespace qdk::chemistry::data
