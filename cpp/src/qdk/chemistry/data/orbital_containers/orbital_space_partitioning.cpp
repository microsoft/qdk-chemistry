// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <fstream>
#include <qdk/chemistry/data/errors.hpp>
#include <qdk/chemistry/data/orbital_containers/orbital_space_partitioning.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace qdk::chemistry::data {

namespace {

constexpr const char* kHdf5JsonDataset = "orbital_space_partitioning_json";

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

OrbitalSpacePartitioning::OrbitalSpacePartitioning(IndexSetArray spaces)
    : _spaces(std::move(spaces)) {
  _validate();
}

void OrbitalSpacePartitioning::_validate() const {
  for (const auto& set : _spaces) {
    if (!set) {
      throw std::invalid_argument(
          "OrbitalSpacePartitioning requires five non-null index sets.");
    }
  }
  const auto& reference_symmetries = _spaces[0]->symmetries();
  const auto& reference_extents = _spaces[0]->extents();
  for (std::size_t i = 1; i < kNumSpaces; ++i) {
    if (!(*_spaces[i]->symmetries() == *reference_symmetries)) {
      throw SymmetryConditionError(
          "OrbitalSpacePartitioning index sets must share a common symmetry "
          "vocabulary.");
    }
    if (_spaces[i]->extents() != reference_extents) {
      throw SymmetryConditionError(
          "OrbitalSpacePartitioning index sets must share common mode "
          "extents.");
    }
  }
}

std::shared_ptr<OrbitalSpacePartitioning> OrbitalSpacePartitioning::all_active(
    std::shared_ptr<const Symmetries> symmetries,
    std::unordered_map<SymmetryLabel, std::size_t> mo_extents) {
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> active_indices;
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> empty_indices;
  for (const auto& [label, extent] : mo_extents) {
    std::vector<std::uint32_t> all(extent);
    for (std::uint32_t i = 0; i < extent; ++i) {
      all[i] = i;
    }
    active_indices.emplace(label, std::move(all));
    empty_indices.emplace(label, std::vector<std::uint32_t>{});
  }

  auto make_set =
      [&](std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>
              indices) {
        return std::make_shared<const SymmetryBlockedIndexSet>(
            symmetries, mo_extents, std::move(indices));
      };

  IndexSetArray spaces{
      make_set(empty_indices),              // Frozen
      make_set(empty_indices),              // Inactive
      make_set(std::move(active_indices)),  // Active
      make_set(empty_indices),              // Virtual
      make_set(std::move(empty_indices))    // External
  };
  return std::make_shared<OrbitalSpacePartitioning>(std::move(spaces));
}

std::string OrbitalSpacePartitioning::get_summary() const {
  std::ostringstream oss;
  oss << "OrbitalSpacePartitioning(frozen=" << frozen()->get_summary()
      << ", inactive=" << inactive()->get_summary()
      << ", active=" << active()->get_summary()
      << ", virtual=" << virtual_orbitals()->get_summary()
      << ", external=" << external()->get_summary() << ")";
  return oss.str();
}

nlohmann::json OrbitalSpacePartitioning::to_json() const {
  nlohmann::json spaces = nlohmann::json::array();
  for (const auto& set : _spaces) {
    spaces.push_back(set->to_json());
  }
  return nlohmann::json{{"type", "orbital_space_partitioning"},
                        {"spaces", std::move(spaces)}};
}

void OrbitalSpacePartitioning::to_json_file(const std::string& filename) const {
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  out << to_json().dump(2);
}

void OrbitalSpacePartitioning::to_hdf5(H5::Group& group) const {
  write_json_string(group, to_json().dump());
}

void OrbitalSpacePartitioning::to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

void OrbitalSpacePartitioning::to_file(const std::string& filename,
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

std::shared_ptr<OrbitalSpacePartitioning> OrbitalSpacePartitioning::from_json(
    const nlohmann::json& j) {
  const auto& spaces_json = j.at("spaces");
  if (spaces_json.size() != kNumSpaces) {
    throw std::invalid_argument(
        "OrbitalSpacePartitioning JSON must contain five index sets.");
  }
  IndexSetArray spaces;
  for (std::size_t i = 0; i < kNumSpaces; ++i) {
    spaces[i] = SymmetryBlockedIndexSet::from_json(spaces_json[i]);
  }
  return std::make_shared<OrbitalSpacePartitioning>(std::move(spaces));
}

std::shared_ptr<OrbitalSpacePartitioning>
OrbitalSpacePartitioning::from_json_file(const std::string& filename) {
  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  nlohmann::json j;
  in >> j;
  return from_json(j);
}

std::shared_ptr<OrbitalSpacePartitioning> OrbitalSpacePartitioning::from_hdf5(
    H5::Group& group) {
  return from_json(nlohmann::json::parse(read_json_string(group)));
}

std::shared_ptr<OrbitalSpacePartitioning>
OrbitalSpacePartitioning::from_hdf5_file(const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

std::shared_ptr<OrbitalSpacePartitioning> OrbitalSpacePartitioning::from_file(
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
