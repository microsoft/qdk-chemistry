// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <fstream>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qdk::chemistry::data {

namespace detail {

static constexpr const char* k_sbis_hdf5_json_dataset =
    "symmetry_blocked_index_set_json";

}  // namespace detail

SymmetryBlockedIndexSet::BlockMap SymmetryBlockedIndexSet::_build_block_map(
    std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>& indices) {
  BlockMap blocks;
  // Build shared_ptr blocks, reusing the same pointer for orbit-equivalent
  // labels that carry identical index lists.
  std::unordered_map<Labels, BlockPtr, LabelsHash<1>> built;
  for (auto& [label, list] : indices) {
    Labels key{label};
    // Check if a spin partner already exists with equal data.
    bool aliased = false;
    for (const auto& [existing_key, existing_ptr] : built) {
      if (*existing_ptr == list) {
        blocks.emplace(key, existing_ptr);
        aliased = true;
        break;
      }
    }
    if (!aliased) {
      auto ptr =
          std::make_shared<const std::vector<std::uint32_t>>(std::move(list));
      blocks.emplace(key, ptr);
      built.emplace(key, ptr);
    }
  }
  return blocks;
}

SymmetryBlockedIndexSet::SymmetryBlockedIndexSet(
    std::shared_ptr<const Symmetries> symmetries,
    std::unordered_map<SymmetryLabel, std::size_t> extents,
    std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices)
    : Base(SymmetriesArray{std::move(symmetries)},
           ExtentsArray{std::move(extents)}, _build_block_map(indices)) {
  _validate_indices();
}

void SymmetryBlockedIndexSet::_validate_indices() const {
  for (const auto& [labels, ptr] : _blocks) {
    const auto extent = _extent_for(0, labels[0]);
    const auto& list = *ptr;
    for (std::size_t i = 0; i < list.size(); ++i) {
      if (list[i] >= extent) {
        throw std::out_of_range(
            "SymmetryBlockedIndexSet index exceeds the declared extent.");
      }
      if (i > 0 && list[i] <= list[i - 1]) {
        throw std::invalid_argument(
            "SymmetryBlockedIndexSet indices must be strictly increasing.");
      }
    }
  }
}

std::span<const std::uint32_t> SymmetryBlockedIndexSet::indices(
    const SymmetryLabel& label) const {
  const auto& list = block(Labels{label});
  return std::span<const std::uint32_t>(list.data(), list.size());
}

std::vector<SymmetryLabel> SymmetryBlockedIndexSet::labels() const {
  std::vector<SymmetryLabel> result;
  result.reserve(_blocks.size());
  for (const auto& [labels, ptr] : _blocks) {
    (void)ptr;
    result.push_back(labels[0]);
  }
  return result;
}

std::string SymmetryBlockedIndexSet::get_summary() const {
  std::ostringstream oss;
  oss << "SymmetryBlockedIndexSet(labels=" << _blocks.size() << ")";
  return oss.str();
}

nlohmann::json SymmetryBlockedIndexSet::to_json() const {
  nlohmann::json j;
  j["type"] = "SymmetryBlockedIndexSet";
  j["symmetries"] = symmetries()->to_json();

  nlohmann::json extents_json = nlohmann::json::array();
  for (const auto& [label, extent] : extents()) {
    extents_json.push_back(
        nlohmann::json{{"label", label.to_json()}, {"extent", extent}});
  }
  j["extents"] = std::move(extents_json);

  nlohmann::json indices_json = nlohmann::json::array();
  for (const auto& [labels, ptr] : _blocks) {
    indices_json.push_back(
        nlohmann::json{{"label", labels[0].to_json()}, {"indices", *ptr}});
  }
  j["indices"] = std::move(indices_json);
  return j;
}

void SymmetryBlockedIndexSet::to_json_file(const std::string& filename) const {
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  out << to_json().dump(2);
}

void SymmetryBlockedIndexSet::to_hdf5(H5::Group& group) const {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  auto dataset = group.createDataSet(detail::k_sbis_hdf5_json_dataset, str_type,
                                     scalar_space);
  dataset.write(to_json().dump(), str_type);
}

void SymmetryBlockedIndexSet::to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

void SymmetryBlockedIndexSet::to_file(const std::string& filename,
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

std::shared_ptr<SymmetryBlockedIndexSet> SymmetryBlockedIndexSet::from_json(
    const nlohmann::json& j) {
  auto symmetries = std::make_shared<const Symmetries>(
      Symmetries::from_json(j.at("symmetries")));

  std::unordered_map<SymmetryLabel, std::size_t> extents;
  for (const auto& entry : j.at("extents")) {
    extents.emplace(SymmetryLabel::from_json(entry.at("label")),
                    entry.at("extent").get<std::size_t>());
  }

  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  for (const auto& entry : j.at("indices")) {
    indices.emplace(SymmetryLabel::from_json(entry.at("label")),
                    entry.at("indices").get<std::vector<std::uint32_t>>());
  }
  return std::make_shared<SymmetryBlockedIndexSet>(
      std::move(symmetries), std::move(extents), std::move(indices));
}

std::shared_ptr<SymmetryBlockedIndexSet>
SymmetryBlockedIndexSet::from_json_file(const std::string& filename) {
  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  nlohmann::json j;
  in >> j;
  return from_json(j);
}

std::shared_ptr<SymmetryBlockedIndexSet> SymmetryBlockedIndexSet::from_hdf5(
    H5::Group& group) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  auto dataset = group.openDataSet(detail::k_sbis_hdf5_json_dataset);
  std::string payload;
  dataset.read(payload, str_type);
  return from_json(nlohmann::json::parse(payload));
}

std::shared_ptr<SymmetryBlockedIndexSet>
SymmetryBlockedIndexSet::from_hdf5_file(const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

std::shared_ptr<SymmetryBlockedIndexSet> SymmetryBlockedIndexSet::from_file(
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
