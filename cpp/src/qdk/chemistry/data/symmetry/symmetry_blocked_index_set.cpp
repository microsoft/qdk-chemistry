// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>

#include <H5Cpp.h>

#include <fstream>
#include <qdk/chemistry/data/errors.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qdk::chemistry::data {

namespace {

constexpr const char* kHdf5JsonDataset = "symmetry_blocked_index_set_json";

bool label_admissible(const Symmetries& sym, const SymmetryLabel& label) {
  if (label.values().size() != sym.axes().size()) {
    return false;
  }
  for (const auto& axis : sym.axes()) {
    if (!label.has(axis.name()) || !axis.admits(*label.get(axis.name()))) {
      return false;
    }
  }
  return true;
}

}  // namespace

SymmetryBlockedIndexSet::SymmetryBlockedIndexSet(
    std::shared_ptr<const Symmetries> symmetries,
    std::unordered_map<SymmetryLabel, std::size_t> extents,
    std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices)
    : _symmetries(std::move(symmetries)),
      _extents(std::move(extents)),
      _indices(std::move(indices)) {
  _validate();
}

void SymmetryBlockedIndexSet::_validate() const {
  if (_symmetries == nullptr) {
    throw BlockLabelInvalidError(
        "SymmetryBlockedIndexSet symmetries must not be null.");
  }
  for (const auto& [label, extent] : _extents) {
    (void)extent;
    if (!label_admissible(*_symmetries, label)) {
      throw BlockLabelInvalidError(
          "SymmetryBlockedIndexSet extent label is not admissible under the "
          "supplied symmetries.");
    }
  }
  for (const auto& [label, list] : _indices) {
    auto extent_it = _extents.find(label);
    if (extent_it == _extents.end()) {
      throw BlockLabelInvalidError(
          "SymmetryBlockedIndexSet index label has no declared extent.");
    }
    const std::size_t extent = extent_it->second;
    for (std::size_t i = 0; i < list.size(); ++i) {
      if (list[i] >= extent) {
        throw IndexSetOutOfRangeError(
            "SymmetryBlockedIndexSet index exceeds the declared extent.");
      }
      if (i > 0 && list[i] <= list[i - 1]) {
        throw IndexSetNotSortedUniqueError(
            "SymmetryBlockedIndexSet indices must be strictly increasing.");
      }
    }
  }
}

std::span<const std::uint32_t> SymmetryBlockedIndexSet::indices(
    const SymmetryLabel& label) const {
  auto it = _indices.find(label);
  if (it == _indices.end()) {
    throw BlockLabelInvalidError(
        "SymmetryBlockedIndexSet has no indices for the requested label.");
  }
  return std::span<const std::uint32_t>(it->second.data(), it->second.size());
}

std::vector<SymmetryLabel> SymmetryBlockedIndexSet::labels() const {
  std::vector<SymmetryLabel> result;
  result.reserve(_indices.size());
  for (const auto& [label, list] : _indices) {
    (void)list;
    result.push_back(label);
  }
  return result;
}

std::string SymmetryBlockedIndexSet::get_summary() const {
  std::ostringstream oss;
  oss << "SymmetryBlockedIndexSet(labels=" << _indices.size() << ")";
  return oss.str();
}

nlohmann::json SymmetryBlockedIndexSet::to_json() const {
  nlohmann::json j;
  j["type"] = "SymmetryBlockedIndexSet";
  j["symmetries"] = _symmetries->to_json();

  nlohmann::json extents = nlohmann::json::array();
  for (const auto& [label, extent] : _extents) {
    extents.push_back(
        nlohmann::json{{"label", label.to_json()}, {"extent", extent}});
  }
  j["extents"] = std::move(extents);

  nlohmann::json indices = nlohmann::json::array();
  for (const auto& [label, list] : _indices) {
    indices.push_back(
        nlohmann::json{{"label", label.to_json()}, {"indices", list}});
  }
  j["indices"] = std::move(indices);
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
  auto dataset = group.createDataSet(kHdf5JsonDataset, str_type, scalar_space);
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
  auto symmetries =
      std::make_shared<const Symmetries>(Symmetries::from_json(j.at("symmetries")));

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
  auto dataset = group.openDataSet(kHdf5JsonDataset);
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
