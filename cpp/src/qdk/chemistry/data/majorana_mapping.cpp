// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <complex>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/data/tapering.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

// ── MajoranaMapping implementation ───────────────────────────────────

MajoranaMapping::MajoranaMapping(
    std::vector<SparsePauliWord> table,
    std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears,
    std::string name, std::size_t num_modes, std::size_t num_qubits,
    std::string base_encoding, std::optional<TaperingSpecification> tapering)
    : table_(std::move(table)),
      bilinears_(std::move(bilinears)),
      name_(std::move(name)),
      base_encoding_(std::move(base_encoding)),
      num_modes_(num_modes),
      num_qubits_(num_qubits),
      majorana_atomic_(!table_.empty()),
      tapering_(std::move(tapering)) {
  if (base_encoding_.empty()) {
    base_encoding_ = name_;
  }
  if (tapering_ && tapering_->num_tapered() >= num_qubits_) {
    throw std::invalid_argument(
        "MajoranaMapping tapering removes all (or more) qubits than the base "
        "mapping contains — at least one qubit must remain");
  }
}

MajoranaMapping MajoranaMapping::from_table(std::vector<SparsePauliWord> table,
                                            std::string name) {
  if (table.empty()) {
    throw std::invalid_argument("MajoranaMapping table must not be empty");
  }
  if (table.size() % 2 != 0) {
    throw std::invalid_argument(
        "MajoranaMapping table must have an even number of entries "
        "(2 per fermionic mode), got " +
        std::to_string(table.size()));
  }
  auto num_modes = table.size() / 2;
  auto num_qubits = compute_num_qubits(table);

  // Precompute all bilinears from the table.
  const std::size_t M = table.size();
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears;
  bilinears.reserve(M * (M - 1) / 2);
  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = j + 1; k < M; ++k) {
      auto [phase, word] =
          PauliTermAccumulator::multiply_uncached(table[j], table[k]);
      bilinears.emplace_back(std::complex<double>(0.0, 1.0) * phase,
                             std::move(word));
    }
  }

  auto base_encoding = name;
  return MajoranaMapping(std::move(table), std::move(bilinears),
                         std::move(name), num_modes, num_qubits,
                         std::move(base_encoding));
}

MajoranaMapping MajoranaMapping::from_bilinears(
    std::size_t num_modes,
    std::vector<std::pair<std::complex<double>, SparsePauliWord>>
        upper_triangle,
    std::string name) {
  if (num_modes == 0) {
    throw std::invalid_argument(
        "MajoranaMapping::from_bilinears requires num_modes > 0");
  }
  const std::size_t M = 2 * num_modes;
  const std::size_t expected = M * (M - 1) / 2;
  if (upper_triangle.size() != expected) {
    throw std::invalid_argument("MajoranaMapping::from_bilinears: expected " +
                                std::to_string(expected) +
                                " upper-triangle entries for " +
                                std::to_string(num_modes) + " modes, got " +
                                std::to_string(upper_triangle.size()));
  }
  std::uint64_t max_idx = 0;
  bool has_any = false;
  for (const auto& [_, word] : upper_triangle) {
    for (const auto& [qubit, op] : word) {
      (void)op;
      if (!has_any || qubit >= max_idx) {
        max_idx = qubit;
        has_any = true;
      }
    }
  }
  auto nq = has_any ? static_cast<std::size_t>(max_idx + 1) : 0;
  auto base_encoding = name;
  return MajoranaMapping({}, std::move(upper_triangle), std::move(name),
                         num_modes, nq, std::move(base_encoding));
}

const SparsePauliWord& MajoranaMapping::operator()(std::size_t k) const {
  if (!majorana_atomic_) {
    throw std::logic_error(
        "MajoranaMapping::majorana(k) is not available for bilinear-only "
        "mappings; use bilinear(j, k) instead.");
  }
  if (k >= table_.size()) {
    throw std::out_of_range("Majorana index " + std::to_string(k) +
                            " out of range [0, " +
                            std::to_string(table_.size()) + ")");
  }
  return table_[k];
}

std::pair<std::complex<double>, const SparsePauliWord&>
MajoranaMapping::bilinear(std::size_t j, std::size_t k) const {
  const std::size_t M = 2 * num_modes_;
  if (j >= M || k >= M) {
    throw std::out_of_range(
        "MajoranaMapping::bilinear index out of range: requested (" +
        std::to_string(j) + ", " + std::to_string(k) + "), valid range [0, " +
        std::to_string(M) + ")");
  }
  if (j == k) {
    throw std::invalid_argument(
        "MajoranaMapping::bilinear is undefined for j == k (got " +
        std::to_string(j) +
        "); the bilinear i*gamma_j*gamma_k requires distinct indices.");
  }
  if (j < k) {
    const auto& entry = bilinears_[bilinear_index(j, k)];
    return {entry.first, entry.second};
  }
  const auto& entry = bilinears_[bilinear_index(k, j)];
  return {-entry.first, entry.second};
}

MajoranaMapping MajoranaMapping::without_tapering() const {
  return MajoranaMapping(table_, bilinears_, base_encoding_, num_modes_,
                         num_qubits_, base_encoding_);
}

std::string MajoranaMapping::get_summary() const {
  std::ostringstream ss;
  ss << "MajoranaMapping";
  if (!name_.empty()) {
    ss << " '" << name_ << "'";
  }
  ss << "\n  Modes: " << num_modes_ << "\n  Qubits: " << num_qubits_;
  if (tapering_) {
    ss << "\n  Tapered qubits: " << tapering_->num_tapered();
  }
  return ss.str();
}

nlohmann::json MajoranaMapping::to_json() const {
  nlohmann::json data{{"version", SERIALIZATION_VERSION},
                      {"table", table_},
                      {"name", name_},
                      {"base_encoding", base_encoding_}};
  if (tapering_) {
    data["tapering"] = tapering_->to_json();
  }
  // Bilinear-only mappings: persist the bilinear entries so the mapping can
  // round-trip even when the Majorana table is empty.
  if (!majorana_atomic_) {
    data["num_modes"] = num_modes_;
    nlohmann::json bl_array = nlohmann::json::array();
    for (const auto& [coeff, word] : bilinears_) {
      bl_array.push_back(
          {{"real", coeff.real()}, {"imag", coeff.imag()}, {"word", word}});
    }
    data["bilinears"] = bl_array;
  }
  return data;
}

void MajoranaMapping::to_file(const std::string& filename,
                              const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5" || type == "h5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported format type: " + type);
  }
}

MajoranaMapping MajoranaMapping::from_json(const nlohmann::json& data) {
  auto table = data.at("table").get<std::vector<SparsePauliWord>>();
  std::string name = data.value("name", "");
  std::string base_encoding = data.value("base_encoding", name);
  std::optional<TaperingSpecification> tapering = std::nullopt;
  if (data.contains("tapering") && !data.at("tapering").is_null()) {
    tapering = TaperingSpecification::from_json(data.at("tapering"));
  }

  // Bilinear-only mapping: table is empty, bilinears stored explicitly.
  if (table.empty() && data.contains("bilinears")) {
    auto num_modes = data.at("num_modes").get<std::size_t>();
    std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears;
    for (const auto& entry : data.at("bilinears")) {
      double real = entry.at("real").get<double>();
      double imag = entry.at("imag").get<double>();
      auto word = entry.at("word").get<SparsePauliWord>();
      bilinears.emplace_back(std::complex<double>(real, imag), std::move(word));
    }
    auto mapping = MajoranaMapping::from_bilinears(
        num_modes, std::move(bilinears), base_encoding);
    if (name == base_encoding && !tapering) {
      return mapping;
    }
    return MajoranaMapping(mapping.table_, mapping.bilinears_, std::move(name),
                           mapping.num_modes_, mapping.num_qubits_,
                           std::move(base_encoding), std::move(tapering));
  }

  auto mapping = MajoranaMapping::from_table(std::move(table), base_encoding);
  if (name == base_encoding && !tapering) {
    return mapping;
  }
  return MajoranaMapping(mapping.table_, mapping.bilinears_, std::move(name),
                         mapping.num_modes_, mapping.num_qubits_,
                         std::move(base_encoding), std::move(tapering));
}

void MajoranaMapping::to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for writing: " + filename);
  }
  file << to_json().dump(2);
}

MajoranaMapping MajoranaMapping::from_json_file(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for reading: " + filename);
  }
  nlohmann::json data;
  file >> data;
  return from_json(data);
}

void MajoranaMapping::to_hdf5(H5::Group& group) const {
  const std::string json = to_json().dump();
  group.createAttribute("json", H5::StrType(0, H5T_VARIABLE), H5::DataSpace())
      .write(H5::StrType(0, H5T_VARIABLE), json);
}

MajoranaMapping MajoranaMapping::from_hdf5(H5::Group& group) {
  std::string json;
  group.openAttribute("json").read(H5::StrType(0, H5T_VARIABLE), json);
  return from_json(nlohmann::json::parse(json));
}

void MajoranaMapping::to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  H5::Group root = file.openGroup("/");
  to_hdf5(root);
}

MajoranaMapping MajoranaMapping::from_hdf5_file(const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  H5::Group root = file.openGroup("/");
  return from_hdf5(root);
}

MajoranaMapping MajoranaMapping::from_file(const std::string& filename,
                                           const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  }
  if (type == "hdf5" || type == "h5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported format type: " + type);
}

std::size_t MajoranaMapping::compute_num_qubits(
    const std::vector<SparsePauliWord>& table) {
  std::uint64_t max_idx = 0;
  bool has_any = false;
  for (const auto& word : table) {
    for (const auto& [qubit, _] : word) {
      if (!has_any || qubit >= max_idx) {
        max_idx = qubit;
        has_any = true;
      }
    }
  }
  return has_any ? static_cast<std::size_t>(max_idx + 1) : 0;
}

}  // namespace qdk::chemistry::data
