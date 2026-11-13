// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
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
namespace detail {

nlohmann::json stabilizers_to_json(
    const std::vector<std::pair<std::complex<double>, SparsePauliWord>>&
        stabilizers) {
  nlohmann::json stab_array = nlohmann::json::array();
  for (const auto& [coeff, word] : stabilizers) {
    stab_array.push_back(
        {{"real", coeff.real()}, {"imag", coeff.imag()}, {"word", word}});
  }
  return stab_array;
}

std::vector<std::pair<std::complex<double>, SparsePauliWord>>
stabilizers_from_json(const nlohmann::json& data) {
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers;
  if (data.is_array()) {
    for (const auto& entry : data) {
      double real = entry.at("real").get<double>();
      double imag = entry.at("imag").get<double>();
      auto word = entry.at("word").get<SparsePauliWord>();
      stabilizers.emplace_back(std::complex<double>(real, imag),
                               std::move(word));
    }
  }
  return stabilizers;
}

}  // namespace detail

static void hash_sparse_pauli_word(qdk::chemistry::utils::HashContext& ctx,
                                   const SparsePauliWord& word) {
  hash_value(ctx, static_cast<std::uint64_t>(word.size()));
  for (const auto& [qubit, op] : word) {
    hash_value(ctx, static_cast<std::uint64_t>(qubit));
    hash_value(ctx, op);
  }
}

static void hash_bilinear_entry(
    qdk::chemistry::utils::HashContext& ctx,
    const std::pair<std::complex<double>, SparsePauliWord>& entry) {
  hash_value(ctx, entry.first);
  hash_sparse_pauli_word(ctx, entry.second);
}

// ── MajoranaMapping implementation ───────────────────────────────────

MajoranaMapping::MajoranaMapping(
    std::vector<std::vector<detail::PauliRun>> runs,
    std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears,
    std::string name, std::size_t num_modes, std::size_t num_qubits,
    std::string base_encoding, std::optional<TaperingSpecification> tapering,
    std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers)
    : runs_(std::move(runs)),
      increasing_(std::all_of(runs_.begin(), runs_.end(),
                              [](const auto& word) {
                                for (std::size_t i = 1; i < word.size(); ++i) {
                                  if (word[i].first <
                                      word[i - 1].first + word[i - 1].count) {
                                    return false;
                                  }
                                }
                                return true;
                              })),
      expanded_(std::make_shared<ExpandedTable>()),
      bilinears_(std::move(bilinears)),
      name_(std::move(name)),
      base_encoding_(std::move(base_encoding)),
      num_modes_(num_modes),
      num_qubits_(num_qubits),
      majorana_atomic_(!runs_.empty()),
      tapering_(std::move(tapering)),
      stabilizers_(std::move(stabilizers)) {
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
  auto base_encoding = name;
  // Consecutive qubits carrying the same Pauli share one run.
  std::vector<std::vector<detail::PauliRun>> runs(table.size());
  for (std::size_t k = 0; k < table.size(); ++k) {
    auto& word = runs[k];
    for (const auto& [qubit, op] : table[k]) {
      if (!word.empty() && word.back().op == op &&
          word.back().first + word.back().count == qubit) {
        ++word.back().count;
      } else {
        word.push_back({qubit, 1, op});
      }
    }
  }
  return MajoranaMapping(std::move(runs), {}, std::move(name), num_modes,
                         num_qubits, std::move(base_encoding));
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

const std::vector<SparsePauliWord>& MajoranaMapping::table() const {
  std::call_once(expanded_->once, [this] {
    // Built aside so a throw leaves nothing for a retried call_once to extend.
    std::vector<SparsePauliWord> words(runs_.size());
    for (std::size_t k = 0; k < runs_.size(); ++k) {
      for (const auto& run : runs_[k]) {
        for (std::uint64_t i = 0; i < run.count; ++i) {
          words[k].emplace_back(run.first + i, run.op);
        }
      }
    }
    expanded_->words = std::move(words);
  });
  return expanded_->words;
}

const SparsePauliWord& MajoranaMapping::operator()(std::size_t k) const {
  if (!majorana_atomic_) {
    throw std::logic_error(
        "MajoranaMapping::majorana(k) is not available for bilinear-only "
        "mappings; use bilinear(j, k) instead.");
  }
  if (k >= runs_.size()) {
    throw std::out_of_range("Majorana index " + std::to_string(k) +
                            " out of range [0, " +
                            std::to_string(runs_.size()) + ")");
  }
  return table()[k];
}

std::pair<std::complex<double>, SparsePauliWord> MajoranaMapping::bilinear(
    std::size_t j, std::size_t k) const {
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
  const std::size_t lo = std::min(j, k);
  const std::size_t hi = std::max(j, k);
  std::pair<std::complex<double>, SparsePauliWord> entry;
  if (!majorana_atomic_) {
    entry = bilinears_[bilinear_index(lo, hi)];
  } else if (!increasing_) {
    auto [phase, word] =
        PauliTermAccumulator::multiply_uncached(table()[lo], table()[hi]);
    entry = {std::complex<double>(0.0, 1.0) * phase, std::move(word)};
  } else {
    // multiply_uncached of the expanded entries, merged run by run so equal
    // runs (the shared Jordan-Wigner Z strings) cancel without expanding.
    const auto& left = runs_[lo];
    const auto& right = runs_[hi];
    const std::complex<double> imag_unit(0.0, 1.0);
    std::complex<double> phase(1.0, 0.0);
    SparsePauliWord word;
    auto emit = [&word](std::uint64_t first, std::uint64_t count,
                        std::uint8_t op) {
      for (std::uint64_t i = 0; i < count; ++i)
        word.emplace_back(first + i, op);
    };
    std::size_t a = 0, b = 0;
    // Qubits already consumed from left[a] and right[b].
    std::uint64_t a_used = 0, b_used = 0;
    while (a < left.size() && b < right.size()) {
      const auto& l = left[a];
      const auto& r = right[b];
      const std::uint64_t l_qubit = l.first + a_used;
      const std::uint64_t r_qubit = r.first + b_used;
      if (l_qubit < r_qubit) {
        const std::uint64_t step =
            std::min(l.count - a_used, r_qubit - l_qubit);
        emit(l_qubit, step, l.op);
        a_used += step;
      } else if (r_qubit < l_qubit) {
        const std::uint64_t step =
            std::min(r.count - b_used, l_qubit - r_qubit);
        emit(r_qubit, step, r.op);
        b_used += step;
      } else {
        const std::uint64_t step = std::min(l.count - a_used, r.count - b_used);
        if (l.op == 0) {
          if (r.op != 0) emit(l_qubit, step, r.op);
        } else if (r.op == 0) {
          emit(l_qubit, step, l.op);
        } else if (l.op != r.op) {
          const bool cyclic = (l.op == 1 && r.op == 2) ||
                              (l.op == 2 && r.op == 3) ||
                              (l.op == 3 && r.op == 1);
          for (std::uint64_t i = 0; i < step; ++i) {
            phase *= cyclic ? imag_unit : -imag_unit;
            word.emplace_back(l_qubit + i,
                              static_cast<std::uint8_t>(6 - l.op - r.op));
          }
        }
        a_used += step;
        b_used += step;
      }
      if (a_used == l.count) {
        ++a;
        a_used = 0;
      }
      if (b_used == r.count) {
        ++b;
        b_used = 0;
      }
    }
    for (; a < left.size(); ++a, a_used = 0) {
      emit(left[a].first + a_used, left[a].count - a_used, left[a].op);
    }
    for (; b < right.size(); ++b, b_used = 0) {
      emit(right[b].first + b_used, right[b].count - b_used, right[b].op);
    }
    entry = {imag_unit * phase, std::move(word)};
  }
  if (j > k) entry.first = -entry.first;
  return entry;
}

MajoranaMapping MajoranaMapping::without_tapering() const {
  return MajoranaMapping(runs_, bilinears_, base_encoding_, num_modes_,
                         num_qubits_, base_encoding_, std::nullopt,
                         stabilizers_);
}

std::string MajoranaMapping::get_summary() const {
  std::ostringstream ss;
  ss << "MajoranaMapping";
  if (!name_.empty()) {
    ss << " '" << name_ << "'";
  }
  ss << "\n  Modes: " << num_modes_ << "\n  Qubits: " << num_qubits_;
  if (!stabilizers_.empty()) {
    ss << "\n  Stabilizers: " << stabilizers_.size();
  }
  if (tapering_) {
    ss << "\n  Tapered qubits: " << tapering_->num_tapered();
  }
  return ss.str();
}

nlohmann::json MajoranaMapping::to_json() const {
  nlohmann::json data{{"version", SERIALIZATION_VERSION},
                      {"table", table()},
                      {"name", name_},
                      {"base_encoding", base_encoding_}};
  if (tapering_) {
    data["tapering"] = tapering_->to_json();
  }
  if (!stabilizers_.empty()) {
    data["stabilizers"] = detail::stabilizers_to_json(stabilizers_);
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
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers;
  if (data.contains("stabilizers") && !data.at("stabilizers").is_null()) {
    stabilizers = detail::stabilizers_from_json(data.at("stabilizers"));
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
    if (name == base_encoding && !tapering && stabilizers.empty()) {
      return mapping;
    }
    return MajoranaMapping(mapping.runs_, mapping.bilinears_, std::move(name),
                           mapping.num_modes_, mapping.num_qubits_,
                           std::move(base_encoding), std::move(tapering),
                           std::move(stabilizers));
  }

  auto mapping = MajoranaMapping::from_table(std::move(table), base_encoding);
  if (name == base_encoding && !tapering && stabilizers.empty()) {
    return mapping;
  }
  return MajoranaMapping(mapping.runs_, mapping.bilinears_, std::move(name),
                         mapping.num_modes_, mapping.num_qubits_,
                         std::move(base_encoding), std::move(tapering),
                         std::move(stabilizers));
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

void MajoranaMapping::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, name_);
  hash_value(ctx, base_encoding_);
  hash_value(ctx, static_cast<std::uint64_t>(num_modes_));
  hash_value(ctx, static_cast<std::uint64_t>(num_qubits_));
  hash_value(ctx, majorana_atomic_);

  // Hashes the expanded table without materializing it.
  hash_value(ctx, static_cast<std::uint64_t>(runs_.size()));
  for (const auto& runs : runs_) {
    std::uint64_t size = 0;
    for (const auto& run : runs) size += run.count;
    hash_value(ctx, size);
    for (const auto& run : runs) {
      for (std::uint64_t i = 0; i < run.count; ++i) {
        hash_value(ctx, static_cast<std::uint64_t>(run.first + i));
        hash_value(ctx, run.op);
      }
    }
  }

  hash_value(ctx, static_cast<std::uint64_t>(bilinears_.size()));
  for (const auto& entry : bilinears_) {
    hash_bilinear_entry(ctx, entry);
  }

  if (tapering_) {
    hash_value(ctx, std::uint8_t{1});
    hash_value(ctx, tapering_->content_hash());
  } else {
    hash_value(ctx, std::uint8_t{0});
  }
}

}  // namespace qdk::chemistry::data
