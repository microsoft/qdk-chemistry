// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <complex>
#include <cstdint>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

// ── MajoranaMapping implementation ───────────────────────────────────

MajoranaMapping::MajoranaMapping(
    std::vector<SparsePauliWord> table,
    std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears,
    std::string name, std::size_t num_modes, std::size_t num_qubits)
    : table_(std::move(table)),
      bilinears_(std::move(bilinears)),
      name_(std::move(name)),
      num_modes_(num_modes),
      num_qubits_(num_qubits),
      majorana_atomic_(!table_.empty()) {}

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

  return MajoranaMapping(std::move(table), std::move(bilinears),
                         std::move(name), num_modes, num_qubits);
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
  return MajoranaMapping({}, std::move(upper_triangle), std::move(name),
                         num_modes, nq);
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
