// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

namespace {

// Re-use the Pauli algebra from pauli_operator.cpp (it's in the detail
// namespace of this translation unit's sibling, but we can call the
// PauliTermAccumulator static helper which is public).

/// Multiply two SparsePauliWords via the public uncached static method.
std::pair<std::complex<double>, SparsePauliWord> multiply_words(
    const SparsePauliWord& a, const SparsePauliWord& b) {
  return PauliTermAccumulator::multiply_uncached(a, b);
}

// ── BK index-set computation ──────────────────────────────────────────

/// Smallest power of 2 ≥ n.
std::size_t next_power_of_two(std::size_t n) {
  if (n == 0) return 1;
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

/// Parity set P(j) for BK encoding.
/// P(j) contains qubit indices whose cumulative parity encodes the
/// occupation of all orbitals with index < j.
std::vector<std::uint64_t> bk_parity_set(std::uint64_t j, std::size_t n) {
  // n must be a power of 2
  if (n <= 1) return {};
  std::size_t half = n / 2;
  if (j < half) {
    return bk_parity_set(j, half);
  }
  auto sub = bk_parity_set(j - half, half);
  std::vector<std::uint64_t> result;
  result.reserve(sub.size() + 1);
  result.push_back(static_cast<std::uint64_t>(half - 1));
  for (auto idx : sub) {
    result.push_back(idx + static_cast<std::uint64_t>(half));
  }
  return result;
}

/// Update (ancestor) set U(j) for BK encoding.
/// U(j) contains qubit indices whose stored parity must be flipped when
/// orbital j is occupied.
std::vector<std::uint64_t> bk_update_set(std::uint64_t j, std::size_t n) {
  if (n <= 1) return {};
  std::size_t half = n / 2;
  if (j < half) {
    auto sub = bk_update_set(j, half);
    sub.push_back(static_cast<std::uint64_t>(n - 1));
    return sub;
  }
  auto sub = bk_update_set(j - half, half);
  std::vector<std::uint64_t> result;
  result.reserve(sub.size());
  for (auto idx : sub) {
    result.push_back(idx + static_cast<std::uint64_t>(half));
  }
  return result;
}

/// Flip (children) set F(j) for BK encoding.
std::vector<std::uint64_t> bk_flip_set(std::uint64_t j, std::size_t n) {
  if (n <= 1) return {};
  std::size_t half = n / 2;
  if (j < half) {
    return bk_flip_set(j, half);
  }
  if (j < static_cast<std::uint64_t>(n - 1)) {
    auto sub = bk_flip_set(j - half, half);
    std::vector<std::uint64_t> result;
    result.reserve(sub.size());
    for (auto idx : sub) {
      result.push_back(idx + static_cast<std::uint64_t>(half));
    }
    return result;
  }
  // j == n-1
  auto sub = bk_flip_set(j - half, half);
  std::vector<std::uint64_t> result;
  result.reserve(sub.size() + 1);
  result.push_back(static_cast<std::uint64_t>(half - 1));
  for (auto idx : sub) {
    result.push_back(idx + static_cast<std::uint64_t>(half));
  }
  return result;
}

/// Remainder set R(j) = P(j) \ F(j) for BK encoding.
std::vector<std::uint64_t> bk_remainder_set(std::uint64_t j, std::size_t n) {
  auto parity = bk_parity_set(j, n);
  auto flip = bk_flip_set(j, n);

  // Convert flip to a set for O(1) lookup
  std::vector<bool> in_flip(n, false);
  for (auto idx : flip) {
    if (idx < n) in_flip[idx] = true;
  }

  std::vector<std::uint64_t> result;
  for (auto idx : parity) {
    if (idx >= n || !in_flip[idx]) {
      result.push_back(idx);
    }
  }
  return result;
}

/// Build a SparsePauliWord from qubit assignments. The word is sorted by
/// qubit index (required invariant for SparsePauliWord).
SparsePauliWord build_sorted_word(
    std::vector<std::pair<std::uint64_t, std::uint8_t>> entries) {
  std::sort(entries.begin(), entries.end());
  return entries;
}

// Pauli operator type constants (matching pauli_operator.hpp convention)
constexpr std::uint8_t OP_X = 1;
constexpr std::uint8_t OP_Y = 2;
constexpr std::uint8_t OP_Z = 3;

}  // anonymous namespace

// ── MajoranaMapping implementation ───────────────────────────────────

MajoranaMapping::MajoranaMapping(std::vector<SparsePauliWord> table,
                                 std::string name,
                                 std::vector<std::int8_t> phases)
    : table_(std::move(table)),
      phases_(std::move(phases)),
      all_positive_(true),
      name_(std::move(name)),
      num_qubits_(compute_num_qubits(table_)) {
  if (table_.empty()) {
    throw std::invalid_argument("MajoranaMapping table must not be empty");
  }
  if (table_.size() % 2 != 0) {
    throw std::invalid_argument(
        "MajoranaMapping table must have an even number of entries "
        "(2 per fermionic mode), got " +
        std::to_string(table_.size()));
  }
  // Default phases to all +1 if not provided
  if (phases_.empty()) {
    phases_.assign(table_.size(), 1);
  } else if (phases_.size() != table_.size()) {
    throw std::invalid_argument(
        "phases size (" + std::to_string(phases_.size()) +
        ") must match table size (" + std::to_string(table_.size()) + ")");
  }
  // Validate phase values and compute all_positive flag
  for (std::size_t k = 0; k < phases_.size(); ++k) {
    if (phases_[k] != 1 && phases_[k] != -1) {
      throw std::invalid_argument("phases[" + std::to_string(k) +
                                  "] = " + std::to_string(phases_[k]) +
                                  "; must be +1 or -1");
    }
    if (phases_[k] != 1) all_positive_ = false;
  }
  validate();
}

const SparsePauliWord& MajoranaMapping::operator()(std::size_t k) const {
  if (k >= table_.size()) {
    throw std::out_of_range("Majorana index " + std::to_string(k) +
                            " out of range [0, " +
                            std::to_string(table_.size()) + ")");
  }
  return table_[k];
}

std::int8_t MajoranaMapping::phase(std::size_t k) const {
  if (k >= phases_.size()) {
    throw std::out_of_range("Majorana index " + std::to_string(k) +
                            " out of range [0, " +
                            std::to_string(phases_.size()) + ")");
  }
  return phases_[k];
}

void MajoranaMapping::validate() const {
  const std::size_t n = table_.size();
  constexpr double tol = 1e-12;

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      // gamma_k = phases_[k] * table_[k], so:
      // gamma_i * gamma_j = phases_[i]*phases_[j] * table_[i]*table_[j]
      auto [phase_ij, word_ij] = multiply_words(table_[i], table_[j]);
      auto [phase_ji, word_ji] = multiply_words(table_[j], table_[i]);

      // Include the sign factors
      double sign_ij = static_cast<double>(phases_[i]) * phases_[j];

      if (i == j) {
        // gamma_i^2 = phases_[i]^2 * P_i^2 = 1 * P_i^2
        // P_i^2 must be I with phase +1
        if (!word_ij.empty()) {
          std::ostringstream msg;
          msg << "Clifford algebra validation failed: gamma_" << i
              << " squared is not proportional to identity";
          throw std::invalid_argument(msg.str());
        }
        // phase_ij should be +1 (Pauli string squares to +I only for I, XX,
        // YY, ZZ on a single qubit gives -I for Y but the overall product
        // should be +I for a valid Majorana operator)
        if (std::abs(phase_ij - std::complex<double>(1.0, 0.0)) > tol) {
          std::ostringstream msg;
          msg << "Clifford algebra validation failed: γ_" << i
              << " squared gives phase " << phase_ij << " (expected +1)";
          throw std::invalid_argument(msg.str());
        }
      } else {
        // γ_i·γ_j + γ_j·γ_i = 0, so the two products must cancel.
        // If words are different, both must have zero coefficient.
        // If words are the same, phases must sum to zero.
        if (word_ij == word_ji) {
          auto sum = phase_ij + phase_ji;
          if (std::abs(sum) > tol) {
            std::ostringstream msg;
            msg << "Clifford algebra validation failed: {γ_" << i << ", γ_" << j
                << "} != 0 (anticommutator phase sum = " << sum << ")";
            throw std::invalid_argument(msg.str());
          }
        } else {
          // Different resulting words — each must be separately zero,
          // which means either the mapping is invalid or the words cancel
          // in a sum expression. For valid Majorana mappings from Clifford
          // algebra homomorphisms, the products always yield the same word
          // (possibly with different phase). If words differ, that's an error.
          if (std::abs(phase_ij) > tol || std::abs(phase_ji) > tol) {
            std::ostringstream msg;
            msg << "Clifford algebra validation failed: {γ_" << i << ", γ_" << j
                << "} produces non-cancelling terms with different Pauli words";
            throw std::invalid_argument(msg.str());
          }
        }
      }
    }
  }
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

// ── Factory: Jordan-Wigner ───────────────────────────────────────────

MajoranaMapping MajoranaMapping::jordan_wigner(std::size_t num_modes) {
  if (num_modes == 0) {
    throw std::invalid_argument("jordan_wigner requires num_modes > 0");
  }

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    // γ_{2j} = Z_{j-1} ... Z_0 ⊗ X_j
    std::vector<std::pair<std::uint64_t, std::uint8_t>> even_entries;
    for (std::size_t k = 0; k < j; ++k) {
      even_entries.emplace_back(static_cast<std::uint64_t>(k), OP_Z);
    }
    even_entries.emplace_back(static_cast<std::uint64_t>(j), OP_X);
    table.push_back(build_sorted_word(std::move(even_entries)));

    // γ_{2j+1} = Z_{j-1} ... Z_0 ⊗ Y_j
    std::vector<std::pair<std::uint64_t, std::uint8_t>> odd_entries;
    for (std::size_t k = 0; k < j; ++k) {
      odd_entries.emplace_back(static_cast<std::uint64_t>(k), OP_Z);
    }
    odd_entries.emplace_back(static_cast<std::uint64_t>(j), OP_Y);
    table.push_back(build_sorted_word(std::move(odd_entries)));
  }

  return MajoranaMapping(std::move(table), "jordan-wigner");
}

// ── Factory: Bravyi-Kitaev ───────────────────────────────────────────

MajoranaMapping MajoranaMapping::bravyi_kitaev(std::size_t num_modes) {
  if (num_modes == 0) {
    throw std::invalid_argument("bravyi_kitaev requires num_modes > 0");
  }

  // BK index sets are defined on a binary tree of size 2^ceil(log2(n))
  std::size_t tree_size = next_power_of_two(num_modes);

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    auto parity = bk_parity_set(static_cast<std::uint64_t>(j), tree_size);
    auto update = bk_update_set(static_cast<std::uint64_t>(j), tree_size);
    auto remainder = bk_remainder_set(static_cast<std::uint64_t>(j), tree_size);

    // Filter out indices ≥ num_modes (virtual tree nodes beyond actual qubits)
    auto filter = [num_modes](std::vector<std::uint64_t>& v) {
      v.erase(std::remove_if(
                  v.begin(), v.end(),
                  [num_modes](std::uint64_t idx) { return idx >= num_modes; }),
              v.end());
    };
    filter(parity);
    filter(update);
    filter(remainder);

    // γ_{2j} = X_{U(j)} ⊗ X_j ⊗ Z_{P(j)}
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), OP_X);
      for (auto q : parity) {
        entries.emplace_back(q, OP_Z);
      }
      for (auto q : update) {
        entries.emplace_back(q, OP_X);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }

    // γ_{2j+1} = X_{U(j)} ⊗ Y_j ⊗ Z_{R(j)}
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), OP_Y);
      for (auto q : remainder) {
        entries.emplace_back(q, OP_Z);
      }
      for (auto q : update) {
        entries.emplace_back(q, OP_X);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }
  }

  return MajoranaMapping(std::move(table), "bravyi-kitaev");
}

// ── Factory: Parity ──────────────────────────────────────────────────

MajoranaMapping MajoranaMapping::parity(std::size_t num_modes) {
  if (num_modes == 0) {
    throw std::invalid_argument("parity requires num_modes > 0");
  }

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    // γ_{2j}: X_j, X_{j+1} (if j < n-1), Z on {j-1, j-3, j-5, ...}
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      // Z on alternating qubits below j (step -2 from j-1)
      for (int64_t k = static_cast<int64_t>(j) - 1; k >= 0; k -= 2) {
        entries.emplace_back(static_cast<std::uint64_t>(k), OP_Z);
      }
      entries.emplace_back(static_cast<std::uint64_t>(j), OP_X);
      if (j < num_modes - 1) {
        entries.emplace_back(static_cast<std::uint64_t>(j + 1), OP_X);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }

    // γ_{2j+1}: Y_j, X_{j+1} (if j < n-1), Z on {j-2, j-4, j-6, ...}
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      // Z on alternating qubits below j (step -2 from j-2)
      for (int64_t k = static_cast<int64_t>(j) - 2; k >= 0; k -= 2) {
        entries.emplace_back(static_cast<std::uint64_t>(k), OP_Z);
      }
      entries.emplace_back(static_cast<std::uint64_t>(j), OP_Y);
      if (j < num_modes - 1) {
        entries.emplace_back(static_cast<std::uint64_t>(j + 1), OP_X);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }
  }

  return MajoranaMapping(std::move(table), "parity");
}

}  // namespace qdk::chemistry::data
