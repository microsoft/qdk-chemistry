// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

namespace {

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
                                 std::string name)
    : table_(std::move(table)),
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
}

const SparsePauliWord& MajoranaMapping::operator()(std::size_t k) const {
  if (k >= table_.size()) {
    throw std::out_of_range("Majorana index " + std::to_string(k) +
                            " out of range [0, " +
                            std::to_string(table_.size()) + ")");
  }
  return table_[k];
}

std::pair<std::complex<double>, SparsePauliWord> MajoranaMapping::bilinear(
    std::size_t j, std::size_t k) const {
  const std::size_t n = table_.size();
  if (j >= n || k >= n) {
    throw std::out_of_range(
        "MajoranaMapping::bilinear index out of range: requested (" +
        std::to_string(j) + ", " + std::to_string(k) + "), valid range [0, " +
        std::to_string(n) + ")");
  }
  if (j == k) {
    throw std::invalid_argument(
        "MajoranaMapping::bilinear is undefined for j == k (got " +
        std::to_string(j) +
        "); the bilinear i*gamma_j*gamma_k requires distinct indices.");
  }
  // i * gamma_j * gamma_k = i * (table_[j] * table_[k])
  auto [pauli_phase, word] =
      PauliTermAccumulator::multiply_uncached(table_[j], table_[k]);
  std::complex<double> coeff = std::complex<double>(0.0, 1.0) * pauli_phase;
  return {coeff, std::move(word)};
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

// ── Factory: Bravyi-Kitaev tree ──────────────────────────────────────

MajoranaMapping MajoranaMapping::bravyi_kitaev_tree(std::size_t num_modes) {
  if (num_modes == 0) {
    throw std::invalid_argument("bravyi_kitaev_tree requires num_modes > 0");
  }

  // Build the balanced binary tree (Algorithm 1 from arXiv:1701.07072).
  // parent[j] = parent index of node j (-1 for root).
  // children[j] = list of child indices of node j.
  std::vector<int64_t> parent(num_modes, -1);
  std::vector<std::vector<std::size_t>> children(num_modes);

  // Recursive tree builder: range [left, right), pivot = midpoint.
  // Left half goes under pivot; right half goes under parent_idx.
  auto build_tree = [&](auto& self, std::size_t left, std::size_t right,
                        std::size_t parent_idx) -> void {
    if (left >= right) return;
    std::size_t pivot = (left + right) >> 1;
    parent[pivot] = static_cast<int64_t>(parent_idx);
    children[parent_idx].push_back(pivot);
    self(self, left, pivot, pivot);              // left subtree under pivot
    self(self, pivot + 1, right, parent_idx);    // right subtree under parent
  };

  // Root is node (num_modes - 1). Build tree on [0, num_modes - 1).
  if (num_modes > 1) {
    build_tree(build_tree, 0, num_modes - 1, num_modes - 1);
  }

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    // U(j) = ancestors of j (walk up parent chain, excluding j itself)
    std::vector<std::uint64_t> update;
    for (int64_t p = parent[j]; p >= 0; p = parent[static_cast<std::size_t>(p)]) {
      update.push_back(static_cast<std::uint64_t>(p));
    }

    // F(j) = children of j
    // C(j) = for each ancestor of j, children of that ancestor with index < j
    std::vector<std::uint64_t> child_set;
    for (auto c : children[j]) {
      child_set.push_back(static_cast<std::uint64_t>(c));
    }
    std::vector<std::uint64_t> remainder;
    for (int64_t p = parent[j]; p >= 0; p = parent[static_cast<std::size_t>(p)]) {
      for (auto c : children[static_cast<std::size_t>(p)]) {
        if (c < j) {
          remainder.push_back(static_cast<std::uint64_t>(c));
        }
      }
    }

    // P(j) = C(j) ∪ F(j)
    std::vector<std::uint64_t> parity_set = remainder;
    parity_set.insert(parity_set.end(), child_set.begin(), child_set.end());

    // γ_{2j} = X_j · Z_{P(j)} · X_{U(j)}
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), OP_X);
      for (auto q : parity_set) {
        entries.emplace_back(q, OP_Z);
      }
      for (auto q : update) {
        entries.emplace_back(q, OP_X);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }

    // γ_{2j+1} = Y_j · Z_{C(j)} · X_{U(j)}
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

  return MajoranaMapping(std::move(table), "bravyi-kitaev-tree");
}

// ── Factory: Parity ──────────────────────────────────────────────────

MajoranaMapping MajoranaMapping::parity(std::size_t num_modes) {
  if (num_modes == 0) {
    throw std::invalid_argument("parity requires num_modes > 0");
  }

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    // γ_{2j} = Z_{j-1} (if j>0) · X_j · X_{j+1} · ... · X_{n-1}
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      if (j > 0) {
        entries.emplace_back(static_cast<std::uint64_t>(j - 1), OP_Z);
      }
      for (std::size_t k = j; k < num_modes; ++k) {
        entries.emplace_back(static_cast<std::uint64_t>(k), OP_X);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }

    // γ_{2j+1} = Y_j · X_{j+1} · ... · X_{n-1}
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), OP_Y);
      for (std::size_t k = j + 1; k < num_modes; ++k) {
        entries.emplace_back(static_cast<std::uint64_t>(k), OP_X);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }
  }

  return MajoranaMapping(std::move(table), "parity");
}

}  // namespace qdk::chemistry::data
