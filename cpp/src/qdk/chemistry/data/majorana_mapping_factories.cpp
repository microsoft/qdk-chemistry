// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cstdint>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/tapering.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {
namespace detail {

std::size_t next_power_of_two(std::size_t n) {
  if (n == 0) return 1;
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

std::vector<std::uint64_t> bk_parity_set(std::uint64_t j, std::size_t n) {
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
  auto sub = bk_flip_set(j - half, half);
  std::vector<std::uint64_t> result;
  result.reserve(sub.size() + 1);
  result.push_back(static_cast<std::uint64_t>(half - 1));
  for (auto idx : sub) {
    result.push_back(idx + static_cast<std::uint64_t>(half));
  }
  return result;
}

std::vector<std::uint64_t> bk_remainder_set(std::uint64_t j, std::size_t n) {
  auto parity = bk_parity_set(j, n);
  auto flip = bk_flip_set(j, n);

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

SparsePauliWord build_sorted_word(
    std::vector<std::pair<std::uint64_t, std::uint8_t>> entries) {
  std::sort(entries.begin(), entries.end());
  return entries;
}

constexpr std::uint8_t op_x = 1;
constexpr std::uint8_t op_y = 2;
constexpr std::uint8_t op_z = 3;

}  // namespace detail

// ── Factory: Jordan-Wigner ───────────────────────────────────────────

MajoranaMapping MajoranaMapping::jordan_wigner(std::size_t num_modes) {
  using namespace detail;
  if (num_modes == 0) {
    throw std::invalid_argument("jordan_wigner requires num_modes > 0");
  }

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    std::vector<std::pair<std::uint64_t, std::uint8_t>> even_entries;
    for (std::size_t k = 0; k < j; ++k) {
      even_entries.emplace_back(static_cast<std::uint64_t>(k), op_z);
    }
    even_entries.emplace_back(static_cast<std::uint64_t>(j), op_x);
    table.push_back(build_sorted_word(std::move(even_entries)));

    std::vector<std::pair<std::uint64_t, std::uint8_t>> odd_entries;
    for (std::size_t k = 0; k < j; ++k) {
      odd_entries.emplace_back(static_cast<std::uint64_t>(k), op_z);
    }
    odd_entries.emplace_back(static_cast<std::uint64_t>(j), op_y);
    table.push_back(build_sorted_word(std::move(odd_entries)));
  }

  return MajoranaMapping::from_table(std::move(table), "jordan-wigner");
}

// ── Factory: Bravyi-Kitaev ───────────────────────────────────────────

MajoranaMapping MajoranaMapping::bravyi_kitaev(std::size_t num_modes) {
  using namespace detail;
  if (num_modes == 0) {
    throw std::invalid_argument("bravyi_kitaev requires num_modes > 0");
  }

  std::size_t tree_size = next_power_of_two(num_modes);

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    auto parity = bk_parity_set(static_cast<std::uint64_t>(j), tree_size);
    auto update = bk_update_set(static_cast<std::uint64_t>(j), tree_size);
    auto remainder = bk_remainder_set(static_cast<std::uint64_t>(j), tree_size);

    auto filter = [num_modes](std::vector<std::uint64_t>& v) {
      v.erase(std::remove_if(
                  v.begin(), v.end(),
                  [num_modes](std::uint64_t idx) { return idx >= num_modes; }),
              v.end());
    };
    filter(parity);
    filter(update);
    filter(remainder);

    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), op_x);
      for (auto q : parity) {
        entries.emplace_back(q, op_z);
      }
      for (auto q : update) {
        entries.emplace_back(q, op_x);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }

    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), op_y);
      for (auto q : remainder) {
        entries.emplace_back(q, op_z);
      }
      for (auto q : update) {
        entries.emplace_back(q, op_x);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }
  }

  return MajoranaMapping::from_table(std::move(table), "bravyi-kitaev");
}

// ── Factory: Bravyi-Kitaev tree ──────────────────────────────────────

MajoranaMapping MajoranaMapping::bravyi_kitaev_tree(std::size_t num_modes) {
  using namespace detail;
  if (num_modes == 0) {
    throw std::invalid_argument("bravyi_kitaev_tree requires num_modes > 0");
  }

  std::vector<int64_t> parent(num_modes, -1);
  std::vector<std::vector<std::size_t>> children(num_modes);

  auto build_tree = [&](auto& self, std::size_t left, std::size_t right,
                        std::size_t parent_idx) -> void {
    if (left >= right) return;
    std::size_t pivot = (left + right) >> 1;
    parent[pivot] = static_cast<int64_t>(parent_idx);
    children[parent_idx].push_back(pivot);
    self(self, left, pivot, pivot);
    self(self, pivot + 1, right, parent_idx);
  };

  if (num_modes > 1) {
    build_tree(build_tree, 0, num_modes - 1, num_modes - 1);
  }

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    std::vector<std::uint64_t> update;
    for (int64_t p = parent[j]; p >= 0;
         p = parent[static_cast<std::size_t>(p)]) {
      update.push_back(static_cast<std::uint64_t>(p));
    }

    std::vector<std::uint64_t> child_set;
    for (auto c : children[j]) {
      child_set.push_back(static_cast<std::uint64_t>(c));
    }
    std::vector<std::uint64_t> remainder;
    for (int64_t p = parent[j]; p >= 0;
         p = parent[static_cast<std::size_t>(p)]) {
      for (auto c : children[static_cast<std::size_t>(p)]) {
        if (c < j) {
          remainder.push_back(static_cast<std::uint64_t>(c));
        }
      }
    }

    std::vector<std::uint64_t> parity_set = remainder;
    parity_set.insert(parity_set.end(), child_set.begin(), child_set.end());

    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), op_x);
      for (auto q : parity_set) {
        entries.emplace_back(q, op_z);
      }
      for (auto q : update) {
        entries.emplace_back(q, op_x);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }

    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), op_y);
      for (auto q : remainder) {
        entries.emplace_back(q, op_z);
      }
      for (auto q : update) {
        entries.emplace_back(q, op_x);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }
  }

  return MajoranaMapping::from_table(std::move(table), "bravyi-kitaev-tree");
}

// ── Factory: Parity ──────────────────────────────────────────────────

MajoranaMapping MajoranaMapping::parity(std::size_t num_modes) {
  using namespace detail;
  if (num_modes == 0) {
    throw std::invalid_argument("parity requires num_modes > 0");
  }

  std::vector<SparsePauliWord> table;
  table.reserve(2 * num_modes);

  for (std::size_t j = 0; j < num_modes; ++j) {
    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      if (j > 0) {
        entries.emplace_back(static_cast<std::uint64_t>(j - 1), op_z);
      }
      for (std::size_t k = j; k < num_modes; ++k) {
        entries.emplace_back(static_cast<std::uint64_t>(k), op_x);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }

    {
      std::vector<std::pair<std::uint64_t, std::uint8_t>> entries;
      entries.emplace_back(static_cast<std::uint64_t>(j), op_y);
      for (std::size_t k = j + 1; k < num_modes; ++k) {
        entries.emplace_back(static_cast<std::uint64_t>(k), op_x);
      }
      table.push_back(build_sorted_word(std::move(entries)));
    }
  }

  return MajoranaMapping::from_table(std::move(table), "parity");
}

MajoranaMapping MajoranaMapping::parity(std::size_t num_modes,
                                        std::size_t n_alpha,
                                        std::size_t n_beta) {
  auto base = MajoranaMapping::parity(num_modes);
  auto tapering = TaperingSpecification::parity_two_qubit_reduction(
      num_modes, n_alpha, n_beta);
  return MajoranaMapping(base.table_, base.bilinears_, "parity-2q-reduced",
                         base.num_modes_, base.num_qubits_, "parity",
                         std::move(tapering));
}

// ── Factory: Verstraete-Cirac ────────────────────────────────────────

// Verstraete-Cirac encoding for a rows x cols lattice, following
// Verstraete & Cirac (J. Stat. Mech. 2005 P09012) and Whitfield,
// Havlicek & Troyer (PRA 94, 030301(R) 2016).
//
// The mapping covers both spin species in blocked order: sites 0..N-1
// (N = rows*cols, row-major) appear once per spin block.  Each site
// carries one physical fermionic mode and one auxiliary mode.  The
// combined 4N modes are Jordan-Wigner mapped in interleaved order
// (phys_0, aux_0, phys_1, aux_1, ... per spin block) onto 4N qubits.
//
// For every vertical lattice edge (s, t = s + cols) the auxiliary modes
// provide a stabilizer S_e = i c_s d_t, where c_s is the X-type Majorana
// of site s's auxiliary mode and d_t the Y-type Majorana of site t's.
// Each auxiliary Majorana appears in at most one stabilizer, so all S_e
// mutually commute, and they commute with every physical bilinear (both
// operators are even products of disjoint Majoranas).
//
// Each physical bilinear i gamma_P gamma_Q whose Majoranas live on the
// two endpoints of a vertical edge is dressed by that edge's stabilizer:
// the auxiliary Z-string of S_e cancels the physical JW string, leaving
// a constant weight-4 operator.  Horizontal hops are JW-adjacent in the
// interleaved layout (weight 3) and need no dressing.  Since S_e = +1 on
// the codespace, the mapped Hamiltonian restricted to the joint +1
// eigenspace of all stabilizers is isospectral to the Jordan-Wigner
// Hamiltonian on 2N qubits.
//
// The result is bilinear-only (no individual Majorana images exist for
// dressed operators), so this factory uses from_bilinears.

MajoranaMapping MajoranaMapping::verstraete_cirac(std::size_t rows,
                                                  std::size_t cols) {
  if (rows < 2 || cols < 2) {
    throw std::invalid_argument(
        "verstraete_cirac requires rows >= 2 and cols >= 2");
  }

  const std::size_t n_sites = rows * cols;
  const std::size_t n_modes = 2 * n_sites;  // exposed spin-orbital modes
  const std::size_t M = 2 * n_modes;        // physical Majorana count

  // Combined-mode layout (4*n_sites modes -> 4*n_sites qubits):
  //   spin block sigma in {0, 1}, site s:
  //     physical mode at combined index sigma*2N + 2s
  //     auxiliary mode at combined index sigma*2N + 2s + 1
  auto mu = [](std::size_t k) -> SparsePauliWord {
    // Pauli image of combined-JW Majorana k (mode k/2, parity k%2).
    std::size_t m = k / 2;
    SparsePauliWord word;
    word.reserve(m + 1);
    for (std::size_t q = 0; q < m; ++q) {
      word.emplace_back(static_cast<std::uint64_t>(q), detail::op_z);
    }
    word.emplace_back(static_cast<std::uint64_t>(m),
                      (k % 2 == 0) ? detail::op_x : detail::op_y);
    return word;
  };

  auto phys_majorana = [&](std::size_t P) -> SparsePauliWord {
    std::size_t sigma = P / (2 * n_sites);
    std::size_t p = P % (2 * n_sites);
    std::size_t s = p / 2;
    std::size_t b = p % 2;
    return mu(2 * (sigma * 2 * n_sites + 2 * s) + b);
  };

  // S_e = i * c_s * d_t for vertical edge (s, t), both in spin block sigma.
  auto stabilizer =
      [&](std::size_t sigma, std::size_t s,
          std::size_t t) -> std::pair<std::complex<double>, SparsePauliWord> {
    auto c = mu(2 * (sigma * 2 * n_sites + 2 * s + 1));
    auto d = mu(2 * (sigma * 2 * n_sites + 2 * t + 1) + 1);
    auto [phase, word] = PauliTermAccumulator::multiply_uncached(c, d);
    return {std::complex<double>(0.0, 1.0) * phase, std::move(word)};
  };

  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears;
  bilinears.reserve(M * (M - 1) / 2);

  for (std::size_t P = 0; P < M; ++P) {
    std::size_t sigma_p = P / (2 * n_sites);
    std::size_t site_p = (P % (2 * n_sites)) / 2;
    auto img_p = phys_majorana(P);
    for (std::size_t Q = P + 1; Q < M; ++Q) {
      std::size_t sigma_q = Q / (2 * n_sites);
      std::size_t site_q = (Q % (2 * n_sites)) / 2;

      auto [phase, word] =
          PauliTermAccumulator::multiply_uncached(img_p, phys_majorana(Q));
      std::complex<double> coeff = std::complex<double>(0.0, 1.0) * phase;

      // Dress vertical-edge bilinears with the edge stabilizer.
      if (sigma_p == sigma_q && site_p != site_q) {
        std::size_t lo = std::min(site_p, site_q);
        std::size_t hi = std::max(site_p, site_q);
        if (hi - lo == cols) {
          auto [s_coeff, s_word] = stabilizer(sigma_p, lo, hi);
          auto [d_phase, d_word] =
              PauliTermAccumulator::multiply_uncached(word, s_word);
          coeff *= s_coeff * d_phase;
          word = std::move(d_word);
        }
      }

      bilinears.emplace_back(coeff, std::move(word));
    }
  }

  return MajoranaMapping::from_bilinears(n_modes, std::move(bilinears),
                                         "verstraete-cirac");
}

MajoranaMapping MajoranaMapping::symmetry_conserving_bravyi_kitaev(
    std::size_t num_modes, std::size_t n_alpha, std::size_t n_beta) {
  auto base = MajoranaMapping::bravyi_kitaev_tree(num_modes);
  auto tapering = TaperingSpecification::symmetry_conserving_bravyi_kitaev(
      num_modes, n_alpha, n_beta);
  return MajoranaMapping(base.table_, base.bilinears_,
                         "symmetry-conserving-bravyi-kitaev", base.num_modes_,
                         base.num_qubits_, "bravyi-kitaev-tree",
                         std::move(tapering));
}

}  // namespace qdk::chemistry::data
