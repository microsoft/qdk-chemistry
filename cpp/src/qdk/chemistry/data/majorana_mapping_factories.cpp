// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <qdk/chemistry/data/lattice_graph.hpp>
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

// ── Factory: Verstraete-Cirac ──────────────────────────────────────────

MajoranaMapping MajoranaMapping::verstraete_cirac(const LatticeGraph& lattice) {
  using namespace detail;
  const std::string name = "verstraete-cirac";

  std::uint64_t V = lattice.num_sites();
  if (V < 3) {
    throw std::invalid_argument(
        name + " requires a lattice graph with at least 3 sites");
  }

  // Since the lattice graph is pre-ordered optimally (if requested) or
  // processed in default ordering, the node sequence/path is simply 0, 1, ...,
  // V-1. Identify all non-adjacent edges incident to each vertex.
  const auto& adj = lattice.sparse_adjacency_matrix();
  std::vector<std::vector<std::uint64_t>> non_adjacent_incident(V);
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      std::uint64_t u = it.row();
      std::uint64_t v = it.col();
      // Since the matrix is symmetric, only process the upper-triangle (u < v)
      if (u < v) {
        std::size_t diff = v - u;
        if (diff > 1) {
          non_adjacent_incident[u].push_back(v);
          non_adjacent_incident[v].push_back(u);
        }
      }
    }
  }

  // Count aux fermionic modes per site
  std::vector<std::size_t> n_aux(V);
  std::size_t total_n_aux = 0;
  for (std::uint64_t u = 0; u < V; ++u) {
    n_aux[u] = (non_adjacent_incident[u].size() + 1) / 2;
    total_n_aux += n_aux[u];
  }

  std::size_t modes_per_spin = V + total_n_aux;
  std::size_t num_modes = 2 * V;                 // 2 spin species
  std::size_t base_qubits = 2 * modes_per_spin;  // total qubits in jw_base
  std::size_t base_modes = 2 * base_qubits;      // total Majoranas in jw_base

  auto jw_base = MajoranaMapping::jordan_wigner(base_qubits);

  // Precompute mode offsets along the sequence of sites
  std::vector<std::size_t> site_to_mode_offset(V);
  site_to_mode_offset[0] = 0;
  for (std::size_t k = 1; k < V; ++k) {
    std::uint64_t prev_site = k - 1;
    site_to_mode_offset[k] = site_to_mode_offset[k - 1] + 1 + n_aux[prev_site];
  }

  auto get_sys_majorana = [&](std::uint64_t site, std::size_t spin,
                              std::size_t offset) -> std::size_t {
    std::size_t mode_idx = spin * modes_per_spin + site_to_mode_offset[site];
    return 2 * mode_idx + offset;
  };

  auto get_aux_majorana = [&](std::uint64_t site, std::size_t spin,
                              std::size_t offset) -> std::size_t {
    std::size_t mode_idx = spin * modes_per_spin + site_to_mode_offset[site];
    return 2 * (mode_idx + 1) + offset;
  };

  // Get a Pauli representation of the antisymmetric Majorana bilinear iγ_pγ_q
  auto get_bilinear =
      [&](std::size_t p,
          std::size_t q) -> std::pair<std::complex<double>, SparsePauliWord> {
    if (p < q) {
      auto b = jw_base.bilinear(p, q);
      return {b.first, b.second};
    } else {
      auto b = jw_base.bilinear(q, p);
      return {-b.first, b.second};
    }
  };

  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers;
  std::vector<std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>>
      edge_to_stab_idx(2);

  for (std::size_t spin = 0; spin < 2; ++spin) {
    // Add 2-body link stabilizers for non-adjacent edges on the lattice
    for (std::uint64_t u = 0; u < V; ++u) {
      for (std::uint64_t v : non_adjacent_incident[u]) {
        if (u < v) {
          auto it_u = std::find(non_adjacent_incident[u].begin(),
                                non_adjacent_incident[u].end(), v);
          auto it_v = std::find(non_adjacent_incident[v].begin(),
                                non_adjacent_incident[v].end(), u);
          std::size_t a = std::distance(non_adjacent_incident[u].begin(), it_u);
          std::size_t b = std::distance(non_adjacent_incident[v].begin(), it_v);

          std::size_t p = get_aux_majorana(u, spin, a);
          std::size_t q = get_aux_majorana(v, spin, b);

          auto [coeff, word] = get_bilinear(p, q);
          edge_to_stab_idx[spin][{u, v}] = stabilizers.size();
          edge_to_stab_idx[spin][{v, u}] = stabilizers.size();
          stabilizers.emplace_back(coeff, std::move(word));
        }
      }
    }

    // Identify and pair up any remaining unused aux Majorana modes to
    // lift boundary/corner codespace degeneracy
    std::vector<std::size_t> unpaired_modes;
    for (std::uint64_t u = 0; u < V; ++u) {
      std::size_t A_u = non_adjacent_incident[u].size();
      if (A_u % 2 != 0) {
        unpaired_modes.push_back(get_aux_majorana(u, spin, A_u));
      }
    }

    // Unused aux Majorana modes become trivial stabilizers
    // (we are guaranteed an even number due to the handshaking lemma)
    for (std::size_t i = 0; i < unpaired_modes.size(); i += 2) {
      if (i + 1 < unpaired_modes.size()) {
        auto [coeff, word] =
            get_bilinear(unpaired_modes[i], unpaired_modes[i + 1]);
        stabilizers.emplace_back(coeff, std::move(word));
      }
    }
  }

  // Build a lookup table of VC bilinears by dressing non-local JW bilinears
  // with their stabilizer to make everything local
  std::size_t num_physical_modes = 2 * V;
  std::size_t num_physical_majoranas = 2 * num_physical_modes;
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> upper_triangle;
  upper_triangle.reserve(num_physical_majoranas * (num_physical_majoranas - 1) /
                         2);

  for (std::size_t u = 0; u < num_physical_majoranas; ++u) {
    for (std::size_t v = u + 1; v < num_physical_majoranas; ++v) {
      std::size_t u_mode = u / 2;
      std::size_t v_mode = v / 2;
      std::size_t s_u = u_mode / V;
      std::size_t s_v = v_mode / V;
      std::size_t i = u_mode % V;
      std::size_t j = v_mode % V;
      std::size_t a = u % 2;
      std::size_t b = v % 2;

      bool connected = (s_u == s_v) && lattice.are_connected(i, j);

      if (connected) {
        std::size_t path_min = std::min(i, j);
        std::size_t path_max = std::max(i, j);

        std::size_t p = get_sys_majorana(i, s_u, a);
        std::size_t q = get_sys_majorana(j, s_v, b);
        auto [coeff, word] = get_bilinear(p, q);

        // For non-local paths, the raw JW bilinear iγ_pγ_q loses its long
        // Z-string by multiplying by edge stabilizer iγ̃_aγ̃_b
        if (path_max - path_min > 1) {
          auto key = std::make_pair(std::min(i, j), std::max(i, j));
          std::size_t stab_idx = edge_to_stab_idx[s_u].at(key);
          const auto& [b_coeff, b_word] = stabilizers[stab_idx];

          auto [phase, new_word] =
              PauliTermAccumulator::multiply_uncached(word, b_word);
          coeff *= b_coeff * phase;
          word = std::move(new_word);
        }

        upper_triangle.emplace_back(coeff, std::move(word));
      } else {
        std::size_t p = get_sys_majorana(i, s_u, a);
        std::size_t q = get_sys_majorana(j, s_v, b);
        auto [coeff, word] = get_bilinear(p, q);
        upper_triangle.emplace_back(coeff, std::move(word));
      }
    }
  }

  return MajoranaMapping(std::vector<SparsePauliWord>{},
                         std::move(upper_triangle), name, num_physical_modes,
                         base_qubits, name, std::nullopt,
                         std::move(stabilizers));
}

}  // namespace qdk::chemistry::data
