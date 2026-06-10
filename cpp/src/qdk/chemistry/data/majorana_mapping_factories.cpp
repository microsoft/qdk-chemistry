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

bool find_hamiltonian_path_dfs(std::uint64_t curr,
                               const Eigen::SparseMatrix<double>& adj,
                               std::vector<bool>& visited,
                               std::vector<std::uint64_t>& path) {
  path.push_back(curr);
  if (path.size() == static_cast<std::size_t>(adj.rows())) {
    return true;
  }
  visited[curr] = true;

  // Iterate directly over the sparse matrix columns/rows for neighbors
  for (Eigen::SparseMatrix<double>::InnerIterator it(adj, curr); it; ++it) {
    std::uint64_t neighbor = it.row();
    if (neighbor != curr && !visited[neighbor]) {
      if (find_hamiltonian_path_dfs(neighbor, adj, visited, path)) {
        return true;
      }
    }
  }
  visited[curr] = false;
  path.pop_back();
  return false;
}

std::vector<std::uint64_t> find_hamiltonian_path(
    const Eigen::SparseMatrix<double>& adj) {
  std::uint64_t V = adj.rows();
  std::vector<bool> visited(V, false);
  std::vector<std::uint64_t> path;
  for (std::uint64_t start = 0; start < V; ++start) {
    if (find_hamiltonian_path_dfs(start, adj, visited, path)) {
      return path;
    }
  }
  return {};
}

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

MajoranaMapping MajoranaMapping::verstraete_cirac(
    const LatticeGraph& lattice, std::size_t num_spin_species) {
  using namespace detail;
  const std::string name = "verstraete-cirac";

  // Support spinless or fermions with spin-½
  if (num_spin_species != 1 && num_spin_species != 2) {
    throw std::invalid_argument(name +
                                " requires num_spin_species to be 1 or 2.");
  }

  std::uint64_t V = lattice.num_sites();
  if (V < 3) {
    throw std::invalid_argument(
        name + " requires a lattice graph with at least 3 sites");
  }

  // Construct a Hamiltonian path from the graph adjacency matrix
  const auto& adj = lattice.sparse_adjacency_matrix();
  std::vector<std::uint64_t> hamiltonian_path = find_hamiltonian_path(adj);
  if (hamiltonian_path.empty()) {
    throw std::runtime_error(
        name +
        ": No Hamiltonian path found in the lattice graph. A Hamiltonian path "
        "is required to construct commuting stabilizers.");
  }

  // Map row-major site index to its position in the Hamiltonian path
  std::vector<std::uint64_t> site_to_path_idx(V);
  for (std::uint64_t k = 0; k < V; ++k) {
    site_to_path_idx[hamiltonian_path[k]] = k;
  }

  std::size_t num_modes = num_spin_species * V;
  std::size_t base_modes = 2 * num_modes;
  auto jw_base = MajoranaMapping::jordan_wigner(base_modes);

  auto get_sys_majorana = [&](std::uint64_t site, std::size_t spin,
                              std::size_t offset) -> std::size_t {
    std::size_t path_idx = site_to_path_idx[site];
    return 2 * (2 * (spin * V + path_idx)) + offset;
  };

  auto get_aux_majorana = [&](std::uint64_t site, std::size_t spin,
                              std::size_t offset) -> std::size_t {
    std::size_t path_idx = site_to_path_idx[site];
    return 2 * (2 * (spin * V + path_idx) + 1) + offset;
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

  // Find all non-path edges incident to each vertex
  std::vector<std::vector<std::uint64_t>> non_path_incident(V);
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      std::uint64_t u = it.row();
      std::uint64_t v = it.col();
      // Since the matrix is symmetric, only process the upper-triangle
      if (u < v) {
        std::size_t s_u = site_to_path_idx[u];
        std::size_t s_v = site_to_path_idx[v];
        std::size_t diff = (s_u > s_v) ? (s_u - s_v) : (s_v - s_u);
        if (diff > 1) {
          non_path_incident[u].push_back(v);
          non_path_incident[v].push_back(u);
        }
      }
    }
  }

  // Ensure that we have no more than 1 aux Majorana mode per sys mode
  for (std::uint64_t u = 0; u < V; ++u) {
    if (non_path_incident[u].size() > 2) {
      throw std::runtime_error(name + ": Vertex has " +
                               std::to_string(non_path_incident[u].size()) +
                               " non-path edges, which exceeds the "
                               "2-auxiliary-Majorana limit of the mapping.");
    }
  }

  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers;
  std::vector<std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>>
      edge_to_stab_idx(num_spin_species);

  for (std::size_t spin = 0; spin < num_spin_species; ++spin) {
    // Add 2-body link stabilizers for non-path edges on the lattice
    for (std::uint64_t u = 0; u < V; ++u) {
      for (std::uint64_t v : non_path_incident[u]) {
        if (u < v) {
          auto it_u = std::find(non_path_incident[u].begin(),
                                non_path_incident[u].end(), v);
          auto it_v = std::find(non_path_incident[v].begin(),
                                non_path_incident[v].end(), u);
          std::size_t a = std::distance(non_path_incident[u].begin(), it_u);
          std::size_t b = std::distance(non_path_incident[v].begin(), it_v);

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
      if (non_path_incident[u].size() == 1) {
        unpaired_modes.push_back(get_aux_majorana(u, spin, 1));
      } else if (non_path_incident[u].size() == 0) {
        unpaired_modes.push_back(get_aux_majorana(u, spin, 0));
        unpaired_modes.push_back(get_aux_majorana(u, spin, 1));
      }
    }

    // Unused aux Majorana modes become trivial stabilizers
    for (std::size_t i = 0; i < unpaired_modes.size(); i += 2) {
      if (i + 1 < unpaired_modes.size()) {
        auto [coeff, word] =
            get_bilinear(unpaired_modes[i], unpaired_modes[i + 1]);
        stabilizers.emplace_back(coeff, std::move(word));
      }
    }
  }

  std::vector<std::pair<std::complex<double>, SparsePauliWord>> upper_triangle;
  upper_triangle.reserve(base_modes * (base_modes - 1) / 2);

  for (std::size_t u = 0; u < base_modes; ++u) {
    for (std::size_t v = u + 1; v < base_modes; ++v) {
      std::size_t u_mode = u / 2;
      std::size_t v_mode = v / 2;
      std::size_t s_u = u_mode / V;
      std::size_t s_v = v_mode / V;
      std::size_t i = u_mode % V;
      std::size_t j = v_mode % V;
      std::size_t a = u % 2;
      std::size_t b_idx = v % 2;

      bool connected = (s_u == s_v) && lattice.are_connected(i, j);

      if (connected) {
        std::size_t path_i = site_to_path_idx[i];
        std::size_t path_j = site_to_path_idx[j];
        std::size_t path_min = std::min(path_i, path_j);
        std::size_t path_max = std::max(path_i, path_j);

        std::size_t p = get_sys_majorana(i, s_u, a);
        std::size_t q = get_sys_majorana(j, s_v, b_idx);
        auto [coeff, word] = get_bilinear(p, q);

        // For non-local paths, the raw JW bilinear iγ_pγ_q loses its long
        // Z-string by multiplying by edge stabilizer iγ̃_aγ̃_b
        if (path_max - path_min > 1) {
          auto key = std::make_pair(std::min(i, j), std::max(i, j));
          // Fetch the coefficient and Pauli word for the edge
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
        std::size_t q = get_sys_majorana(j, s_v, b_idx);
        auto [coeff, word] = get_bilinear(p, q);
        upper_triangle.emplace_back(coeff, std::move(word));
      }
    }
  }

  return MajoranaMapping(std::vector<SparsePauliWord>{},
                         std::move(upper_triangle), name, num_modes, base_modes,
                         name, std::nullopt, std::move(stabilizers));
}

}  // namespace qdk::chemistry::data
