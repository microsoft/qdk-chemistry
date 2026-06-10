// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/data/tapering.hpp>
#include <set>
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

// ── Factory: Verstraete-Cirac ────────────────────────────────────────
//
// The Verstraete-Cirac encoding pairs each lattice site with one auxiliary
// qubit so that nearest-neighbour hopping maps to constant-weight Pauli
// operators.  The construction is built from the lattice *edges*:
//
//   1. Recognise a 2D site layout (square/triangular grid labelling, or
//      recover coordinates from connectivity for other planar graphs).
//   2. Order sites along a boustrophedon ("snake") path; combined qubit 2*s
//      carries the physical mode and 2*s+1 carries its auxiliary partner.
//   3. The mapping is the combined Jordan-Wigner transform over the snake,
//      with every non-path lattice edge decorated by an auxiliary bilinear.
//   4. Codespace stabilizers are *local* products of auxiliary Majorana
//      bilinears along the auxiliary-fermion graph (Verstraete-Cirac 2005,
//      eqs. 80–84; OpenFermion stabilizer_local_2d_square).  Individual
//      bilinears would JW-map to non-local penalties; the products do not.
//
// The lattice describes one spin species; the factory emits two independent
// Verstraete-Cirac blocks (alpha, then beta) so the result has
// num_modes = 2 * n_sites and is consumed exactly like a Jordan-Wigner
// mapping over 2 * n_sites modes.
namespace detail {

enum class VcGridKind { SQUARE, TRIANGULAR, RECOVERED };

struct VcGridLayout {
  int nx;
  int ny;
  VcGridKind kind;
  std::vector<std::pair<int, int>> coord;
};

bool vc_has_edge(const std::vector<std::set<std::size_t>>& adj, std::size_t u,
                 std::size_t v) {
  return adj[u].count(v) != 0;
}

std::size_t vc_grid_index(int x, int y, int nx) {
  return static_cast<std::size_t>(y * nx + x);
}

bool vc_matches_square_grid(int nx, int ny,
                            const std::vector<std::set<std::size_t>>& adj) {
  const std::size_t n = adj.size();
  if (static_cast<std::size_t>(nx * ny) != n) {
    return false;
  }
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const std::size_t v = vc_grid_index(x, y, nx);
      if (x + 1 < nx && !vc_has_edge(adj, v, vc_grid_index(x + 1, y, nx))) {
        return false;
      }
      if (y + 1 < ny && !vc_has_edge(adj, v, vc_grid_index(x, y + 1, nx))) {
        return false;
      }
      if (x + 1 < nx && y + 1 < ny &&
          vc_has_edge(adj, v, vc_grid_index(x + 1, y + 1, nx))) {
        return false;
      }
    }
  }
  for (std::size_t v = 0; v < n; ++v) {
    for (auto w : adj[v]) {
      if (w <= v) {
        continue;
      }
      const int dx =
          std::abs(static_cast<int>(v % nx) - static_cast<int>(w % nx));
      const int dy =
          std::abs(static_cast<int>(v / nx) - static_cast<int>(w / nx));
      if (!((dx == 1 && dy == 0) || (dx == 0 && dy == 1))) {
        return false;
      }
      if (dx == 1 && dy == 1) {
        return false;
      }
    }
  }
  return true;
}

bool vc_matches_triangular_grid(int nx, int ny,
                                const std::vector<std::set<std::size_t>>& adj) {
  const std::size_t n = adj.size();
  if (static_cast<std::size_t>(nx * ny) != n) {
    return false;
  }
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const std::size_t v = vc_grid_index(x, y, nx);
      if (x + 1 < nx && !vc_has_edge(adj, v, vc_grid_index(x + 1, y, nx))) {
        return false;
      }
      if (y + 1 < ny && !vc_has_edge(adj, v, vc_grid_index(x, y + 1, nx))) {
        return false;
      }
      if (x + 1 < nx && y + 1 < ny &&
          !vc_has_edge(adj, v, vc_grid_index(x + 1, y + 1, nx))) {
        return false;
      }
    }
  }
  for (std::size_t v = 0; v < n; ++v) {
    for (auto w : adj[v]) {
      if (w <= v) {
        continue;
      }
      const int x0 = static_cast<int>(v % nx);
      const int y0 = static_cast<int>(v / nx);
      const int x1 = static_cast<int>(w % nx);
      const int y1 = static_cast<int>(w / nx);
      const int dx = std::abs(x0 - x1);
      const int dy = std::abs(y0 - y1);
      const bool square_edge = (dx == 1 && dy == 0) || (dx == 0 && dy == 1);
      const bool diag_edge = (dx == 1 && dy == 1 && x0 < x1 && y0 < y1);
      if (!square_edge && !diag_edge) {
        return false;
      }
    }
  }
  return true;
}

/// Combined Jordan-Wigner Majorana on `Q`: Z on all qubits < Q, then op on Q
/// (op = X for the "c" Majorana, Y for the "d" Majorana).
SparsePauliWord vc_jw_majorana(std::size_t qubit, std::uint8_t op) {
  SparsePauliWord word;
  word.reserve(qubit + 1);
  for (std::size_t q = 0; q < qubit; ++q) {
    word.emplace_back(static_cast<std::uint64_t>(q), op_z);
  }
  word.emplace_back(static_cast<std::uint64_t>(qubit), op);
  return word;
}

/// Pauli image of i * c_a * d_b (a, b combined-qubit indices).
std::pair<std::complex<double>, SparsePauliWord> vc_pstab(std::size_t a,
                                                          std::size_t b) {
  auto c_a = vc_jw_majorana(a, op_x);
  auto d_b = vc_jw_majorana(b, op_y);
  auto [phase, word] = PauliTermAccumulator::multiply_uncached(c_a, d_b);
  return {std::complex<double>(0.0, 1.0) * phase, std::move(word)};
}

void vc_multiply_stabilizer(
    std::complex<double>& coeff, SparsePauliWord& word,
    const std::pair<std::complex<double>, SparsePauliWord>& factor) {
  auto prod = PauliTermAccumulator::multiply_uncached(word, factor.second);
  coeff = coeff * factor.first * prod.first;
  word = std::move(prod.second);
}

int vc_snake_index(int col, int row, int nx) {
  return row % 2 == 0 ? row * nx + col : (row + 1) * nx - 1 - col;
}

std::pair<int, int> vc_snake_to_coord(int site, int nx, int ny) {
  for (int row = 0; row < ny; ++row) {
    for (int col = 0; col < nx; ++col) {
      if (vc_snake_index(col, row, nx) == site) {
        return {col, row};
      }
    }
  }
  throw std::invalid_argument(
      "MajoranaMapping::verstraete_cirac: invalid snake site index");
}

/// Local auxiliary stabilizer for a directed edge on the auxiliary graph.
/// Follows OpenFermion ``stabilizer_local_2d_square`` (products of P_ij).
std::pair<std::complex<double>, SparsePauliWord> vc_stabilizer_local_edge(
    int site_from, int site_to, int nx, int ny, std::size_t block_off) {
  const auto [col_i, row_i] = vc_snake_to_coord(site_from, nx, ny);
  const auto [col_j, row_j] = vc_snake_to_coord(site_to, nx, ny);

  const std::size_t aux_from =
      2 * (block_off + static_cast<std::size_t>(site_from)) + 1;
  const std::size_t aux_to =
      2 * (block_off + static_cast<std::size_t>(site_to)) + 1;

  auto result = (col_i % 2 == 0) ? vc_pstab(aux_from, aux_to)
                                 : vc_pstab(aux_to, aux_from);

  if (col_i == col_j && std::abs(row_i - row_j) == 1) {
    const int top_row = std::min(row_i, row_j);
    const int col = col_i;
    if (top_row % 2 == 0) {
      if (col + 1 < nx) {
        const int et = vc_snake_index(col + 1, top_row, nx);
        const int eb = vc_snake_index(col + 1, top_row + 1, nx);
        const std::size_t aux_et =
            2 * (block_off + static_cast<std::size_t>(et)) + 1;
        const std::size_t aux_eb =
            2 * (block_off + static_cast<std::size_t>(eb)) + 1;
        const auto extra = ((col + 1) % 2 == 0) ? vc_pstab(aux_et, aux_eb)
                                                : vc_pstab(aux_eb, aux_et);
        vc_multiply_stabilizer(result.first, result.second, extra);
      }
    } else if (col > 0) {
      const int et = vc_snake_index(col - 1, top_row, nx);
      const int eb = vc_snake_index(col - 1, top_row + 1, nx);
      const std::size_t aux_et =
          2 * (block_off + static_cast<std::size_t>(et)) + 1;
      const std::size_t aux_eb =
          2 * (block_off + static_cast<std::size_t>(eb)) + 1;
      const auto extra = ((col - 1) % 2 == 0) ? vc_pstab(aux_et, aux_eb)
                                              : vc_pstab(aux_eb, aux_et);
      vc_multiply_stabilizer(result.first, result.second, extra);
    }
  }
  return result;
}

/// Directed edges of the auxiliary-fermion graph (OpenFermion layout).
std::vector<std::pair<int, int>> vc_auxiliary_graph_edges(int nx, int ny) {
  std::vector<std::pair<int, int>> edges;
  for (int k = 0; k < nx; k += 2) {
    if (k + 1 >= nx) {
      for (int row = 0; row + 1 < ny; ++row) {
        edges.emplace_back(vc_snake_index(k, row, nx),
                           vc_snake_index(k, row + 1, nx));
      }
      if (nx > 1) {
        edges.emplace_back(vc_snake_index(k, ny - 1, nx),
                           vc_snake_index(0, 0, nx));
      }
      continue;
    }
    edges.emplace_back(vc_snake_index(k + 1, 0, nx), vc_snake_index(k, 0, nx));
    edges.emplace_back(vc_snake_index(k, ny - 1, nx),
                       vc_snake_index(k + 1, ny - 1, nx));
    for (int row = 0; row + 1 < ny; ++row) {
      edges.emplace_back(vc_snake_index(k, row, nx),
                         vc_snake_index(k, row + 1, nx));
      edges.emplace_back(vc_snake_index(k + 1, row + 1, nx),
                         vc_snake_index(k + 1, row, nx));
    }
  }
  return edges;
}

std::vector<std::pair<std::complex<double>, SparsePauliWord>>
vc_build_local_stabilizers(int nx, int ny, std::size_t block_off) {
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers;
  for (const auto& [from, to] : vc_auxiliary_graph_edges(nx, ny)) {
    stabilizers.push_back(
        vc_stabilizer_local_edge(from, to, nx, ny, block_off));
  }
  return stabilizers;
}

/// Recover (column, row) coordinates of each vertex from edges alone.
/// Returns coord[v] = (x, y); throws if the graph is not one rectangular grid.
std::vector<std::pair<int, int>> vc_recover_coords(
    const std::vector<std::set<std::size_t>>& adj) {
  const std::size_t n = adj.size();
  std::size_t corner = n;
  for (std::size_t v = 0; v < n; ++v) {
    if (adj[v].size() == 2) {
      corner = v;
      break;
    }
  }
  if (corner == n) {
    throw std::invalid_argument(
        "MajoranaMapping::verstraete_cirac: lattice has no degree-2 corner; "
        "connectivity is not a rectangular grid");
  }
  auto it = adj[corner].begin();
  std::size_t a0 = *it++;
  std::size_t b0 = *it;
  if (a0 > b0) std::swap(a0, b0);

  std::vector<std::pair<int, int>> coord(n, {-1, -1});
  std::map<std::pair<int, int>, std::size_t> pos;
  auto assign = [&](std::size_t v, int x, int y) {
    const auto key = std::make_pair(x, y);
    const auto existing = pos.find(key);
    if (existing != pos.end() && existing->second != v) {
      throw std::invalid_argument(
          "MajoranaMapping::verstraete_cirac: coordinate recovery assigned "
          "multiple vertices to the same grid position");
    }
    coord[v] = key;
    pos[key] = v;
  };
  assign(corner, 0, 0);
  assign(a0, 1, 0);
  assign(b0, 0, 1);
  auto assigned = [&](std::size_t v) { return coord[v].first >= 0; };

  auto common_unassigned = [&](std::size_t u, std::size_t w) -> long long {
    long long found = -1;
    int cnt = 0;
    for (std::size_t x : adj[u]) {
      if (adj[w].count(x) && !assigned(x)) {
        found = static_cast<long long>(x);
        ++cnt;
      }
    }
    return cnt == 1 ? found : -1;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<std::pair<std::pair<int, int>, std::size_t>> items(pos.begin(),
                                                                   pos.end());
    for (const auto& [xy, v] : items) {
      (void)v;
      int x = xy.first, y = xy.second;
      if (pos.count({x + 1, y}) && pos.count({x, y + 1}) &&
          !pos.count({x + 1, y + 1})) {
        long long t = common_unassigned(pos[{x + 1, y}], pos[{x, y + 1}]);
        if (t >= 0) {
          assign(static_cast<std::size_t>(t), x + 1, y + 1);
          changed = true;
        }
      }
    }
    for (int x = 1; pos.count({x, 0}); ++x) {
      if (!pos.count({x + 1, 0}) && pos.count({x, 1})) {
        std::size_t excl_a = pos[{x - 1, 0}], excl_b = pos[{x, 1}];
        long long cand = -1;
        int cnt = 0;
        for (std::size_t w : adj[pos[{x, 0}]]) {
          if (!assigned(w) && w != excl_a && w != excl_b) {
            cand = static_cast<long long>(w);
            ++cnt;
          }
        }
        if (cnt == 1) {
          assign(static_cast<std::size_t>(cand), x + 1, 0);
          changed = true;
        }
      }
    }
    for (int y = 1; pos.count({0, y}); ++y) {
      if (!pos.count({0, y + 1}) && pos.count({1, y})) {
        std::size_t excl_a = pos[{0, y - 1}], excl_b = pos[{1, y}];
        long long cand = -1;
        int cnt = 0;
        for (std::size_t w : adj[pos[{0, y}]]) {
          if (!assigned(w) && w != excl_a && w != excl_b) {
            cand = static_cast<long long>(w);
            ++cnt;
          }
        }
        if (cnt == 1) {
          assign(static_cast<std::size_t>(cand), 0, y + 1);
          changed = true;
        }
      }
    }
  }
  for (std::size_t v = 0; v < n; ++v) {
    if (!assigned(v)) {
      throw std::invalid_argument(
          "MajoranaMapping::verstraete_cirac: lattice connectivity is not a "
          "single rectangular grid");
    }
  }
  return coord;
}

VcGridLayout vc_detect_grid_layout(
    const std::vector<std::set<std::size_t>>& adj) {
  const std::size_t n = adj.size();
  for (int nx = 1; nx <= static_cast<int>(std::sqrt(static_cast<double>(n)));
       ++nx) {
    if (static_cast<std::size_t>(nx) > n ||
        n % static_cast<std::size_t>(nx) != 0) {
      continue;
    }
    const int ny = static_cast<int>(n / nx);
    if (vc_matches_triangular_grid(nx, ny, adj)) {
      VcGridLayout layout{nx, ny, VcGridKind::TRIANGULAR, {}};
      layout.coord.reserve(n);
      for (std::size_t v = 0; v < n; ++v) {
        layout.coord.emplace_back(static_cast<int>(v % nx),
                                  static_cast<int>(v / nx));
      }
      return layout;
    }
    if (vc_matches_square_grid(nx, ny, adj)) {
      VcGridLayout layout{nx, ny, VcGridKind::SQUARE, {}};
      layout.coord.reserve(n);
      for (std::size_t v = 0; v < n; ++v) {
        layout.coord.emplace_back(static_cast<int>(v % nx),
                                  static_cast<int>(v / nx));
      }
      return layout;
    }
  }
  VcGridLayout layout{0, 0, VcGridKind::RECOVERED, vc_recover_coords(adj)};
  for (const auto& [x, y] : layout.coord) {
    layout.nx = std::max(layout.nx, x + 1);
    layout.ny = std::max(layout.ny, y + 1);
  }
  if (static_cast<std::size_t>(layout.nx * layout.ny) != n) {
    throw std::invalid_argument(
        "MajoranaMapping::verstraete_cirac: lattice connectivity is not a "
        "recognised 2D nearest-neighbour layout");
  }
  return layout;
}

}  // namespace detail

MajoranaMapping MajoranaMapping::verstraete_cirac(const LatticeGraph& lattice) {
  using namespace detail;
  const std::size_t n = static_cast<std::size_t>(lattice.num_sites());
  if (n == 0) {
    throw std::invalid_argument(
        "MajoranaMapping::verstraete_cirac requires a non-empty lattice");
  }

  // Build undirected adjacency from the sparse matrix.
  std::vector<std::set<std::size_t>> adj(n);
  const auto& A = lattice.sparse_adjacency_matrix();
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      auto r = static_cast<std::size_t>(it.row());
      auto c = static_cast<std::size_t>(it.col());
      if (r != c) {
        adj[r].insert(c);
        adj[c].insert(r);
      }
    }
  }

  const auto layout = vc_detect_grid_layout(adj);
  const auto& coord = layout.coord;
  const int rnx = layout.nx;
  const int rny = layout.ny;

  if (layout.kind == VcGridKind::RECOVERED) {
    for (std::size_t v = 0; v < n; ++v) {
      for (auto w : adj[v]) {
        if (w <= v) {
          continue;
        }
        const int dx =
            std::abs(coord[v].first - coord[static_cast<std::size_t>(w)].first);
        const int dy = std::abs(coord[v].second -
                                coord[static_cast<std::size_t>(w)].second);
        if (dx + dy != 1) {
          throw std::invalid_argument(
              "MajoranaMapping::verstraete_cirac: lattice edge does not "
              "connect nearest neighbours on the recovered 2D layout");
        }
      }
    }
  }
  auto snake = [rnx](int c, int r) -> std::size_t {
    return r % 2 == 0 ? static_cast<std::size_t>(r * rnx + c)
                      : static_cast<std::size_t>((r + 1) * rnx - 1 - c);
  };

  // Per-block (spin) layout: block b occupies snake positions b*n .. b*n+n-1.
  const std::size_t num_modes = 2 * n;
  const std::size_t M = 2 * num_modes;  // total Majorana / qubit count

  // s(mode): combined snake position of the physical qubit for a given mode.
  auto sigma = [&](std::size_t mode) -> std::size_t {
    std::size_t block = mode / n;
    std::size_t v = mode % n;
    return block * n + snake(coord[v].first, coord[v].second);
  };

  // ── Bilinear table: i * gamma_J * gamma_K for all J < K in [0, M) ──
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears;
  bilinears.reserve(M * (M - 1) / 2);
  for (std::size_t J = 0; J < M; ++J) {
    std::size_t p = J / 2, ap = J % 2;
    std::size_t qp_qubit = 2 * sigma(p);
    auto gamma_J = vc_jw_majorana(qp_qubit, ap == 0 ? op_x : op_y);
    for (std::size_t K = J + 1; K < M; ++K) {
      std::size_t q = K / 2, bq = K % 2;
      std::size_t qq_qubit = 2 * sigma(q);
      auto gamma_K = vc_jw_majorana(qq_qubit, bq == 0 ? op_x : op_y);

      auto bare = PauliTermAccumulator::multiply_uncached(gamma_J, gamma_K);
      std::complex<double> coeff = std::complex<double>(0.0, 1.0) * bare.first;
      SparsePauliWord word = std::move(bare.second);

      // Decorate non-path edges within the same spin block.
      if (p != q && (p / n) == (q / n)) {
        std::size_t vp = p % n, vq = q % n;
        if (adj[vp].count(vq)) {
          std::size_t sp = sigma(p), sq = sigma(q);
          long long d = static_cast<long long>(sp) - static_cast<long long>(sq);
          if (d != 1 && d != -1) {
            // top = endpoint with the smaller row (vertical bond).
            bool p_is_top = coord[vp].second < coord[vq].second;
            std::size_t s_top = p_is_top ? sp : sq;
            std::size_t s_bot = p_is_top ? sq : sp;
            auto dec = vc_pstab(2 * s_top + 1, 2 * s_bot + 1);
            auto prod =
                PauliTermAccumulator::multiply_uncached(word, dec.second);
            coeff = coeff * dec.first * prod.first;
            word = std::move(prod.second);
          }
        }
      }
      bilinears.emplace_back(coeff, std::move(word));
    }
  }

  // ── Stabilizers: local plaquette products on the auxiliary graph ──
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers;
  for (std::size_t block = 0; block < 2; ++block) {
    auto block_stabs = vc_build_local_stabilizers(rnx, rny, block * n);
    stabilizers.insert(stabilizers.end(), block_stabs.begin(),
                       block_stabs.end());
  }

  return MajoranaMapping({}, std::move(bilinears), "verstraete-cirac",
                         num_modes,
                         /*num_qubits=*/M, "verstraete-cirac", std::nullopt,
                         std::move(stabilizers));
}

}  // namespace qdk::chemistry::data
