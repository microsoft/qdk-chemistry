// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
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
// operators.  The construction is fully general over the input graph — no
// lattice type is pattern-matched:
//
//   1. Recover a 2D layout from connectivity alone: corner seed plus
//      king's-move constraint propagation, with backtracking when several
//      grid cells remain ambiguous.  Axis-aligned and diagonal nearest-
//      neighbour bonds are both accepted (square, rectangular, and
//      triangular lattices are instances of the same embedding problem, not
//      separate code paths).
//   2. Order sites along a boustrophedon ("snake") path; combined qubit
//      2*s carries the physical mode and 2*s+1 its auxiliary partner.
//   3. Build the auxiliary coupling graph as a vertex-disjoint cover of
//      directed cycles (column pairs by default).  Whenever a lattice
//      edge straddles two cycles they are spliced into one, so every
//      edge's endpoints are always connected in the auxiliary graph.
//      Because each vertex appears exactly once as a tail (c-Majorana)
//      and once as a head (d-Majorana), all elementary auxiliary
//      bilinears P_e = i*c_a*d_b commute pairwise.
//   4. Every lattice edge that is not snake-adjacent is decorated with
//      the product of P_e along an auxiliary path between its endpoints.
//      The decoration is a member of the stabilizer group, hence acts as
//      +1 on the codespace: the encoded Hamiltonian restricted to the
//      codespace equals the bare Jordan-Wigner one for *any* lattice.
//      Vertical bonds use a single auxiliary edge, giving weight-4
//      hopping terms independent of system size.
//   5. Codespace stabilizers are *local* products of auxiliary Majorana
//      bilinears (Verstraete-Cirac 2005, eqs. 80–84): per cycle, one
//      snake-local seed edge plus all consecutive-edge products.  These
//      generate the full per-cycle group (unique auxiliary state) while
//      avoiding the non-local JW images individual long-range bilinears
//      would have.
//
// The lattice describes one spin species; the factory emits two
// independent Verstraete-Cirac blocks (alpha, then beta) so the result
// has num_modes = 2 * n_sites and is consumed exactly like a
// Jordan-Wigner mapping over 2 * n_sites modes.
namespace detail {

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

/// In-place (coeff, word) *= factor.
void vc_multiply_term(
    std::complex<double>& coeff, SparsePauliWord& word,
    const std::pair<std::complex<double>, SparsePauliWord>& factor) {
  auto prod = PauliTermAccumulator::multiply_uncached(word, factor.second);
  coeff = coeff * factor.first * prod.first;
  word = std::move(prod.second);
}

struct VcLayout {
  int nx = 0;
  int ny = 0;
  std::vector<std::pair<int, int>> coord;  // (column, row) per site
};

/// Collect grid cells within king's-move distance 1 of (x, y), excluding
/// (x, y) itself.
std::vector<std::pair<int, int>> vc_nn_cells(int x, int y) {
  std::vector<std::pair<int, int>> cells;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) continue;
      cells.emplace_back(x + dx, y + dy);
    }
  }
  return cells;
}

/// Free grid cells where vertex `w` may sit given the partial layout.
std::vector<std::pair<int, int>> vc_free_candidates(
    const std::vector<std::set<std::size_t>>& adj, std::size_t w,
    const std::vector<std::pair<int, int>>& coord,
    const std::map<std::pair<int, int>, std::size_t>& pos) {
  std::vector<std::pair<int, int>> candidates;
  bool have_candidates = false;
  for (std::size_t v : adj[w]) {
    if (coord[v].first < 0) continue;
    const auto nn = vc_nn_cells(coord[v].first, coord[v].second);
    if (!have_candidates) {
      candidates = nn;
      have_candidates = true;
    } else {
      std::vector<std::pair<int, int>> inter;
      for (const auto& cell : candidates) {
        if (std::find(nn.begin(), nn.end(), cell) != nn.end()) {
          inter.push_back(cell);
        }
      }
      candidates = std::move(inter);
    }
  }
  if (!have_candidates) return {};
  std::vector<std::pair<int, int>> free_cells;
  for (const auto& cell : candidates) {
    if (pos.count(cell) == 0) {
      free_cells.push_back(cell);
    }
  }
  return free_cells;
}

/// Apply forced placements: any vertex whose candidate set is a single free
/// cell is assigned immediately.  Returns false when a placed neighbour
/// leaves no consistent candidate cell.
bool vc_propagate_forced(const std::vector<std::set<std::size_t>>& adj,
                         std::vector<std::pair<int, int>>& coord,
                         std::map<std::pair<int, int>, std::size_t>& pos) {
  const std::size_t n = adj.size();
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t w = 0; w < n; ++w) {
      if (coord[w].first >= 0) continue;

      bool has_placed_neighbor = false;
      for (std::size_t v : adj[w]) {
        if (coord[v].first >= 0) has_placed_neighbor = true;
      }
      if (!has_placed_neighbor) continue;

      const auto free_cells = vc_free_candidates(adj, w, coord, pos);
      if (free_cells.empty()) {
        return false;
      }
      if (free_cells.size() == 1) {
        const auto [x, y] = free_cells[0];
        coord[w] = {x, y};
        pos[{x, y}] = w;
        changed = true;
      }
    }
  }
  return true;
}

bool vc_layout_valid(const std::vector<std::set<std::size_t>>& adj,
                     const std::vector<std::pair<int, int>>& coord) {
  const std::size_t n = adj.size();
  int nx = 0;
  int ny = 0;
  for (const auto& [x, y] : coord) {
    nx = std::max(nx, x + 1);
    ny = std::max(ny, y + 1);
  }
  if (static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) != n) {
    return false;
  }
  for (std::size_t v = 0; v < n; ++v) {
    for (std::size_t w : adj[v]) {
      if (w <= v) continue;
      const int dx = std::abs(coord[v].first - coord[w].first);
      const int dy = std::abs(coord[v].second - coord[w].second);
      if (dx > 1 || dy > 1) {
        return false;
      }
    }
  }
  return true;
}

/// Backtracking search after forced propagation stalls with ambiguous cells.
bool vc_search_layout(const std::vector<std::set<std::size_t>>& adj,
                      std::vector<std::pair<int, int>>& coord,
                      std::map<std::pair<int, int>, std::size_t>& pos) {
  if (!vc_propagate_forced(adj, coord, pos)) {
    return false;
  }
  const std::size_t n = adj.size();
  if (pos.size() == n) {
    return vc_layout_valid(adj, coord);
  }

  std::size_t best_w = n;
  std::size_t best_placed = 0;
  std::vector<std::pair<int, int>> best_cells;
  for (std::size_t w = 0; w < n; ++w) {
    if (coord[w].first >= 0) continue;
    std::size_t placed = 0;
    for (std::size_t v : adj[w]) {
      if (coord[v].first >= 0) ++placed;
    }
    if (placed == 0) continue;
    const auto free_cells = vc_free_candidates(adj, w, coord, pos);
    if (free_cells.empty()) {
      return false;
    }
    if (placed > best_placed) {
      best_placed = placed;
      best_w = w;
      best_cells = free_cells;
    }
  }
  if (best_w == n) {
    return false;
  }

  for (const auto& [x, y] : best_cells) {
    coord[best_w] = {x, y};
    pos[{x, y}] = best_w;
    if (vc_search_layout(adj, coord, pos)) {
      return true;
    }
    pos.erase({x, y});
    coord[best_w] = {-1, -1};
  }
  return false;
}

/// One embedding attempt: seed `corner` at (0,0) with axis neighbours
/// `a0` -> (1,0) and `b0` -> (0,1), then recover coordinates by
/// intersecting king's-move constraints from placed graph neighbours.
/// Forced placements are applied first; when several cells remain
/// ambiguous, backtracking picks among them.  No lattice-type-specific
/// rules are used.  Returns an empty vector when this seed does not
/// produce a complete rectangular layout whose edges are all (axis or
/// diagonal) nearest neighbours.
std::vector<std::pair<int, int>> vc_try_layout(
    const std::vector<std::set<std::size_t>>& adj, std::size_t corner,
    std::size_t a0, std::size_t b0) {
  const std::size_t n = adj.size();
  std::vector<std::pair<int, int>> coord(n, {-1, -1});
  std::map<std::pair<int, int>, std::size_t> pos;

  coord[corner] = {0, 0};
  pos[{0, 0}] = corner;
  coord[a0] = {1, 0};
  pos[{1, 0}] = a0;
  coord[b0] = {0, 1};
  pos[{0, 1}] = b0;

  if (!vc_search_layout(adj, coord, pos)) {
    return {};
  }
  return coord;
}

/// Recover a 2D layout from connectivity alone.  A single general
/// algorithm covers every supported lattice: corner candidates are tried
/// in deterministic (degree, index) order, each seeded with a pair of
/// mutually non-adjacent neighbours as the two axis directions.
VcLayout vc_recover_layout(const std::vector<std::set<std::size_t>>& adj) {
  const std::size_t n = adj.size();
  std::vector<std::size_t> order(n);
  for (std::size_t v = 0; v < n; ++v) order[v] = v;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return std::make_pair(adj[a].size(), a) < std::make_pair(adj[b].size(), b);
  });

  for (std::size_t corner : order) {
    for (std::size_t a : adj[corner]) {
      for (std::size_t b : adj[corner]) {
        if (b <= a || adj[a].count(b) != 0) continue;
        auto coord = vc_try_layout(adj, corner, a, b);
        if (coord.empty()) continue;
        VcLayout layout;
        layout.coord = std::move(coord);
        for (const auto& [x, y] : layout.coord) {
          layout.nx = std::max(layout.nx, x + 1);
          layout.ny = std::max(layout.ny, y + 1);
        }
        return layout;
      }
    }
  }
  throw std::invalid_argument(
      "MajoranaMapping::verstraete_cirac: lattice connectivity is not a 2D "
      "nearest-neighbour layout (could not recover a rectangular site grid "
      "from the edges)");
}

}  // namespace detail

MajoranaMapping MajoranaMapping::verstraete_cirac(const LatticeGraph& lattice) {
  using namespace detail;
  const std::size_t n = static_cast<std::size_t>(lattice.num_sites());
  if (n == 0) {
    throw std::invalid_argument(
        "MajoranaMapping::verstraete_cirac requires a non-empty lattice");
  }
  // The factory materializes O(M^2) Majorana bilinears (M = 4*n_sites).  Cap
  // the lattice size so oversized inputs fail fast with a clear message instead
  // of exhausting memory (acceptance tests exercise up to 4x4 sites).
  constexpr std::size_t kMaxVcSitesPerSpecies = 25;
  if (n > kMaxVcSitesPerSpecies) {
    throw std::invalid_argument(
        "MajoranaMapping::verstraete_cirac: lattice has " + std::to_string(n) +
        " sites per spin species, which exceeds the "
        "supported maximum of " +
        std::to_string(kMaxVcSitesPerSpecies) +
        " (the factory precomputes all Majorana bilinears and is intended "
        "for modest 2D lattices)");
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

  const VcLayout layout = vc_recover_layout(adj);
  const auto& coord = layout.coord;
  const int rnx = layout.nx;
  const int rny = layout.ny;
  if (rnx < 2 || rny < 2) {
    throw std::invalid_argument(
        "MajoranaMapping::verstraete_cirac requires a genuinely 2D lattice "
        "(at least 2 sites along each direction); 1D chains are already "
        "local under Jordan-Wigner");
  }

  // Snake position of every site and its O(1) inverse.
  std::vector<std::size_t> sigma(n);
  std::vector<std::size_t> site_of(n);
  std::map<std::pair<int, int>, std::size_t> site_at;
  for (std::size_t v = 0; v < n; ++v) {
    site_at[coord[v]] = v;
  }
  for (std::size_t v = 0; v < n; ++v) {
    const int x = coord[v].first;
    const int y = coord[v].second;
    const std::size_t s = y % 2 == 0
                              ? static_cast<std::size_t>(y * rnx + x)
                              : static_cast<std::size_t>((y + 1) * rnx - 1 - x);
    sigma[v] = s;
    site_of[s] = v;
  }

  // ── Auxiliary coupling graph: vertex-disjoint directed cycles ──
  //
  // nxt[v] is the successor of site v in its cycle; every site has
  // in-degree and out-degree exactly one, so the elementary auxiliary
  // bilinears P_v = i * c~(v) * d~(nxt[v]) all commute pairwise.
  // Column-pair cycles (down the left column, across the bottom, up the
  // right column, across the top) keep every vertical lattice bond a
  // single auxiliary edge; an odd trailing column forms its own cycle.
  std::vector<std::size_t> nxt(n);
  for (int k = 0; k < rnx; k += 2) {
    if (k + 1 < rnx) {
      for (int y = 0; y + 1 < rny; ++y) {
        nxt[site_at[{k, y}]] = site_at[{k, y + 1}];
      }
      nxt[site_at[{k, rny - 1}]] = site_at[{k + 1, rny - 1}];
      for (int y = rny - 1; y > 0; --y) {
        nxt[site_at[{k + 1, y}]] = site_at[{k + 1, y - 1}];
      }
      nxt[site_at[{k + 1, 0}]] = site_at[{k, 0}];
    } else {
      for (int y = 0; y + 1 < rny; ++y) {
        nxt[site_at[{k, y}]] = site_at[{k, y + 1}];
      }
      nxt[site_at[{k, rny - 1}]] = site_at[{k, 0}];
    }
  }

  // Decorated lattice edges: every bond whose endpoints are not adjacent
  // on the snake path needs an auxiliary decoration.
  std::vector<std::pair<std::size_t, std::size_t>> decorated;
  for (std::size_t v = 0; v < n; ++v) {
    for (std::size_t w : adj[v]) {
      if (w <= v) continue;
      const long long d =
          static_cast<long long>(sigma[v]) - static_cast<long long>(sigma[w]);
      if (d != 1 && d != -1) {
        decorated.emplace_back(v, w);
      }
    }
  }

  // Splice cycles together whenever a decorated edge straddles two of
  // them, so an auxiliary path always exists between its endpoints.
  // Swapping the successors of the two endpoints merges their cycles
  // while preserving in/out-degree one everywhere.
  auto cycle_ids = [&]() {
    std::vector<long long> id(n, -1);
    long long next_id = 0;
    for (std::size_t v = 0; v < n; ++v) {
      if (id[v] >= 0) continue;
      std::size_t w = v;
      while (id[w] < 0) {
        id[w] = next_id;
        w = nxt[w];
      }
      ++next_id;
    }
    return id;
  };
  auto ids = cycle_ids();
  for (const auto& [u, v] : decorated) {
    if (ids[u] != ids[v]) {
      std::swap(nxt[u], nxt[v]);
      ids = cycle_ids();
    }
  }

  // Undirected auxiliary neighbours for path finding.
  std::vector<std::size_t> prv(n);
  for (std::size_t v = 0; v < n; ++v) prv[nxt[v]] = v;
  auto aux_path_edges = [&](std::size_t from,
                            std::size_t to) -> std::vector<std::size_t> {
    // BFS on the undirected cycle graph; returns the tail site of every
    // directed auxiliary edge along the path.
    std::vector<long long> parent(n, -1);
    std::vector<std::size_t> queue{from};
    parent[from] = static_cast<long long>(from);
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const std::size_t v = queue[head];
      if (v == to) break;
      for (std::size_t w : {nxt[v], prv[v]}) {
        if (parent[w] < 0) {
          parent[w] = static_cast<long long>(v);
          queue.push_back(w);
        }
      }
    }
    if (parent[to] < 0) {
      throw std::logic_error(
          "MajoranaMapping::verstraete_cirac: internal error — no auxiliary "
          "path between decorated edge endpoints");
    }
    std::vector<std::size_t> edges;
    for (std::size_t v = to; v != from;
         v = static_cast<std::size_t>(parent[v])) {
      const auto p = static_cast<std::size_t>(parent[v]);
      edges.push_back(nxt[p] == v ? p : v);
    }
    return edges;
  };

  // Per-block (spin) layout: block b occupies snake positions
  // b*n .. b*n + n - 1; the auxiliary qubit of site v in block b is
  // 2*(b*n + sigma(v)) + 1.
  const std::size_t num_modes = 2 * n;
  const std::size_t M = 2 * num_modes;  // total Majorana / qubit count
  auto aux_qubit = [&](std::size_t block, std::size_t v) {
    return 2 * (block * n + sigma[v]) + 1;
  };

  // Elementary auxiliary bilinears and decorations, per spin block.
  std::vector<std::vector<std::pair<std::complex<double>, SparsePauliWord>>>
      edge_p(2);
  using DecorationMap =
      std::map<std::pair<std::size_t, std::size_t>,
               std::pair<std::complex<double>, SparsePauliWord>>;
  std::vector<DecorationMap> decoration(2);
  for (std::size_t block = 0; block < 2; ++block) {
    edge_p[block].reserve(n);
    for (std::size_t v = 0; v < n; ++v) {
      edge_p[block].push_back(
          vc_pstab(aux_qubit(block, v), aux_qubit(block, nxt[v])));
    }
    for (const auto& [u, v] : decorated) {
      // Product of P_e along an auxiliary path between the endpoints: a
      // stabilizer-group element, so it acts as +1 on the codespace and
      // the encoded operator equals the bare one there.  The P_e commute
      // pairwise, so the multiplication order is irrelevant.
      std::complex<double> coeff(1.0, 0.0);
      SparsePauliWord word;
      for (std::size_t tail : aux_path_edges(u, v)) {
        vc_multiply_term(coeff, word, edge_p[block][tail]);
      }
      decoration[block].emplace(std::make_pair(u, v),
                                std::make_pair(coeff, std::move(word)));
    }
  }

  // ── Bilinear table: i * gamma_J * gamma_K for all J < K in [0, M) ──
  //
  // Materializes the full upper triangle (O(M^2) Pauli words).  Practical for
  // modest lattices (acceptance tests up to 4x4 sites); larger systems should
  // use a different encoding or a lazily-built mapping.
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears;
  bilinears.reserve(M * (M - 1) / 2);
  for (std::size_t J = 0; J < M; ++J) {
    const std::size_t p = J / 2;
    const std::size_t ap = J % 2;
    const std::size_t qp_qubit = 2 * ((p / n) * n + sigma[p % n]);
    auto gamma_J = vc_jw_majorana(qp_qubit, ap == 0 ? op_x : op_y);
    for (std::size_t K = J + 1; K < M; ++K) {
      const std::size_t q = K / 2;
      const std::size_t bq = K % 2;
      const std::size_t qq_qubit = 2 * ((q / n) * n + sigma[q % n]);
      auto gamma_K = vc_jw_majorana(qq_qubit, bq == 0 ? op_x : op_y);

      auto bare = PauliTermAccumulator::multiply_uncached(gamma_J, gamma_K);
      std::complex<double> coeff = std::complex<double>(0.0, 1.0) * bare.first;
      SparsePauliWord word = std::move(bare.second);

      // Decorate non-path lattice edges within the same spin block.
      if (p != q && (p / n) == (q / n)) {
        const std::size_t block = p / n;
        const std::size_t vp = std::min(p % n, q % n);
        const std::size_t vq = std::max(p % n, q % n);
        auto dec = decoration[block].find({vp, vq});
        if (dec != decoration[block].end()) {
          vc_multiply_term(coeff, word, dec->second);
        }
      }
      bilinears.emplace_back(coeff, std::move(word));
    }
  }

  // ── Stabilizers: local products along each auxiliary cycle ──
  //
  // Per cycle of length L: one "seed" elementary bilinear on the most
  // snake-local edge plus the L-1 consecutive-edge products.  Together
  // they generate every elementary P_e of the cycle (rank L), fixing a
  // unique auxiliary state, while each stored generator stays local —
  // individual bilinears along a long cycle would JW-map to non-local
  // penalties (Verstraete-Cirac 2005, eqs. 80–84).
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers;
  ids = cycle_ids();
  std::vector<std::size_t> cycle_start;
  for (std::size_t v = 0; v < n; ++v) {
    if (static_cast<std::size_t>(ids[v]) == cycle_start.size()) {
      cycle_start.push_back(v);
    }
  }
  for (std::size_t block = 0; block < 2; ++block) {
    for (std::size_t start : cycle_start) {
      std::vector<std::size_t> tails;  // directed edges of this cycle
      std::size_t v = start;
      do {
        tails.push_back(v);
        v = nxt[v];
      } while (v != start);

      std::size_t seed = 0;
      std::size_t best_span = std::numeric_limits<std::size_t>::max();
      for (std::size_t i = 0; i < tails.size(); ++i) {
        const std::size_t a = sigma[tails[i]];
        const std::size_t b = sigma[nxt[tails[i]]];
        const std::size_t span = a < b ? b - a : a - b;
        if (span < best_span) {
          best_span = span;
          seed = i;
        }
      }
      stabilizers.push_back(edge_p[block][tails[seed]]);
      for (std::size_t i = 0; i + 1 < tails.size(); ++i) {
        auto product = edge_p[block][tails[i]];
        vc_multiply_term(product.first, product.second,
                         edge_p[block][tails[i + 1]]);
        stabilizers.push_back(std::move(product));
      }
    }
  }

  return MajoranaMapping({}, std::move(bilinears), "verstraete-cirac",
                         num_modes,
                         /*num_qubits=*/M, "verstraete-cirac", std::nullopt,
                         std::move(stabilizers));
}

}  // namespace qdk::chemistry::data
