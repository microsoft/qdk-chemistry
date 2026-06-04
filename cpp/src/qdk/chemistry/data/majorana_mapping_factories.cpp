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

namespace detail {

std::pair<std::uint64_t, std::uint64_t> snake_coordinates(std::uint64_t index,
                                                          std::uint64_t nx,
                                                          std::uint64_t ny) {
  std::uint64_t row = index / nx;
  std::uint64_t col;
  if (row % 2 == 0) {
    col = index % nx;
  } else {
    col = nx - 1 - (index % nx);
  }
  return {col, row};
}

bool is_vertical_edge(std::uint64_t i, std::uint64_t j, std::uint64_t nx,
                      std::uint64_t ny) {
  auto [ix, iy] = snake_coordinates(i, nx, ny);
  auto [jx, jy] = snake_coordinates(j, nx, ny);
  return (ix == jx && (iy + 1 == jy || jy + 1 == iy));
}

}  // namespace detail

MajoranaMapping MajoranaMapping::verstraete_cirac(
    const LatticeGraph& lattice, std::size_t num_spin_species) {
  if (num_spin_species == 0) {
    throw std::invalid_argument(
        "verstraete_cirac requires num_spin_species > 0");
  }

  std::uint64_t V = lattice.num_sites();
  std::uint64_t E = lattice.num_edges();

  // Solve: nx * ny = V, nx + ny = 2V - E
  double b = double(2 * V - E);
  double c = double(V);
  double disc = b * b - 4.0 * c;
  if (disc < 0) {
    throw std::invalid_argument(
        "Lattice graph is not a 2D square lattice with open boundaries");
  }
  double r1 = 0.5 * (b - std::sqrt(disc));
  double r2 = 0.5 * (b + std::sqrt(disc));
  std::uint64_t nx1 = std::round(r1);
  std::uint64_t nx2 = std::round(r2);

  auto check_nx = [&](std::uint64_t nx) -> bool {
    if (nx == 0 || V % nx != 0) return false;
    std::uint64_t ny = V / nx;
    std::uint64_t count = 0;
    const auto& adj = lattice.sparse_adjacency_matrix();
    for (int k = 0; k < adj.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
        std::uint64_t i = it.row();
        std::uint64_t j = it.col();
        if (i >= j) continue;
        if (std::abs(int(i - j)) == 1) {
          if (std::min(i, j) % nx == nx - 1) return false;
          count++;
        } else if (std::abs(int(i - j)) == nx) {
          count++;
        } else {
          return false;
        }
      }
    }
    return count == E;
  };

  std::uint64_t nx = 0, ny = 0;
  if (check_nx(nx1)) {
    nx = nx1;
    ny = V / nx1;
  } else if (check_nx(nx2)) {
    nx = nx2;
    ny = V / nx2;
  } else {
    throw std::invalid_argument(
        "Lattice graph is not a 2D square lattice with open boundaries");
  }

  if (nx < 2 || ny < 2) {
    throw std::invalid_argument(
        "verstraete_cirac requires a 2D grid of size at least 2x2");
  }

  std::size_t num_modes = num_spin_species * V;
  // Combined system has 2 * num_modes modes.
  std::size_t base_modes = 2 * num_modes;
  auto jw_base = MajoranaMapping::jordan_wigner(base_modes);

  // Construct upper triangle of bilinears: M * (M - 1) / 2 entries, where M = 2
  // * num_modes.
  std::size_t M = 2 * num_modes;
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> upper_triangle;
  upper_triangle.reserve(M * (M - 1) / 2);

  for (std::size_t u = 0; u < M; ++u) {
    for (std::size_t v = u + 1; v < M; ++v) {
      std::size_t u_mode = u / 2;
      std::size_t v_mode = v / 2;
      std::size_t s_u = u_mode / V;
      std::size_t s_v = v_mode / V;
      std::size_t i = u_mode % V;
      std::size_t j = v_mode % V;
      std::size_t a = u % 2;
      std::size_t b_idx = v % 2;

      if (s_u == s_v && detail::is_vertical_edge(i, j, nx, ny)) {
        // Vertical edge: multiply by stabilizer
        std::uint64_t top = std::min(i, j);
        std::uint64_t bot = std::max(i, j);
        auto coord_top = detail::snake_coordinates(top, nx, ny);
        std::uint64_t col = coord_top.first;

        std::size_t top_aux_mode = 2 * s_u * V + 2 * top + 1;
        std::size_t bot_aux_mode = 2 * s_u * V + 2 * bot + 1;

        std::size_t p, q;
        if (col % 2 == 0) {
          p = 2 * top_aux_mode;
          q = 2 * bot_aux_mode + 1;
        } else {
          p = 2 * bot_aux_mode;
          q = 2 * top_aux_mode + 1;
        }

        // Stabilizer S = i * gamma_p * gamma_q
        auto [coeff_stab, word_stab] = jw_base.bilinear(p, q);

        // System modes:
        std::size_t r = 2 * (2 * s_u * V + 2 * i) + a;
        std::size_t s_jw = 2 * (2 * s_v * V + 2 * j) + b_idx;
        auto [coeff_sys, word_sys] = jw_base.bilinear(r, s_jw);

        auto [phase, word_final] =
            PauliTermAccumulator::multiply_uncached(word_sys, word_stab);
        std::complex<double> coeff_final = coeff_sys * coeff_stab * phase;

        upper_triangle.emplace_back(coeff_final, std::move(word_final));
      } else {
        // Not a vertical edge
        std::size_t r = 2 * (2 * s_u * V + 2 * i) + a;
        std::size_t s_jw = 2 * (2 * s_v * V + 2 * j) + b_idx;
        auto [coeff, word] = jw_base.bilinear(r, s_jw);
        upper_triangle.emplace_back(coeff, word);
      }
    }
  }

  return MajoranaMapping(std::vector<SparsePauliWord>{},
                         std::move(upper_triangle), "verstraete-cirac",
                         num_modes, 2 * num_modes, "verstraete-cirac",
                         std::nullopt, nx, ny);
}

}  // namespace qdk::chemistry::data
