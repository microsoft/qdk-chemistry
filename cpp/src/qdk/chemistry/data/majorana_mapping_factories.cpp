// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <map>
#include <queue>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
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

using StabilizerTerm = std::pair<std::complex<double>, SparsePauliWord>;

std::pair<std::uint64_t, std::uint64_t> canonical_edge(std::uint64_t u,
                                                       std::uint64_t v) {
  return std::minmax(u, v);
}

bool are_neighbors(const std::vector<std::vector<std::uint64_t>>& neighbors,
                   std::uint64_t u, std::uint64_t v) {
  return std::binary_search(neighbors[u].begin(), neighbors[u].end(), v);
}

std::size_t unvisited_neighbor_count(
    const std::vector<std::vector<std::uint64_t>>& neighbors,
    const std::vector<bool>& visited, std::uint64_t u) {
  std::size_t count = 0;
  for (auto v : neighbors[u]) {
    if (!visited[v]) {
      ++count;
    }
  }
  return count;
}

std::vector<std::uint64_t> vertices_by_degree(
    const std::vector<std::vector<std::uint64_t>>& neighbors) {
  std::vector<std::uint64_t> vertices;
  vertices.reserve(neighbors.size());
  for (std::uint64_t u = 0; u < neighbors.size(); ++u) {
    vertices.push_back(u);
  }
  std::sort(vertices.begin(), vertices.end(),
            [&](std::uint64_t a, std::uint64_t b) {
              if (neighbors[a].size() != neighbors[b].size()) {
                return neighbors[a].size() < neighbors[b].size();
              }
              return a < b;
            });
  return vertices;
}

std::vector<std::uint64_t> greedy_hamiltonian_path(
    const std::vector<std::vector<std::uint64_t>>& neighbors) {
  const auto n = neighbors.size();
  for (auto start : vertices_by_degree(neighbors)) {
    std::vector<bool> visited(n, false);
    std::vector<std::uint64_t> path;
    path.reserve(n);

    std::uint64_t current = start;
    visited[current] = true;
    path.push_back(current);

    while (path.size() < n) {
      std::vector<std::uint64_t> candidates;
      for (auto next : neighbors[current]) {
        if (!visited[next]) {
          candidates.push_back(next);
        }
      }
      if (candidates.empty()) {
        break;
      }
      std::sort(candidates.begin(), candidates.end(),
                [&](std::uint64_t a, std::uint64_t b) {
                  auto a_count = unvisited_neighbor_count(neighbors, visited, a);
                  auto b_count = unvisited_neighbor_count(neighbors, visited, b);
                  if (a_count != b_count) {
                    return a_count < b_count;
                  }
                  return a < b;
                });
      current = candidates.front();
      visited[current] = true;
      path.push_back(current);
    }

    if (path.size() == n) {
      return path;
    }
  }
  return {};
}

bool bounded_hamiltonian_path_dfs(
    const std::vector<std::vector<std::uint64_t>>& neighbors,
    std::vector<bool>& visited, std::vector<std::uint64_t>& path,
    std::size_t& steps, std::size_t max_steps) {
  if (path.size() == neighbors.size()) {
    return true;
  }
  if (++steps > max_steps) {
    return false;
  }

  std::uint64_t current = path.back();
  std::vector<std::uint64_t> candidates;
  for (auto next : neighbors[current]) {
    if (!visited[next]) {
      candidates.push_back(next);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [&](std::uint64_t a, std::uint64_t b) {
              auto a_count = unvisited_neighbor_count(neighbors, visited, a);
              auto b_count = unvisited_neighbor_count(neighbors, visited, b);
              if (a_count != b_count) {
                return a_count < b_count;
              }
              return a < b;
            });

  for (auto next : candidates) {
    visited[next] = true;
    path.push_back(next);
    if (bounded_hamiltonian_path_dfs(neighbors, visited, path, steps,
                                     max_steps)) {
      return true;
    }
    path.pop_back();
    visited[next] = false;
  }
  return false;
}

std::vector<std::uint64_t> bounded_hamiltonian_path(
    const std::vector<std::vector<std::uint64_t>>& neighbors) {
  constexpr std::size_t max_steps = 1000000;
  const auto n = neighbors.size();
  std::size_t steps = 0;

  for (auto start : vertices_by_degree(neighbors)) {
    if (steps >= max_steps) {
      break;
    }
    std::vector<bool> visited(n, false);
    std::vector<std::uint64_t> path;
    path.reserve(n);

    visited[start] = true;
    path.push_back(start);
    if (bounded_hamiltonian_path_dfs(neighbors, visited, path, steps,
                                     max_steps)) {
      return path;
    }
  }
  return {};
}

void append_dfs_order(const std::vector<std::vector<std::uint64_t>>& neighbors,
                      std::uint64_t start, std::vector<bool>& visited,
                      std::vector<std::uint64_t>& order) {
  visited[start] = true;
  order.push_back(start);

  std::vector<std::uint64_t> candidates;
  for (auto next : neighbors[start]) {
    if (!visited[next]) {
      candidates.push_back(next);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [&](std::uint64_t a, std::uint64_t b) {
              if (neighbors[a].size() != neighbors[b].size()) {
                return neighbors[a].size() < neighbors[b].size();
              }
              return a < b;
            });

  for (auto next : candidates) {
    if (!visited[next]) {
      append_dfs_order(neighbors, next, visited, order);
    }
  }
}

std::vector<std::uint64_t> dfs_vertex_order(
    const std::vector<std::vector<std::uint64_t>>& neighbors) {
  const auto n = neighbors.size();
  std::vector<bool> visited(n, false);
  std::vector<std::uint64_t> order;
  order.reserve(n);

  for (auto start : vertices_by_degree(neighbors)) {
    if (!visited[start]) {
      append_dfs_order(neighbors, start, visited, order);
    }
  }
  return order;
}

std::vector<std::uint64_t> choose_vc_path_order(
    const std::vector<std::vector<std::uint64_t>>& neighbors) {
  std::vector<std::uint64_t> identity;
  identity.reserve(neighbors.size());
  for (std::uint64_t u = 0; u < neighbors.size(); ++u) {
    identity.push_back(u);
  }

  bool identity_is_path = true;
  for (std::size_t i = 0; i + 1 < identity.size(); ++i) {
    if (!are_neighbors(neighbors, identity[i], identity[i + 1])) {
      identity_is_path = false;
      break;
    }
  }
  if (identity_is_path) {
    return identity;
  }

  auto greedy_path = greedy_hamiltonian_path(neighbors);
  if (!greedy_path.empty()) {
    return greedy_path;
  }

  auto exact_path = bounded_hamiltonian_path(neighbors);
  if (!exact_path.empty()) {
    return exact_path;
  }

  // A Hamiltonian path is not guaranteed for arbitrary custom lattices. The
  // encoding still has a well-defined JW order; it simply treats every lattice
  // edge that is not consecutive in this traversal as an auxiliary edge.
  return dfs_vertex_order(neighbors);
}

std::set<std::pair<std::uint64_t, std::uint64_t>> path_edges_from_order(
    const std::vector<std::vector<std::uint64_t>>& neighbors,
    const std::vector<std::uint64_t>& order) {
  std::set<std::pair<std::uint64_t, std::uint64_t>> path_edges;
  for (std::size_t i = 0; i + 1 < order.size(); ++i) {
    auto u = order[i];
    auto v = order[i + 1];
    if (are_neighbors(neighbors, u, v)) {
      path_edges.insert(canonical_edge(u, v));
    }
  }
  return path_edges;
}

std::set<std::pair<std::uint64_t, std::uint64_t>> identity_path_edges(
    const std::vector<std::vector<std::uint64_t>>& neighbors) {
  std::set<std::pair<std::uint64_t, std::uint64_t>> path_edges;
  for (std::uint64_t u = 0; u + 1 < neighbors.size(); ++u) {
    if (are_neighbors(neighbors, u, u + 1)) {
      path_edges.insert({u, u + 1});
    }
  }
  return path_edges;
}

std::uint64_t find_root(std::vector<std::uint64_t>& parent,
                        std::uint64_t node) {
  while (parent[node] != node) {
    parent[node] = parent[parent[node]];
    node = parent[node];
  }
  return node;
}

std::set<std::pair<std::uint64_t, std::uint64_t>> colored_linear_forest_edges(
    const EdgeColoring& coloring, std::uint64_t num_sites) {
  struct Candidate {
    int color;
    std::pair<std::uint64_t, std::uint64_t> edge;
  };

  std::vector<Candidate> candidates;
  for (const auto& [edge, color] : coloring) {
    if (color <= 1) {
      candidates.push_back({color, edge});
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.color != b.color) {
                return a.color < b.color;
              }
              return a.edge < b.edge;
            });

  std::vector<std::uint64_t> parent(num_sites);
  std::vector<std::size_t> degree(num_sites, 0);
  for (std::uint64_t u = 0; u < num_sites; ++u) {
    parent[u] = u;
  }

  std::set<std::pair<std::uint64_t, std::uint64_t>> path_edges;
  for (const auto& candidate : candidates) {
    auto [u, v] = candidate.edge;
    if (u >= num_sites || v >= num_sites || degree[u] >= 2 ||
        degree[v] >= 2) {
      continue;
    }

    auto root_u = find_root(parent, u);
    auto root_v = find_root(parent, v);
    if (root_u == root_v) {
      continue;
    }

    parent[root_u] = root_v;
    ++degree[u];
    ++degree[v];
    path_edges.insert(candidate.edge);
  }
  return path_edges;
}

std::set<std::pair<std::uint64_t, std::uint64_t>> choose_vc_path_edges(
    const LatticeGraph& lattice,
    const std::vector<std::vector<std::uint64_t>>& neighbors) {
  auto selected = identity_path_edges(neighbors);

  if (lattice.edge_coloring().has_value()) {
    auto colored = colored_linear_forest_edges(*lattice.edge_coloring(),
                                               lattice.num_sites());
    if (colored.size() > selected.size()) {
      selected = std::move(colored);
    }
  }

  if (selected.size() >= neighbors.size() / 2) {
    return selected;
  }

  auto graph_order = choose_vc_path_order(neighbors);
  auto graph_edges = path_edges_from_order(neighbors, graph_order);
  if (graph_edges.size() > selected.size()) {
    return graph_edges;
  }
  return selected;
}

std::vector<std::uint64_t> order_from_linear_forest(
    std::uint64_t num_sites,
    const std::set<std::pair<std::uint64_t, std::uint64_t>>& path_edges) {
  std::vector<std::vector<std::uint64_t>> forest_neighbors(num_sites);
  for (auto [u, v] : path_edges) {
    forest_neighbors[u].push_back(v);
    forest_neighbors[v].push_back(u);
  }
  for (auto& neighbors : forest_neighbors) {
    std::sort(neighbors.begin(), neighbors.end());
  }

  std::vector<bool> visited(num_sites, false);
  std::vector<std::uint64_t> order;
  order.reserve(num_sites);

  auto append_component = [&](std::uint64_t start) {
    std::uint64_t prev = num_sites;
    std::uint64_t current = start;
    while (current < num_sites && !visited[current]) {
      visited[current] = true;
      order.push_back(current);

      std::uint64_t next = num_sites;
      for (auto candidate : forest_neighbors[current]) {
        if (candidate != prev && !visited[candidate]) {
          next = candidate;
          break;
        }
      }
      prev = current;
      current = next;
    }
  };

  for (std::uint64_t u = 0; u < num_sites; ++u) {
    if (!visited[u] && forest_neighbors[u].size() == 1) {
      append_component(u);
    }
  }
  for (std::uint64_t u = 0; u < num_sites; ++u) {
    if (!visited[u]) {
      append_component(u);
    }
  }
  return order;
}

std::vector<std::uint64_t> shortest_path_excluding_edge(
    const std::vector<std::vector<std::uint64_t>>& neighbors,
    std::uint64_t source, std::uint64_t target,
    std::pair<std::uint64_t, std::uint64_t> excluded_edge) {
  const auto n = neighbors.size();
  std::vector<bool> visited(n, false);
  std::vector<std::uint64_t> parent(n, static_cast<std::uint64_t>(n));
  std::queue<std::uint64_t> queue;

  visited[source] = true;
  queue.push(source);

  while (!queue.empty() && !visited[target]) {
    std::uint64_t u = queue.front();
    queue.pop();
    for (std::uint64_t v : neighbors[u]) {
      if (canonical_edge(u, v) == excluded_edge || visited[v]) {
        continue;
      }
      visited[v] = true;
      parent[v] = u;
      queue.push(v);
      if (v == target) {
        break;
      }
    }
  }

  if (!visited[target]) {
    return {};
  }

  std::vector<std::uint64_t> path;
  for (std::uint64_t u = target; u != source; u = parent[u]) {
    path.push_back(u);
  }
  path.push_back(source);
  std::reverse(path.begin(), path.end());
  return path;
}

StabilizerTerm multiply_stabilizer_terms(
    const std::vector<StabilizerTerm>& stabilizers,
    const std::vector<std::size_t>& indices) {
  std::complex<double> coeff{1.0, 0.0};
  SparsePauliWord word;

  for (std::size_t idx : indices) {
    auto [phase, product] =
        PauliTermAccumulator::multiply_uncached(word, stabilizers[idx].second);
    coeff *= phase * stabilizers[idx].first;
    word = std::move(product);
  }

  return {coeff, std::move(word)};
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
    const LatticeGraph& lattice, std::size_t spin_species) {
  using namespace detail;
  const std::string name = "verstraete-cirac";

  if (spin_species != 1 && spin_species != 2) {
    throw std::invalid_argument(name +
                                " requires spin_species to be either 1 or 2");
  }

  std::uint64_t V = lattice.num_sites();
  if (V < 3) {
    throw std::invalid_argument(
        name + " requires a lattice graph with at least 3 sites");
  }

  if (!lattice.is_symmetric()) {
    throw std::invalid_argument(name +
                                " requires an undirected/symmetric lattice");
  }

  const auto& adj = lattice.sparse_adjacency_matrix();
  std::vector<std::vector<std::uint64_t>> lattice_neighbors(V);
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      std::uint64_t u = it.row();
      std::uint64_t v = it.col();
      if (u != v && it.value() != 0.0) {
        lattice_neighbors[u].push_back(v);
      }
    }
  }
  for (auto& neighbors : lattice_neighbors) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }

  const auto path_edges = choose_vc_path_edges(lattice, lattice_neighbors);
  const auto path_order = order_from_linear_forest(V, path_edges);

  std::vector<std::vector<std::uint64_t>> non_adjacent_incident(V);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> non_adjacent_edges;
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      std::uint64_t u = it.row();
      std::uint64_t v = it.col();
      if (u < v && it.value() != 0.0) {
        auto edge = canonical_edge(u, v);
        if (path_edges.count(edge) == 0) {
          non_adjacent_incident[u].push_back(v);
          non_adjacent_incident[v].push_back(u);
          non_adjacent_edges.emplace_back(edge);
        }
      }
    }
  }
  for (auto& incident : non_adjacent_incident) {
    std::sort(incident.begin(), incident.end());
    incident.erase(std::unique(incident.begin(), incident.end()),
                   incident.end());
  }

  // Count aux fermionic modes per site
  std::vector<std::size_t> n_aux(V);
  std::size_t total_n_aux = 0;
  for (std::uint64_t u = 0; u < V; ++u) {
    n_aux[u] = (non_adjacent_incident[u].size() + 1) / 2;
    total_n_aux += n_aux[u];
  }

  std::size_t modes_per_species = V + total_n_aux;
  std::size_t num_modes = spin_species * V;
  std::size_t base_qubits = spin_species * modes_per_species;

  auto jw_base = MajoranaMapping::jordan_wigner(base_qubits);

  // Precompute mode offsets along the graph-derived sequence of sites while
  // keeping public physical mode labels unchanged.
  std::vector<std::size_t> site_to_mode_offset(V);
  std::size_t mode_offset = 0;
  for (auto site : path_order) {
    site_to_mode_offset[site] = mode_offset;
    mode_offset += 1 + n_aux[site];
  }

  auto get_sys_majorana = [&](std::uint64_t site, std::size_t spin,
                              std::size_t offset) -> std::size_t {
    std::size_t mode_idx =
        spin * modes_per_species + site_to_mode_offset[site];
    return 2 * mode_idx + offset;
  };

  auto get_aux_majorana = [&](std::uint64_t site, std::size_t spin,
                              std::size_t offset) -> std::size_t {
    std::size_t mode_idx =
        spin * modes_per_species + site_to_mode_offset[site];
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
      edge_to_stab_idx(spin_species);

  for (std::size_t spin = 0; spin < spin_species; ++spin) {
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

  // Use products of explicit auxiliary link stabilizers around lattice cycles
  // as the automatic auxiliary penalty terms. In this reduced construction,
  // path-local cycle edges do not carry auxiliary links, so a valid cycle
  // product can contain a single explicit link.
  std::vector<StabilizerTerm> auxiliary_penalty_terms;
  std::set<std::vector<std::size_t>> seen_cycle_products;
  for (std::size_t spin = 0; spin < spin_species; ++spin) {
    for (const auto& edge : non_adjacent_edges) {
      auto path = shortest_path_excluding_edge(lattice_neighbors, edge.first,
                                               edge.second, edge);
      if (path.size() < 2) {
        continue;
      }

      std::vector<std::size_t> cycle_indices;
      auto add_cycle_edge = [&](std::uint64_t u, std::uint64_t v) {
        auto it = edge_to_stab_idx[spin].find({u, v});
        if (it == edge_to_stab_idx[spin].end()) {
          return;
        }
        if (std::find(cycle_indices.begin(), cycle_indices.end(),
                      it->second) == cycle_indices.end()) {
          cycle_indices.push_back(it->second);
        }
      };

      add_cycle_edge(edge.first, edge.second);
      for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        add_cycle_edge(path[i], path[i + 1]);
      }

      if (cycle_indices.empty()) {
        continue;
      }

      auto canonical_indices = cycle_indices;
      std::sort(canonical_indices.begin(), canonical_indices.end());
      if (!seen_cycle_products.insert(canonical_indices).second) {
        continue;
      }

      auxiliary_penalty_terms.push_back(
          multiply_stabilizer_terms(stabilizers, cycle_indices));
    }
  }

  // Build a lookup table of VC bilinears by dressing non-local JW bilinears
  // with their stabilizer to make everything local
  std::size_t num_physical_modes = spin_species * V;
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
        std::size_t p = get_sys_majorana(i, s_u, a);
        std::size_t q = get_sys_majorana(j, s_v, b);
        auto [coeff, word] = get_bilinear(p, q);

        // For non-local paths, the raw JW bilinear iγ_pγ_q loses its long
        // Z-string by multiplying by edge stabilizer iγ̃_aγ̃_b
        auto key = std::make_pair(std::min(i, j), std::max(i, j));
        if (path_edges.count(key) == 0) {
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
                         std::move(stabilizers),
                         std::move(auxiliary_penalty_terms));
}

}  // namespace qdk::chemistry::data
