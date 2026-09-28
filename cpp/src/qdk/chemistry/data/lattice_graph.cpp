// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Sparse>
#include <algorithm>
#include <blas.hh>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "json_serialization.hpp"

namespace qdk::chemistry::data {

namespace detail {
using Triplet = Eigen::Triplet<double>;

// Graph files written before serialization versioning have no version.
constexpr const char* kUnversionedGraphMessage =
    "LatticeGraph data has no serialization version. If this file was written "
    "by an older qdk-chemistry release, migrate it with: python -m "
    "qdk_chemistry.migrate <old_file> <new_file>.";

static EdgeColoring color_edges(
    std::uint64_t num_sites,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges_in,
    int seed, int trials);

// Helper: add an undirected edge (i, j) with weight t to the triplet list.
static void add_edge(std::vector<Triplet>& triplets, int i, int j, double t) {
  triplets.emplace_back(i, j, t);
  triplets.emplace_back(j, i, t);
}

// An axis and its negation describe one unoriented bond class. The first
// component exceeding the tolerance selects the representative sign.
static void orient_axis(Eigen::RowVectorXd& axis, double tolerance) {
  for (Eigen::Index k = 0; k < axis.size(); ++k) {
    if (std::abs(axis[k]) > tolerance) {
      if (axis[k] < 0.0) axis = -axis;
      return;
    }
  }
}

static std::vector<BondFlavorDefinition> prepare_flavors(
    std::vector<BondFlavorDefinition> definitions, double tolerance) {
  for (auto& definition : definitions) {
    if (definition.shell == 0 || definition.axis.size() == 0 ||
        !definition.axis.allFinite() ||
        definition.axis.cwiseAbs().maxCoeff() == 0.0) {
      throw std::invalid_argument(
          "Bond flavors require a positive shell and a finite nonzero axis.");
    }
    if (definition.axis.size() != 2) {
      throw std::invalid_argument("Bond-flavor axes must be two-dimensional.");
    }
    definition.axis /=
        blas::nrm2(definition.axis.size(), definition.axis.data(), 1);
    if (!definition.axis.allFinite() ||
        std::abs(definition.axis.squaredNorm() - 1.0) > tolerance) {
      throw std::invalid_argument("Bond-flavor axis normalization failed.");
    }
    orient_axis(definition.axis, tolerance);
  }
  std::sort(definitions.begin(), definitions.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.shell != rhs.shell) return lhs.shell < rhs.shell;
              return std::lexicographical_compare(
                  lhs.axis.data(), lhs.axis.data() + lhs.axis.size(),
                  rhs.axis.data(), rhs.axis.data() + rhs.axis.size());
            });
  for (std::size_t i = 0; i < definitions.size(); ++i) {
    for (std::size_t j = i + 1;
         j < definitions.size() && definitions[j].shell == definitions[i].shell;
         ++j) {
      const Eigen::RowVectorXd difference =
          definitions[i].axis - definitions[j].axis;
      if (blas::nrm2(difference.size(), difference.data(), 1) <= tolerance) {
        throw std::invalid_argument(
            "Each shell-axis class may have only one bond flavor.");
      }
    }
  }
  return definitions;
}

// Geometry bond axes already have the orient_axis representative sign.
static std::optional<BondFlavorId> flavor_of(
    const std::vector<BondFlavorDefinition>& definitions, std::uint64_t shell,
    const Eigen::RowVector2d& axis, double tolerance) {
  for (const auto& definition : definitions) {
    if (definition.shell != shell) continue;
    const Eigen::RowVector2d difference = definition.axis - axis;
    if (blas::nrm2(2, difference.data(), 1) <= tolerance) {
      return definition.flavor;
    }
  }
  return std::nullopt;
}

template <typename Integer>
Integer json_integer(const nlohmann::json& value) {
  bool valid = value.is_number_integer();
  if (valid && value.is_number_unsigned()) {
    valid = value.get<std::uint64_t>() <=
            static_cast<std::uint64_t>(std::numeric_limits<Integer>::max());
  } else if (valid) {
    const auto integer = value.get<std::int64_t>();
    if constexpr (std::is_unsigned_v<Integer>) {
      valid = integer >= 0 && static_cast<std::uint64_t>(integer) <=
                                  std::numeric_limits<Integer>::max();
    } else {
      valid = integer >= std::numeric_limits<Integer>::min() &&
              integer <= std::numeric_limits<Integer>::max();
    }
  }
  if (!valid) {
    throw std::invalid_argument(
        "Lattice JSON integer is invalid or out of range.");
  }
  return value.get<Integer>();
}

/**
 * @brief Depth-first search (DFS) helper to find a Hamiltonian path in a sparse
 * graph.
 *
 * Recursively visits unvisited neighbor vertices to build a path that visits
 * every vertex in the graph exactly once.
 *
 * @param curr    The vertex index currently being visited.
 * @param adj     The sparse adjacency matrix of the graph.
 * @param visited Tracks which vertices have already been visited.
 * @param path    Stores the sequence of vertices in the current path.
 * @return True if a Hamiltonian path is found, false otherwise.
 */
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

/**
 * @brief Search for a Hamiltonian path in the given sparse graph.
 *
 * Tries to find a path that visits every vertex exactly once, starting the
 * search from each possible vertex in the graph.
 *
 * @param adj The sparse adjacency matrix representing the graph.
 * @return A vector of vertex indices in path order, or an empty vector if no
 * path exists.
 */
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

LatticeGraph::LatticeGraph(
    const std::map<std::pair<std::uint64_t, std::uint64_t>, double>&
        edge_weights,
    std::uint64_t num_sites, EdgeLabels edge_labels) {
  // get num_sites if not provided
  if (num_sites == 0) {
    for (const auto& [edge, weight] : edge_weights) {
      const auto& [i, j] = edge;
      if (i + 1 > num_sites) num_sites = i + 1;
      if (j + 1 > num_sites) num_sites = j + 1;
    }
  }
  _num_sites = num_sites;

  // build triplet list
  std::vector<detail::Triplet> triplets;
  triplets.reserve(edge_weights.size());
  for (const auto& [edge, weight] : edge_weights) {
    const auto& [i, j] = edge;
    if (i >= _num_sites || j >= _num_sites) {
      throw std::invalid_argument("Edge (" + std::to_string(i) + ", " +
                                  std::to_string(j) +
                                  ") has index out of range for num_sites=" +
                                  std::to_string(_num_sites) + ".");
    }
    triplets.emplace_back(static_cast<int>(i), static_cast<int>(j), weight);
  }

  // build sparse adjacency matrix
  auto n = static_cast<Eigen::Index>(_num_sites);
  adjacency_.resize(n, n);
  adjacency_.setFromTriplets(triplets.begin(), triplets.end());
  adjacency_.makeCompressed();
  _is_symmetric = _check_symmetry(adjacency_);
  _edge_labels = std::move(edge_labels);
  _validate_edge_labels();
}

LatticeGraph::LatticeGraph(Eigen::SparseMatrix<double> adjacency,
                           std::optional<EdgeColoring> coloring,
                           EdgeLabels edge_labels)
    : _num_sites(static_cast<std::uint64_t>(adjacency.rows())),
      adjacency_(std::move(adjacency)),
      _is_symmetric(_check_symmetry(adjacency_)),
      _edge_coloring(std::move(coloring)),
      _edge_labels(std::move(edge_labels)) {
  _validate_coloring();
  _validate_edge_labels();
}

void LatticeGraph::_validate_coloring() const {
  if (!_edge_coloring.has_value()) return;
  std::set<std::pair<std::uint64_t, int>> used;
  for (const auto& [edge, color] : *_edge_coloring) {
    if (edge.first >= edge.second || edge.second >= _num_sites || color < 0 ||
        !used.emplace(edge.first, color).second ||
        !used.emplace(edge.second, color).second) {
      throw std::invalid_argument("Invalid lattice edge coloring.");
    }
  }
  std::size_t matched = 0;
  for (int k = 0; k < adjacency_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      if (it.row() >= it.col()) continue;
      const bool colored =
          _edge_coloring->contains({static_cast<std::uint64_t>(it.row()),
                                    static_cast<std::uint64_t>(it.col())});
      if (it.value() != 0.0 && !colored) {
        throw std::invalid_argument("Adjacency edge missing from coloring.");
      }
      matched += colored;
    }
  }
  // Topology colorings may include stored zero-weight edges, but not new edges.
  if (matched != _edge_coloring->size()) {
    throw std::invalid_argument(
        "Coloring contains an edge outside the topology.");
  }
}

void LatticeGraph::_validate_edge_labels() const {
  if (_edge_labels.empty()) return;
  // Every stored upper-triangular pair has exactly one label.
  std::size_t labelled = 0;
  for (int k = 0; k < adjacency_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      if (it.row() >= it.col()) continue;
      if (!_edge_labels.contains({static_cast<std::uint64_t>(it.row()),
                                  static_cast<std::uint64_t>(it.col())})) {
        throw std::invalid_argument("Adjacency edge missing an edge label.");
      }
      ++labelled;
    }
  }
  if (labelled != _edge_labels.size() ||
      std::any_of(_edge_labels.begin(), _edge_labels.end(),
                  [](const auto& item) { return item.second.shell == 0; })) {
    throw std::invalid_argument("Invalid lattice edge label.");
  }
}

LatticeGraph LatticeGraph::from_dense_matrix(
    const Eigen::MatrixXd& adjacency_matrix, EdgeLabels edge_labels) {
  if (adjacency_matrix.rows() != adjacency_matrix.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }
  Eigen::SparseMatrix<double> sparse = adjacency_matrix.sparseView();
  sparse.makeCompressed();
  return LatticeGraph(std::move(sparse), std::nullopt, std::move(edge_labels));
}

LatticeGraph LatticeGraph::from_sparse_matrix(
    const Eigen::SparseMatrix<double>& sparse, EdgeLabels edge_labels) {
  if (sparse.rows() != sparse.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }
  Eigen::SparseMatrix<double> copy = sparse;
  copy.makeCompressed();
  return LatticeGraph(std::move(copy), std::nullopt, std::move(edge_labels));
}

LatticeGraph LatticeGraph::make_bidirectional(const LatticeGraph& graph) {
  Eigen::SparseMatrix<double> sym =
      (graph.adjacency_ +
       Eigen::SparseMatrix<double>(graph.adjacency_.transpose()));
  sym.makeCompressed();
  return LatticeGraph(std::move(sym), std::nullopt, graph._edge_labels);
}

std::uint64_t LatticeGraph::num_sites() const { return _num_sites; }

const Eigen::SparseMatrix<double>& LatticeGraph::sparse_adjacency_matrix()
    const {
  return adjacency_;
}

Eigen::MatrixXd LatticeGraph::adjacency_matrix() const {
  return Eigen::MatrixXd(adjacency_);
}

bool LatticeGraph::is_symmetric() const { return _is_symmetric; }

double LatticeGraph::weight(std::uint64_t i, std::uint64_t j) const {
  return adjacency_.coeff(static_cast<Eigen::Index>(i),
                          static_cast<Eigen::Index>(j));
}

bool LatticeGraph::are_connected(std::uint64_t i, std::uint64_t j) const {
  return weight(i, j) != 0.0;
}

std::uint64_t LatticeGraph::num_nonzeros() const {
  return static_cast<std::uint64_t>(adjacency_.nonZeros());
}

std::uint64_t LatticeGraph::num_edges() const {
  std::uint64_t count = 0;
  for (int k = 0; k < adjacency_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      if (it.row() < it.col()) count++;
    }
  }
  return count;
}

const EdgeLabels& LatticeGraph::edge_labels() const { return _edge_labels; }

LatticeGraph LatticeGraph::from_geometry(
    const LatticeGeometry& geometry, const std::vector<std::uint64_t>& shells,
    const std::vector<BondFlavorDefinition>& definitions, double weight,
    double tolerance) {
  if (!std::isfinite(weight)) {
    throw std::invalid_argument("Connection weight must be finite.");
  }
  const auto bonds = geometry._shell_bonds(shells, tolerance);
  const auto flavors = detail::prepare_flavors(definitions, tolerance);
  EdgeLabels labels;
  std::vector<detail::Triplet> triplets;
  triplets.reserve(2 * bonds.size());
  for (const auto& bond : bonds) {
    const auto i = bond.site_i;
    const auto j = bond.site_j;
    if (i == j) {
      throw std::invalid_argument(
          "A lattice edge cannot join a site to its own periodic image.");
    }
    // Geometry records order endpoints with site_i < site_j.
    if (!labels
             .try_emplace(
                 {i, j}, bond.shell,
                 detail::flavor_of(flavors, bond.shell, bond.axis, tolerance))
             .second) {
      throw std::invalid_argument(
          "Several periodic images join sites " + std::to_string(i) + " and " +
          std::to_string(j) + "; enlarge the periodic lattice.");
    }
    detail::add_edge(triplets, static_cast<int>(i), static_cast<int>(j),
                     weight);
  }
  const auto n = static_cast<Eigen::Index>(geometry.num_sites());
  Eigen::SparseMatrix<double> adjacency(n, n);
  adjacency.setFromTriplets(triplets.begin(), triplets.end());
  adjacency.makeCompressed();
  // Color labelled pairs, not weights: shell couplings ignore edge weights.
  // Match the native sparse-adjacency traversal before shuffled trials.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> pairs;
  pairs.reserve(labels.size());
  for (const auto& [pair, label] : labels) pairs.push_back(pair);
  std::sort(pairs.begin(), pairs.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.second, lhs.first) < std::tie(rhs.second, rhs.first);
  });
  auto coloring = detail::color_edges(geometry.num_sites(), pairs, 0, 32);
  return LatticeGraph(std::move(adjacency), std::move(coloring),
                      std::move(labels));
}

LatticeGraph LatticeGraph::chain(std::uint64_t n, bool periodic, double t,
                                 bool dfs_ordering) {
  if (n == 0) {
    throw std::invalid_argument("chain: n must be > 0.");
  }

  auto N = static_cast<int>(n);
  std::vector<detail::Triplet> triplets;
  triplets.reserve(2 * N);

  // chain
  for (int i = 0; i < N - 1; ++i) {
    detail::add_edge(triplets, i, i + 1, t);
  }

  // periodic boundary
  if (periodic && N > 2) {
    detail::add_edge(triplets, N - 1, 0, t);
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  LatticeGraph g(std::move(adj),
                 chain_coloring(static_cast<std::int64_t>(N), periodic));
  if (dfs_ordering) {
    auto path = detail::find_hamiltonian_path(g.sparse_adjacency_matrix());
    if (!path.empty()) {
      return permute(g, path);
    } else {
      throw std::runtime_error(
          "No Hamiltonian path found in the lattice graph.");
    }
  }
  return g;
}

LatticeGraph LatticeGraph::square(std::uint64_t nx, std::uint64_t ny,
                                  bool periodic_x, bool periodic_y, double t,
                                  bool dfs_ordering) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("square: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("square: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("square: periodic_y requires ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = Nx * Ny;

  // Helper to convert (x, y) coordinates to site index
  auto idx = [Nx](int x, int y) { return y * Nx + x; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(4 * N);

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Right neighbour
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idx(x, y), idx(x + 1, y), t);
        // periodic boundary
      } else if (periodic_x) {
        detail::add_edge(triplets, idx(x, y), idx(0, y), t);
      }
      // Upper neighbour
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y), idx(x, y + 1), t);
        // periodic boundary
      } else if (periodic_y) {
        detail::add_edge(triplets, idx(x, y), idx(x, 0), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  LatticeGraph g(std::move(adj),
                 square_coloring(Nx, Ny, periodic_x, periodic_y));
  if (dfs_ordering) {
    auto path = detail::find_hamiltonian_path(g.sparse_adjacency_matrix());
    if (!path.empty()) {
      return permute(g, path);
    } else {
      throw std::runtime_error(
          "No Hamiltonian path found in the lattice graph.");
    }
  }
  return g;
}

LatticeGraph LatticeGraph::triangular(std::uint64_t nx, std::uint64_t ny,
                                      bool periodic_x, bool periodic_y,
                                      double t, int coloring_seed,
                                      bool dfs_ordering) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("triangular: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("triangular: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("triangular: periodic_y requires ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = Nx * Ny;

  auto idx = [Nx](int x, int y) { return y * Nx + x; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(6 * N);

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Right neighbour
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idx(x, y), idx(x + 1, y), t);
      } else if (periodic_x) {
        detail::add_edge(triplets, idx(x, y), idx(0, y), t);
      }
      // Upper neighbour
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y), idx(x, y + 1), t);
      } else if (periodic_y) {
        detail::add_edge(triplets, idx(x, y), idx(x, 0), t);
      }
      // Diagonal neighbour (upper-right)
      if (x + 1 < Nx && y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y), idx(x + 1, y + 1), t);
      } else if (x + 1 >= Nx && y + 1 < Ny && periodic_x) {
        // x wraps, y does not
        detail::add_edge(triplets, idx(x, y), idx(0, y + 1), t);
      } else if (x + 1 < Nx && y + 1 >= Ny && periodic_y) {
        // y wraps, x does not
        detail::add_edge(triplets, idx(x, y), idx(x + 1, 0), t);
      } else if (x + 1 >= Nx && y + 1 >= Ny && periodic_x && periodic_y) {
        // both wrap (corner)
        detail::add_edge(triplets, idx(x, y), idx(0, 0), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  // No known deterministic coloring for triangular lattices with arbitrary
  // periodic boundaries; use greedy with multiple trials instead.
  auto coloring = greedy_edge_coloring(adj, coloring_seed, 32);
  LatticeGraph g(std::move(adj), std::move(coloring));
  if (dfs_ordering) {
    auto path = detail::find_hamiltonian_path(g.sparse_adjacency_matrix());
    if (!path.empty()) {
      return permute(g, path);
    } else {
      throw std::runtime_error(
          "No Hamiltonian path found in the lattice graph.");
    }
  }
  return g;
}

LatticeGraph LatticeGraph::honeycomb(std::uint64_t nx, std::uint64_t ny,
                                     bool periodic_x, bool periodic_y, double t,
                                     bool dfs_ordering) {
  (void)dfs_ordering;
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("honeycomb: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("honeycomb: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("honeycomb: periodic_y requires ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = 2 * Nx * Ny;  // 2 sites per unit cell

  // Site indices within unit cell (x, y):
  //   A = 2 * (y * Nx + x),  B = 2 * (y * Nx + x) + 1
  auto idxA = [Nx](int x, int y) { return 2 * (y * Nx + x); };
  auto idxB = [Nx](int x, int y) { return 2 * (y * Nx + x) + 1; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(3 * N);

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Intra-cell bond: A -- B
      detail::add_edge(triplets, idxA(x, y), idxB(x, y), t);

      // Inter-cell bond 1: B(x,y) -- A(x+1, y)  (horizontal)
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idxB(x, y), idxA(x + 1, y), t);
      } else if (periodic_x) {
        detail::add_edge(triplets, idxB(x, y), idxA(0, y), t);
      }

      // Inter-cell bond 2: B(x,y) -- A(x, y+1)  (vertical)
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idxB(x, y), idxA(x, y + 1), t);
      } else if (periodic_y) {
        detail::add_edge(triplets, idxB(x, y), idxA(x, 0), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  return LatticeGraph(std::move(adj),
                      honeycomb_coloring(Nx, Ny, periodic_x, periodic_y));
}

LatticeGraph LatticeGraph::kagome(std::uint64_t nx, std::uint64_t ny,
                                  bool periodic_x, bool periodic_y, double t,
                                  int coloring_seed, bool dfs_ordering) {
  (void)dfs_ordering;
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("kagome: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("kagome: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("kagome: periodic_y requires ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = 3 * Nx * Ny;  // 3 sites per unit cell

  // Layout per unit cell:
  //   s0 -- s1  (horizontal edge, bottom of up-triangle)
  //   s0 -- s2  (left edge of up-triangle)
  //   s1 -- s2  (right edge of up-triangle)
  // Inter-cell bonds form the down-triangles.
  auto idx = [Nx](int x, int y, int s) { return 3 * (y * Nx + x) + s; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(6 * N);  // 4 edges per site, stored as pairs

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Intra-cell (up-triangle) edges
      detail::add_edge(triplets, idx(x, y, 0), idx(x, y, 1), t);
      detail::add_edge(triplets, idx(x, y, 0), idx(x, y, 2), t);
      detail::add_edge(triplets, idx(x, y, 1), idx(x, y, 2), t);

      // Inter-cell edges (down-triangle connections)
      // s1(x,y) -- s0(x+1, y)  (horizontal, right)
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idx(x, y, 1), idx(x + 1, y, 0), t);
      } else if (periodic_x) {
        detail::add_edge(triplets, idx(x, y, 1), idx(0, y, 0), t);
      }

      // s2(x,y) -- s0(x, y+1)  (vertical, up)
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y, 2), idx(x, y + 1, 0), t);
      } else if (periodic_y) {
        detail::add_edge(triplets, idx(x, y, 2), idx(x, 0, 0), t);
      }

      // s2(x,y) -- s1(x-1, y+1)  (diagonal, upper-left)
      if (x - 1 >= 0 && y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y, 2), idx(x - 1, y + 1, 1), t);
      } else if (x - 1 < 0 && y + 1 < Ny && periodic_x) {
        // x wraps, y does not
        int xl = (x - 1 + Nx) % Nx;
        detail::add_edge(triplets, idx(x, y, 2), idx(xl, y + 1, 1), t);
      } else if (x - 1 >= 0 && y + 1 >= Ny && periodic_y) {
        // y wraps, x does not
        detail::add_edge(triplets, idx(x, y, 2), idx(x - 1, 0, 1), t);
      } else if (x - 1 < 0 && y + 1 >= Ny && periodic_x && periodic_y) {
        // both wrap (corner)
        int xl = (x - 1 + Nx) % Nx;
        detail::add_edge(triplets, idx(x, y, 2), idx(xl, 0, 1), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  auto coloring = greedy_edge_coloring(adj, coloring_seed, 32);
  return LatticeGraph(std::move(adj), std::move(coloring));
}

namespace detail {

// Collect every undirected edge (i, j) with i < j from the adjacency matrix.
std::vector<std::pair<std::uint64_t, std::uint64_t>> undirected_edges(
    const Eigen::SparseMatrix<double>& adj) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  edges.reserve(static_cast<std::size_t>(adj.nonZeros()) / 2);
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      if (it.row() < it.col() && it.value() != 0.0) {
        edges.emplace_back(static_cast<std::uint64_t>(it.row()),
                           static_cast<std::uint64_t>(it.col()));
      }
    }
  }
  return edges;
}

// Greedy edge coloring: place each edge in the lowest-index color whose
// vertices do not already touch that color.  Optionally retry with shuffled
// edge orders and keep the result with fewest colors.
static EdgeColoring color_edges(
    std::uint64_t num_sites,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges_in,
    int seed, int trials) {
  if (edges_in.empty() || trials < 1) {
    return {};
  }

  // Compute max degree to bound the colour count.
  auto num_vertices = static_cast<std::size_t>(num_sites);
  std::vector<int> degree(num_vertices, 0);
  for (const auto& [u, v] : edges_in) {
    ++degree[u];
    ++degree[v];
  }
  int max_degree = *std::max_element(degree.begin(), degree.end());
  int max_colors = 2 * max_degree;  // upper bound: 2*Δ - 1 rounded up

  EdgeColoring best;
  int best_count = std::numeric_limits<int>::max();
  std::mt19937 rng(static_cast<std::uint32_t>(seed));

  std::vector<std::size_t> order(edges_in.size());
  std::iota(order.begin(), order.end(), 0);

  for (int trial = 0; trial < trials; ++trial) {
    if (trial > 0) {
      std::shuffle(order.begin(), order.end(), rng);
    }

    EdgeColoring coloring;
    // For each vertex, a bitset of colours already incident to it.
    std::vector<std::vector<bool>> vertex_used(
        num_vertices, std::vector<bool>(max_colors, false));
    int max_color = -1;

    for (std::size_t pos : order) {
      const auto& edge = edges_in[pos];
      const auto& used_i = vertex_used[edge.first];
      const auto& used_j = vertex_used[edge.second];
      int chosen = 0;
      while (chosen < max_colors && (used_i[chosen] || used_j[chosen])) {
        ++chosen;
      }
      coloring[edge] = chosen;
      vertex_used[edge.first][chosen] = true;
      vertex_used[edge.second][chosen] = true;
      if (chosen > max_color) max_color = chosen;
    }

    int distinct = max_color + 1;
    if (distinct < best_count) {
      best_count = distinct;
      best = std::move(coloring);
    }
  }
  return best;
}

}  // namespace detail

EdgeColoring greedy_edge_coloring(const Eigen::SparseMatrix<double>& adj,
                                  int seed, int trials) {
  return detail::color_edges(static_cast<std::uint64_t>(adj.rows()),
                             detail::undirected_edges(adj), seed, trials);
}

// Deterministic two-coloring of an open chain: edge (i, i+1) gets color i % 2.
// For periodic chains, even N keeps two colors, odd N requires a third for
// the wrap edge to satisfy the no-incident-same-color constraint.
EdgeColoring chain_coloring(std::int64_t n, bool periodic) {
  EdgeColoring out;
  for (std::int64_t i = 0; i + 1 < n; ++i) {
    out[{static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(i + 1)}] =
        i % 2;
  }
  if (periodic && n > 2) {
    int wrap_color = (n % 2 == 0) ? 1 : 2;  // last edge color is (n-2)%2
    out[{0, static_cast<std::uint64_t>(n - 1)}] = wrap_color;
  }
  return out;
}

// Deterministic edge coloring for the square lattice.  Horizontal and vertical
// edges live on disjoint axes; each axis can be 2-colored by alternating.
// With periodic boundaries, an odd extent on that axis forces a third color
// on its wrap edges.  Total colors: 2 (open) up to 4 (both axes odd-periodic).
EdgeColoring square_coloring(std::int64_t Nx, std::int64_t Ny, bool periodic_x,
                             bool periodic_y) {
  EdgeColoring out;
  auto idx = [Nx](std::int64_t x, std::int64_t y) {
    return static_cast<std::uint64_t>(y * Nx + x);
  };
  auto put = [&out](std::uint64_t a, std::uint64_t b, int c) {
    auto edge = std::minmax(a, b);
    out[{edge.first, edge.second}] = c;
  };

  // Horizontal edges use colors {0, 1}; vertical edges use {2, 3}.  When a
  // periodic dimension has odd extent the wrap edge needs its own color
  // (4 for x-wrap parity-conflict, 5 for y-wrap parity-conflict).
  for (std::int64_t y = 0; y < Ny; ++y) {
    for (std::int64_t x = 0; x + 1 < Nx; ++x) {
      put(idx(x, y), idx(x + 1, y), x % 2);
    }
    if (periodic_x && Nx > 2) {
      int wrap_color = (Nx % 2 == 0) ? 1 : 4;
      put(idx(Nx - 1, y), idx(0, y), wrap_color);
    }
  }
  for (std::int64_t x = 0; x < Nx; ++x) {
    for (std::int64_t y = 0; y + 1 < Ny; ++y) {
      put(idx(x, y), idx(x, y + 1), 2 + y % 2);
    }
    if (periodic_y && Ny > 2) {
      int wrap_color = (Ny % 2 == 0) ? 3 : 5;
      put(idx(x, Ny - 1), idx(x, 0), wrap_color);
    }
  }

  // Compact the color labels so the result is in 0..(distinct-1).
  std::map<int, int> remap;
  for (const auto& [edge, c] : out) {
    remap.emplace(c, static_cast<int>(remap.size()));
  }
  for (auto& [edge, c] : out) {
    c = remap.at(c);
  }
  return out;
}

// Deterministic 3-coloring for honeycomb lattice.  The honeycomb has max
// degree 3 with three structurally distinct bond types: intra-cell (A–B),
// horizontal inter-cell (B–A right), vertical inter-cell (B–A up).
// Each bond type gets its own color, which is valid because no vertex
// is incident to two bonds of the same type.
EdgeColoring honeycomb_coloring(std::int64_t Nx, std::int64_t Ny,
                                bool periodic_x, bool periodic_y) {
  EdgeColoring out;
  auto idxA = [Nx](std::int64_t x, std::int64_t y) -> std::uint64_t {
    return static_cast<std::uint64_t>(2 * (y * Nx + x));
  };
  auto idxB = [Nx](std::int64_t x, std::int64_t y) -> std::uint64_t {
    return static_cast<std::uint64_t>(2 * (y * Nx + x) + 1);
  };
  auto put = [&out](std::uint64_t a, std::uint64_t b, int c) {
    auto edge = std::minmax(a, b);
    out[{edge.first, edge.second}] = c;
  };

  for (std::int64_t y = 0; y < Ny; ++y) {
    for (std::int64_t x = 0; x < Nx; ++x) {
      // Intra-cell: color 0
      put(idxA(x, y), idxB(x, y), 0);
      // Horizontal inter-cell: color 1
      if (x + 1 < Nx) {
        put(idxB(x, y), idxA(x + 1, y), 1);
      } else if (periodic_x) {
        put(idxB(x, y), idxA(0, y), 1);
      }
      // Vertical inter-cell: color 2
      if (y + 1 < Ny) {
        put(idxB(x, y), idxA(x, y + 1), 2);
      } else if (periodic_y) {
        put(idxB(x, y), idxA(x, 0), 2);
      }
    }
  }
  return out;
}

EdgeColoring trivial_edge_coloring(const Eigen::SparseMatrix<double>& adj) {
  EdgeColoring out;
  int color = 0;
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      if (it.row() < it.col() && it.value() != 0.0) {
        out[{static_cast<std::uint64_t>(it.row()),
             static_cast<std::uint64_t>(it.col())}] = color++;
      }
    }
  }
  return out;
}

const std::optional<EdgeColoring>& LatticeGraph::edge_coloring() const {
  return _edge_coloring;
}

bool LatticeGraph::_check_symmetry(const Eigen::SparseMatrix<double>& mat) {
  if (mat.rows() != mat.cols()) {
    return false;
  }
  return mat.isApprox(Eigen::SparseMatrix<double>(mat.transpose()));
}

std::string LatticeGraph::get_summary() const {
  QDK_LOG_TRACE_ENTERING();

  std::ostringstream oss;
  oss << "LatticeGraph Summary:\n";
  oss << "  Sites: " << _num_sites << "\n";
  oss << "  Edges: " << num_edges() << "\n";
  oss << "  Non-zeros: " << num_nonzeros() << "\n";
  oss << "  Symmetric: " << (_is_symmetric ? "true" : "false") << "\n";
  return oss.str();
}

void LatticeGraph::to_file(const std::string& filename,
                           const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

nlohmann::json LatticeGraph::to_json() const {
  QDK_LOG_TRACE_ENTERING();

  // Store adjacency as sparse triplets [row, col, value]
  nlohmann::json edges = nlohmann::json::array();
  for (int k = 0; k < adjacency_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      edges.push_back({it.row(), it.col(), it.value()});
    }
  }

  nlohmann::json j;
  j["version"] = SERIALIZATION_VERSION;
  j["num_sites"] = _num_sites;
  j["is_symmetric"] = _is_symmetric;
  j["adjacency_sparse"] = edges;

  if (_edge_coloring.has_value()) {
    nlohmann::json coloring_json = nlohmann::json::array();
    for (const auto& [edge, color] : *_edge_coloring) {
      coloring_json.push_back({edge.first, edge.second, color});
    }
    j["edge_coloring"] = coloring_json;
  }

  if (!_edge_labels.empty()) {
    nlohmann::json labels = nlohmann::json::array();
    for (const auto& [edge, label] : _edge_labels) {
      labels.push_back({edge.first, edge.second, label.shell,
                        label.flavor ? nlohmann::json(*label.flavor)
                                     : nlohmann::json(nullptr)});
    }
    j["edge_labels"] = labels;
  }

  return j;
}

void LatticeGraph::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  file << to_json().dump(2);
  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

void LatticeGraph::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::DataSpace scalar_space(H5S_SCALAR);

    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    group.createAttribute("version", string_type, scalar_space)
        .write(string_type, std::string(SERIALIZATION_VERSION));

    // Store num_sites as attribute on the group
    H5::Attribute sites_attr = group.createAttribute(
        "num_sites", H5::PredType::NATIVE_UINT64, scalar_space);
    sites_attr.write(H5::PredType::NATIVE_UINT64, &_num_sites);

    // Store is_symmetric as attribute
    hbool_t sym_val = _is_symmetric ? 1 : 0;
    H5::Attribute sym_attr = group.createAttribute(
        "is_symmetric", H5::PredType::NATIVE_HBOOL, scalar_space);
    sym_attr.write(H5::PredType::NATIVE_HBOOL, &sym_val);

    // Write adjacency as sparse dataset: N x 3 (row, col, value)
    auto nnz = static_cast<hsize_t>(adjacency_.nonZeros());
    hsize_t dims[2] = {nnz, 3};
    H5::DataSpace dataspace(2, dims);

    std::vector<double> buffer(nnz * 3);
    hsize_t idx = 0;
    for (int k = 0; k < adjacency_.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
           ++it) {
        buffer[idx * 3 + 0] = static_cast<double>(it.row());
        buffer[idx * 3 + 1] = static_cast<double>(it.col());
        buffer[idx * 3 + 2] = it.value();
        ++idx;
      }
    }

    H5::DataSet dataset = group.createDataSet(
        "adjacency_sparse", H5::PredType::NATIVE_DOUBLE, dataspace);
    if (!buffer.empty()) {
      dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Serialize edge coloring as Nx3 dataset: [i, j, color]
    if (_edge_coloring.has_value()) {
      auto nc = static_cast<hsize_t>(_edge_coloring->size());
      hsize_t cdims[2] = {nc, 3};
      H5::DataSpace cspace(2, cdims);
      std::vector<double> cbuf(nc * 3);
      hsize_t ci = 0;
      for (const auto& [edge, color] : *_edge_coloring) {
        cbuf[ci * 3 + 0] = static_cast<double>(edge.first);
        cbuf[ci * 3 + 1] = static_cast<double>(edge.second);
        cbuf[ci * 3 + 2] = static_cast<double>(color);
        ++ci;
      }
      H5::DataSet cds = group.createDataSet(
          "edge_coloring", H5::PredType::NATIVE_DOUBLE, cspace);
      if (!cbuf.empty()) cds.write(cbuf.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Serialize edge labels as Nx4 dataset: [i, j, shell, flavor or -1]
    if (!_edge_labels.empty()) {
      auto nl = static_cast<hsize_t>(_edge_labels.size());
      hsize_t ldims[2] = {nl, 4};
      H5::DataSpace lspace(2, ldims);
      std::vector<double> lbuf;
      lbuf.reserve(nl * 4);
      for (const auto& [edge, label] : _edge_labels) {
        lbuf.insert(lbuf.end(), {static_cast<double>(edge.first),
                                 static_cast<double>(edge.second),
                                 static_cast<double>(label.shell),
                                 label.flavor ? *label.flavor : -1.0});
      }
      H5::DataSet lds = group.createDataSet(
          "edge_labels", H5::PredType::NATIVE_DOUBLE, lspace);
      lds.write(lbuf.data(), H5::PredType::NATIVE_DOUBLE);
    }
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in LatticeGraph::to_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

void LatticeGraph::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root_group = file.openGroup("/");
    to_hdf5(root_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

LatticeGraph LatticeGraph::from_file(const std::string& filename,
                                     const std::string& type) {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

LatticeGraph LatticeGraph::from_json_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open LatticeGraph JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }
  nlohmann::json json_obj;
  file >> json_obj;
  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }
  return from_json(json_obj);
}

LatticeGraph LatticeGraph::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  if (!j.contains("version")) {
    throw std::runtime_error(detail::kUnversionedGraphMessage);
  }
  validate_serialization_version(SERIALIZATION_VERSION,
                                 j.at("version").get<std::string>());
  if (!j.is_object() || !j.contains("num_sites")) {
    throw std::runtime_error("JSON missing required 'num_sites' field");
  }
  if (!j.contains("adjacency_sparse")) {
    throw std::runtime_error("JSON missing required 'adjacency_sparse' field");
  }

  const auto n = detail::json_integer<std::uint64_t>(j.at("num_sites"));
  if (n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Lattice site count exceeds the sparse index range.");
  }
  const auto n_idx = static_cast<int>(n);

  if (!j.at("adjacency_sparse").is_array()) {
    throw std::invalid_argument("Sparse adjacency must be a triplet array.");
  }
  std::vector<detail::Triplet> triplets;
  for (const auto& entry : j.at("adjacency_sparse")) {
    if (!entry.is_array() || entry.size() != 3) {
      throw std::invalid_argument(
          "Sparse adjacency requires [row, col, weight].");
    }
    const auto row = detail::json_integer<int>(entry[0]);
    const auto col = detail::json_integer<int>(entry[1]);
    if (row < 0 || row >= n_idx || col < 0 || col >= n_idx) {
      throw std::runtime_error("Adjacency index out of range in JSON data.");
    }
    triplets.emplace_back(row, col, entry[2].get<double>());
  }
  Eigen::SparseMatrix<double> sparse(n_idx, n_idx);
  sparse.setFromTriplets(triplets.begin(), triplets.end());
  sparse.makeCompressed();

  std::optional<EdgeColoring> coloring;
  if (j.contains("edge_coloring")) {
    if (!j.at("edge_coloring").is_array()) {
      throw std::invalid_argument("Edge coloring must be a triplet array.");
    }
    coloring.emplace();
    for (const auto& entry : j.at("edge_coloring")) {
      if (!entry.is_array() || entry.size() != 3) {
        throw std::invalid_argument("Edge coloring requires [i, j, color].");
      }
      const auto i = detail::json_integer<std::uint64_t>(entry[0]);
      const auto k = detail::json_integer<std::uint64_t>(entry[1]);
      const auto color = detail::json_integer<int>(entry[2]);
      if (!coloring->emplace(std::pair{i, k}, color).second) {
        throw std::invalid_argument("Duplicate edge in stored coloring.");
      }
    }
  }

  LatticeGraph graph(std::move(sparse), std::move(coloring));
  if (j.contains("edge_labels")) {
    if (!j.at("edge_labels").is_array()) {
      throw std::invalid_argument("Edge labels must be an array.");
    }
    for (const auto& entry : j.at("edge_labels")) {
      if (!entry.is_array() || entry.size() != 4) {
        throw std::invalid_argument(
            "Edge labels require [i, j, shell, flavor].");
      }
      std::optional<BondFlavorId> flavor;
      if (!entry[3].is_null()) {
        flavor = detail::json_integer<BondFlavorId>(entry[3]);
      }
      if (!graph._edge_labels
               .try_emplace({detail::json_integer<std::uint64_t>(entry[0]),
                             detail::json_integer<std::uint64_t>(entry[1])},
                            detail::json_integer<std::uint64_t>(entry[2]),
                            flavor)
               .second) {
        throw std::invalid_argument("Duplicate edge in stored labels.");
      }
    }
  }
  graph._validate_edge_labels();
  return graph;
}

LatticeGraph LatticeGraph::from_hdf5_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open LatticeGraph HDF5 file '" +
                             filename +
                             "'. Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    H5::Group root_group = file.openGroup("/");
    return from_hdf5(root_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "Unable to read LatticeGraph data from HDF5 file '" + filename +
        "'. HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

LatticeGraph LatticeGraph::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    if (!group.attrExists("version")) {
      throw std::runtime_error(detail::kUnversionedGraphMessage);
    }
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string version;
    group.openAttribute("version").read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);
    // Row-major datasets with the fixed column count written by to_hdf5.
    const auto read_rows = [&](const std::string& name, hsize_t columns) {
      const auto dataset = group.openDataSet(name);
      const auto space = dataset.getSpace();
      hsize_t dimensions[2] = {};
      if (space.getSimpleExtentNdims() != 2 ||
          dataset.getTypeClass() != H5T_FLOAT) {
        throw std::invalid_argument("Invalid lattice dataset: " + name);
      }
      space.getSimpleExtentDims(dimensions);
      if (dimensions[1] != columns ||
          dimensions[0] > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid lattice dataset: " + name);
      }
      std::vector<double> buffer(columns * dimensions[0]);
      if (!buffer.empty()) {
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
      }
      return buffer;
    };

    if (!group.attrExists("num_sites")) {
      throw std::runtime_error(
          "HDF5 group missing required 'num_sites' attribute for "
          "LatticeGraph.");
    }
    const auto sites_attr = group.openAttribute("num_sites");
    if (sites_attr.getSpace().getSimpleExtentNdims() != 0 ||
        sites_attr.getTypeClass() != H5T_INTEGER ||
        sites_attr.getIntType().getSign() != H5T_SGN_NONE ||
        sites_attr.getIntType().getSize() > sizeof(std::uint64_t)) {
      throw std::invalid_argument("Invalid lattice site-count attribute.");
    }
    std::uint64_t n = 0;
    sites_attr.read(H5::PredType::NATIVE_UINT64, &n);
    if (n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "Lattice site count exceeds the sparse index range.");
    }
    if (!group.nameExists("adjacency_sparse")) {
      throw std::runtime_error(
          "HDF5 group missing required 'adjacency_sparse' dataset.");
    }
    const auto site_index = [n](double value) {
      if (!std::isfinite(value) || value < 0.0 ||
          value >= static_cast<double>(n) || value != std::trunc(value)) {
        throw std::invalid_argument("Invalid lattice endpoint in HDF5 data.");
      }
      return static_cast<int>(value);
    };
    const auto adjacency_rows = read_rows("adjacency_sparse", 3);
    std::vector<detail::Triplet> triplets;
    triplets.reserve(adjacency_rows.size() / 3);
    for (std::size_t i = 0; i < adjacency_rows.size(); i += 3) {
      triplets.emplace_back(site_index(adjacency_rows[i]),
                            site_index(adjacency_rows[i + 1]),
                            adjacency_rows[i + 2]);
    }
    const auto n_idx = static_cast<Eigen::Index>(n);
    Eigen::SparseMatrix<double> sparse(n_idx, n_idx);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    sparse.makeCompressed();

    std::optional<EdgeColoring> coloring;
    if (group.nameExists("edge_coloring")) {
      coloring.emplace();
      const auto buffer = read_rows("edge_coloring", 3);
      for (std::size_t i = 0; i < buffer.size(); i += 3) {
        const double color = buffer[i + 2];
        if (!std::isfinite(color) || color < 0.0 ||
            color > std::numeric_limits<int>::max() ||
            color != std::trunc(color) ||
            !coloring
                 ->emplace(std::pair{site_index(buffer[i]),
                                     site_index(buffer[i + 1])},
                           static_cast<int>(color))
                 .second) {
          throw std::invalid_argument(
              "Invalid or duplicate stored edge coloring.");
        }
      }
    }

    LatticeGraph graph(std::move(sparse), std::move(coloring));
    if (group.nameExists("edge_labels")) {
      const auto labels = read_rows("edge_labels", 4);
      for (std::size_t i = 0; i < labels.size(); i += 4) {
        const double shell = labels[i + 2];
        const double flavor = labels[i + 3];
        if (!(shell >= 1.0 && shell <= 0x1p53 && shell == std::trunc(shell)) ||
            !(flavor >= -1.0 &&
              flavor <= std::numeric_limits<BondFlavorId>::max() &&
              flavor == std::trunc(flavor)) ||
            !graph._edge_labels
                 .try_emplace(
                     {static_cast<std::uint64_t>(site_index(labels[i])),
                      static_cast<std::uint64_t>(site_index(labels[i + 1]))},
                     static_cast<std::uint64_t>(shell),
                     flavor < 0.0 ? std::nullopt
                                  : std::optional<BondFlavorId>(
                                        static_cast<BondFlavorId>(flavor)))
                 .second) {
          throw std::invalid_argument(
              "Invalid or duplicate stored edge label.");
        }
      }
    }
    graph._validate_edge_labels();
    return graph;
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in LatticeGraph::from_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

void LatticeGraph::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, static_cast<uint64_t>(_num_sites));
  hash_value(ctx, adjacency_);
  hash_value(ctx, _is_symmetric);
  hash_value(ctx, static_cast<std::uint64_t>(_edge_labels.size()));
  for (const auto& [edge, label] : _edge_labels) {
    hash_value(ctx, edge.first);
    hash_value(ctx, edge.second);
    hash_value(ctx, label.shell);
    hash_value(ctx, label.flavor);
  }
}

LatticeGraph LatticeGraph::permute(const LatticeGraph& graph,
                                   const std::vector<std::uint64_t>& path) {
  std::uint64_t V = graph.num_sites();
  if (path.size() != V) {
    throw std::invalid_argument("Permutation must contain every lattice site.");
  }
  std::vector<std::uint64_t> inv_p(V, V);
  for (std::uint64_t i = 0; i < V; ++i) {
    if (path[i] >= V || inv_p[path[i]] != V) {
      throw std::invalid_argument(
          "Permutation must contain each lattice site exactly once.");
    }
    inv_p[path[i]] = i;
  }
  const auto& adj = graph.sparse_adjacency_matrix();

  // Reorder the adjacency matrix using Eigen's PermutationMatrix
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P(
      static_cast<Eigen::Index>(V));
  for (std::uint64_t i = 0; i < V; ++i) {
    P.indices()[static_cast<Eigen::Index>(i)] = static_cast<int>(path[i]);
  }
  Eigen::SparseMatrix<double> new_adj = P.transpose() * adj * P;
  new_adj.makeCompressed();

  std::optional<EdgeColoring> new_coloring = std::nullopt;
  if (graph.edge_coloring().has_value()) {
    EdgeColoring coloring;
    for (const auto& [edge, color] : *(graph.edge_coloring())) {
      std::uint64_t new_u = inv_p[edge.first];
      std::uint64_t new_v = inv_p[edge.second];
      auto new_edge = std::minmax(new_u, new_v);
      coloring[{new_edge.first, new_edge.second}] = color;
    }
    new_coloring = std::move(coloring);
  }

  LatticeGraph result(std::move(new_adj), std::move(new_coloring));
  for (const auto& [edge, label] : graph._edge_labels) {
    const auto new_edge = std::minmax(inv_p[edge.first], inv_p[edge.second]);
    result._edge_labels[{new_edge.first, new_edge.second}] = label;
  }
  return result;
}

}  // namespace qdk::chemistry::data
