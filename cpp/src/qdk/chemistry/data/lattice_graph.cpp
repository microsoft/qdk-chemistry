// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <blas.hh>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
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

#include "hdf5_serialization.hpp"

namespace qdk::chemistry::data {

namespace detail {
using Triplet = Eigen::Triplet<double>;

static EdgeColoring color_edges(
    std::uint64_t num_sites,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges_in,
    int seed, int trials);

// Helper: add an undirected edge (i, j) with weight t to the triplet list.
static void add_edge(std::vector<Triplet>& triplets, int i, int j, double t) {
  triplets.emplace_back(i, j, t);
  triplets.emplace_back(j, i, t);
}

static void normalize_shells(std::vector<std::uint64_t>& shells) {
  if (std::find(shells.begin(), shells.end(), 0) != shells.end()) {
    throw std::invalid_argument("Neighbor shell index must be > 0.");
  }
  std::sort(shells.begin(), shells.end());
  shells.erase(std::unique(shells.begin(), shells.end()), shells.end());
}

static bool connection_less(const NeighborConnection& lhs,
                            const NeighborConnection& rhs) {
  return std::tie(lhs.bond_class.shell, lhs.bond_class.orientation, lhs.site_i,
                  lhs.site_j, lhs.image_shift) <
         std::tie(rhs.bond_class.shell, rhs.bond_class.orientation, rhs.site_i,
                  rhs.site_j, rhs.image_shift);
}

static void canonicalize_connection(NeighborConnection& connection) {
  const auto& image = connection.image_shift;
  if (connection.site_i > connection.site_j ||
      (connection.site_i == connection.site_j &&
       (image[0] < 0 || (image[0] == 0 && image[1] < 0)))) {
    if (image[0] == std::numeric_limits<std::int64_t>::min() ||
        image[1] == std::numeric_limits<std::int64_t>::min()) {
      throw std::overflow_error("Connection image shift cannot be reversed.");
    }
    std::swap(connection.site_i, connection.site_j);
    connection.displacement = -connection.displacement;
    connection.image_shift = {-image[0], -image[1]};
  }
}

static void label_connections(std::vector<NeighborConnection>& connections,
                              std::vector<BondFlavorDefinition> definitions,
                              double tolerance) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw std::invalid_argument("Bond-flavor tolerance must be positive.");
  }
  for (auto& definition : definitions) {
    if (definition.shell == 0 || !definition.axis.allFinite() ||
        definition.axis.cwiseAbs().maxCoeff() == 0.0) {
      throw std::invalid_argument(
          "Bond flavors require a positive shell and a finite nonzero axis.");
    }
    definition.axis /= blas::nrm2(2, definition.axis.data(), 1);
    if (!definition.axis.allFinite() ||
        std::abs(definition.axis.squaredNorm() - 1.0) > tolerance) {
      throw std::invalid_argument("Bond-flavor axis normalization failed.");
    }
    if (definition.axis.x() < -tolerance ||
        (std::abs(definition.axis.x()) <= tolerance &&
         definition.axis.y() < 0.0)) {
      definition.axis = -definition.axis;
    }
  }
  std::sort(definitions.begin(), definitions.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.shell != rhs.shell) return lhs.shell < rhs.shell;
              return std::atan2(lhs.axis.y(), lhs.axis.x()) <
                     std::atan2(rhs.axis.y(), rhs.axis.x());
            });
  for (std::size_t i = 0; i < definitions.size(); ++i) {
    for (std::size_t j = i + 1;
         j < definitions.size() && definitions[j].shell == definitions[i].shell;
         ++j) {
      const Eigen::RowVector2d difference =
          definitions[i].axis - definitions[j].axis;
      if (blas::nrm2(2, difference.data(), 1) <= tolerance) {
        throw std::invalid_argument(
            "Each shell-axis class may have only one bond flavor.");
      }
    }
  }
  for (auto& connection : connections) {
    connection.flavor.reset();
    Eigen::RowVector2d axis = connection.bond_class.axis;
    if (axis.x() < -tolerance ||
        (std::abs(axis.x()) <= tolerance && axis.y() < 0.0)) {
      axis = -axis;
    }
    for (const auto& definition : definitions) {
      if (definition.shell != connection.bond_class.shell) continue;
      const Eigen::RowVector2d difference = definition.axis - axis;
      if (blas::nrm2(2, difference.data(), 1) <= tolerance) {
        connection.flavor = definition.flavor;
        break;
      }
    }
  }
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
    std::uint64_t num_sites) {
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
}

LatticeGraph::LatticeGraph(Eigen::SparseMatrix<double> adjacency,
                           std::optional<EdgeColoring> coloring)
    : _num_sites(static_cast<std::uint64_t>(adjacency.rows())),
      adjacency_(std::move(adjacency)),
      _is_symmetric(_check_symmetry(adjacency_)),
      _edge_coloring(std::move(coloring)) {
  _validate_coloring();
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

LatticeGraph LatticeGraph::from_dense_matrix(
    const Eigen::MatrixXd& adjacency_matrix) {
  if (adjacency_matrix.rows() != adjacency_matrix.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }
  Eigen::SparseMatrix<double> sparse = adjacency_matrix.sparseView();
  sparse.makeCompressed();
  return LatticeGraph(std::move(sparse));
}

LatticeGraph LatticeGraph::from_sparse_matrix(
    const Eigen::SparseMatrix<double>& sparse) {
  if (sparse.rows() != sparse.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }
  Eigen::SparseMatrix<double> copy = sparse;
  copy.makeCompressed();
  return LatticeGraph(std::move(copy));
}

LatticeGraph LatticeGraph::make_bidirectional(const LatticeGraph& graph) {
  Eigen::SparseMatrix<double> sym =
      (graph.adjacency_ +
       Eigen::SparseMatrix<double>(graph.adjacency_.transpose()));
  sym.makeCompressed();
  LatticeGraph result = graph;
  result.adjacency_ = std::move(sym);
  result._is_symmetric = _check_symmetry(result.adjacency_);
  result._edge_coloring.reset();
  for (auto& connection : result._connections) {
    connection.weight *= 2.0;
    if (!std::isfinite(connection.weight)) {
      throw std::overflow_error("Bidirectional connection weight overflowed.");
    }
  }
  if (result._has_connections) {
    for (int k = 0; k < result.adjacency_.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(result.adjacency_, k);
           it; ++it) {
        if (!std::isfinite(it.value())) {
          throw std::overflow_error(
              "Bidirectional adjacency weight overflowed.");
        }
      }
    }
  }
  return result;
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

const std::shared_ptr<const LatticeGeometry>& LatticeGraph::geometry() const {
  return _geometry;
}

const std::vector<std::uint64_t>& LatticeGraph::selected_shells() const {
  return _selected_shells;
}

const std::vector<NeighborConnection>& LatticeGraph::connections() const {
  return _connections;
}

LatticeGraph LatticeGraph::from_geometry(
    const LatticeGeometry& geometry, const std::vector<std::uint64_t>& shells,
    const std::vector<BondFlavorDefinition>& definitions, double weight,
    double tolerance) {
  if (!std::isfinite(weight)) {
    throw std::invalid_argument("Connection weight must be finite.");
  }
  auto selected = shells;
  detail::normalize_shells(selected);
  auto connections = geometry.neighbor_connections(selected, tolerance);
  for (auto& connection : connections) connection.weight = weight;
  detail::label_connections(connections, definitions, tolerance);
  return from_connections(geometry.num_sites(), std::move(connections),
                          std::make_shared<LatticeGeometry>(geometry),
                          std::move(selected));
}

LatticeGraph LatticeGraph::from_connections(
    std::uint64_t num_sites, std::vector<NeighborConnection> connections,
    std::shared_ptr<const LatticeGeometry> geometry,
    std::vector<std::uint64_t> selected_shells) {
  return _from_connections(num_sites, std::move(connections),
                           std::move(geometry), std::move(selected_shells),
                           std::nullopt);
}

LatticeGraph LatticeGraph::_from_connections(
    std::uint64_t num_sites, std::vector<NeighborConnection> connections,
    std::shared_ptr<const LatticeGeometry> geometry,
    std::vector<std::uint64_t> selected_shells,
    std::optional<EdgeColoring> coloring) {
  if (num_sites > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Lattice site count exceeds the sparse index range.");
  }
  if (geometry && geometry->num_sites() != num_sites) {
    throw std::invalid_argument("Graph and geometry site counts must match.");
  }
  std::set<std::tuple<std::uint64_t, std::uint64_t, std::int64_t, std::int64_t>>
      seen;
  for (auto& connection : connections) {
    const auto& axis = connection.bond_class.axis;
    if (connection.site_i >= num_sites || connection.site_j >= num_sites ||
        connection.bond_class.shell == 0 || !std::isfinite(connection.weight) ||
        !connection.displacement.allFinite() ||
        connection.displacement.cwiseAbs().maxCoeff() == 0.0 ||
        !axis.allFinite() ||
        std::abs(blas::nrm2(2, axis.data(), 1) - 1.0) > 1.0e-9 ||
        (connection.site_i == connection.site_j &&
         connection.image_shift == std::array<std::int64_t, 2>{0, 0})) {
      throw std::invalid_argument("Invalid physical lattice connection.");
    }
    detail::canonicalize_connection(connection);
    if (!seen.emplace(connection.site_i, connection.site_j,
                      connection.image_shift[0], connection.image_shift[1])
             .second) {
      throw std::invalid_argument("Duplicate canonical lattice connection.");
    }
    selected_shells.push_back(connection.bond_class.shell);
  }
  detail::normalize_shells(selected_shells);
  std::sort(connections.begin(), connections.end(), detail::connection_less);

  // Scale each pair's sum to avoid intermediate overflow before cancellation.
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           std::pair<double, long double>>
      pair_weights;
  for (const auto& connection : connections) {
    auto& [scale, sum] = pair_weights[{connection.site_i, connection.site_j}];
    const double magnitude = std::abs(connection.weight);
    if (magnitude > scale) {
      sum *= static_cast<long double>(scale) / magnitude;
      scale = magnitude;
    }
    if (scale != 0.0)
      sum += static_cast<long double>(connection.weight) / scale;
  }
  std::vector<detail::Triplet> triplets;
  triplets.reserve(2 * pair_weights.size());
  for (const auto& [pair, scaled] : pair_weights) {
    const auto [scale, sum] = scaled;
    const double weight = static_cast<double>(sum * scale);
    if (!std::isfinite(weight)) {
      throw std::overflow_error("Connection weights overflow the adjacency.");
    }
    const int i = static_cast<int>(pair.first);
    const int j = static_cast<int>(pair.second);
    triplets.emplace_back(i, j, weight);
    if (i != j) triplets.emplace_back(j, i, weight);
  }
  const auto n = static_cast<Eigen::Index>(num_sites);
  Eigen::SparseMatrix<double> adjacency(n, n);
  adjacency.setFromTriplets(triplets.begin(), triplets.end());
  adjacency.makeCompressed();
  if (!coloring) {
    // Color topology, not summed weights: cancelled images can still carry
    // distinct flavored interactions, and shell couplings ignore weights.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pairs;
    pairs.reserve(pair_weights.size());
    for (const auto& [pair, weight] : pair_weights) {
      if (pair.first != pair.second) pairs.push_back(pair);
    }
    // Match the native sparse-adjacency traversal before shuffled trials.
    std::sort(pairs.begin(), pairs.end(), [](const auto& lhs, const auto& rhs) {
      return std::tie(lhs.second, lhs.first) < std::tie(rhs.second, rhs.first);
    });
    coloring = detail::color_edges(num_sites, pairs, 0, 32);
  }
  LatticeGraph result(std::move(adjacency), std::move(coloring));
  result._geometry = std::move(geometry);
  result._selected_shells = std::move(selected_shells);
  result._connections = std::move(connections);
  result._has_connections = true;
  return result;
}

void LatticeGraph::_restore_adjacency(Eigen::SparseMatrix<double> adjacency) {
  if (adjacency.rows() != adjacency_.rows() ||
      adjacency.cols() != adjacency_.cols()) {
    throw std::invalid_argument("Adjacency cache has the wrong dimensions.");
  }
  // A cached factory projection preserves exact t and sparse zero entries.
  // Bound summation/division roundoff per physical pair, including cancellation
  // after permutation and underflow when splitting very small periodic weights.
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           std::pair<long double, std::size_t>>
      bounds;
  for (const auto& connection : _connections) {
    auto& [magnitude, count] = bounds[{connection.site_i, connection.site_j}];
    magnitude += std::abs(static_cast<long double>(connection.weight));
    ++count;
  }
  const auto check = [&](Eigen::Index i, Eigen::Index j, double cached,
                         double projected) {
    auto bound = bounds.find({static_cast<std::uint64_t>(std::min(i, j)),
                              static_cast<std::uint64_t>(std::max(i, j))});
    long double tolerance = 0.0L;
    if (bound != bounds.end()) {
      const auto& [magnitude, count] = bound->second;
      tolerance = 8.0L * count *
                  (std::numeric_limits<double>::epsilon() * magnitude +
                   std::numeric_limits<double>::denorm_min());
    }
    if (!std::isfinite(cached) ||
        std::abs(static_cast<long double>(cached) - projected) > tolerance) {
      throw std::invalid_argument(
          "Adjacency cache disagrees with the physical connections.");
    }
  };
  for (int k = 0; k < adjacency.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency, k); it;
         ++it) {
      if (it.value() != adjacency.coeff(it.col(), it.row())) {
        throw std::invalid_argument(
            "Physical-connection adjacency must be symmetric.");
      }
      check(it.row(), it.col(), it.value(),
            adjacency_.coeff(it.row(), it.col()));
    }
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      check(it.row(), it.col(), adjacency.coeff(it.row(), it.col()),
            it.value());
    }
  }
  adjacency.makeCompressed();
  adjacency_ = std::move(adjacency);
  _is_symmetric = _check_symmetry(adjacency_);
}

LatticeGraph LatticeGraph::_with_geometry(
    Eigen::SparseMatrix<double> adjacency, std::optional<EdgeColoring> coloring,
    std::shared_ptr<const LatticeGeometry> geometry) {
  if (geometry->num_sites() != static_cast<std::uint64_t>(adjacency.rows()) ||
      adjacency.rows() != adjacency.cols()) {
    throw std::invalid_argument("Graph and geometry site counts must match.");
  }
  auto connections = geometry->neighbor_connections({1});
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> multiplicity;
  for (const auto& connection : connections) {
    ++multiplicity[{connection.site_i, connection.site_j}];
  }
  for (auto& connection : connections) {
    connection.weight =
        adjacency.coeff(static_cast<Eigen::Index>(connection.site_i),
                        static_cast<Eigen::Index>(connection.site_j)) /
        static_cast<double>(
            multiplicity.at({connection.site_i, connection.site_j}));
  }
  auto result = _from_connections(static_cast<std::uint64_t>(adjacency.rows()),
                                  std::move(connections), std::move(geometry),
                                  {1}, coloring);
  result._restore_adjacency(std::move(adjacency));
  result._edge_coloring = std::move(coloring);
  result._validate_coloring();
  return result;
}

LatticeGraph LatticeGraph::with_bond_flavors(
    const std::vector<BondFlavorDefinition>& definitions,
    double tolerance) const {
  LatticeGraph result = *this;
  detail::label_connections(result._connections, definitions, tolerance);
  return result;
}

LatticeGraph LatticeGraph::chain(std::uint64_t n, bool periodic, double t,
                                 bool dfs_ordering) {
  auto geometry =
      std::make_shared<LatticeGeometry>(LatticeGeometry::chain(n, periodic));

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
  auto g = _with_geometry(
      std::move(adj), chain_coloring(static_cast<std::int64_t>(N), periodic),
      std::move(geometry));
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
  auto geometry = std::make_shared<LatticeGeometry>(
      LatticeGeometry::square(nx, ny, periodic_x, periodic_y));

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
  auto g = _with_geometry(std::move(adj),
                          square_coloring(Nx, Ny, periodic_x, periodic_y),
                          std::move(geometry));
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
  auto geometry = std::make_shared<LatticeGeometry>(
      LatticeGeometry::triangular(nx, ny, periodic_x, periodic_y));

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
  auto g =
      _with_geometry(std::move(adj), std::move(coloring), std::move(geometry));
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
  auto geometry = std::make_shared<LatticeGeometry>(
      LatticeGeometry::honeycomb(nx, ny, periodic_x, periodic_y));
  return _honeycomb(nx, ny, false, periodic_x, periodic_y, t,
                    std::move(geometry));
}

LatticeGraph LatticeGraph::honeycomb_plaquettes(std::uint64_t nx,
                                                std::uint64_t ny,
                                                bool periodic_x,
                                                bool periodic_y, double t,
                                                bool dfs_ordering) {
  (void)dfs_ordering;
  auto geometry = std::make_shared<LatticeGeometry>(
      LatticeGeometry::honeycomb_plaquettes(nx, ny, periodic_x, periodic_y));
  const auto num_cells_x = nx + static_cast<std::uint64_t>(!periodic_x);
  const auto num_cells_y = ny + static_cast<std::uint64_t>(!periodic_y);
  return _honeycomb(num_cells_x, num_cells_y, !periodic_x && !periodic_y,
                    periodic_x, periodic_y, t, std::move(geometry));
}

LatticeGraph LatticeGraph::_honeycomb(
    std::uint64_t num_cells_x, std::uint64_t num_cells_y,
    bool remove_open_corners, bool periodic_x, bool periodic_y, double t,
    std::shared_ptr<const LatticeGeometry> geometry) {
  const auto Nx = static_cast<int>(num_cells_x);
  const auto Ny = static_cast<int>(num_cells_y);
  const int full_num_sites = 2 * Nx * Ny;

  // Site indices within unit cell (x, y):
  //   A = 2 * (y * Nx + x),  B = 2 * (y * Nx + x) + 1
  auto idxA = [Nx](int x, int y) { return 2 * (y * Nx + x); };
  auto idxB = [Nx](int x, int y) { return 2 * (y * Nx + x) + 1; };

  std::vector<int> old_to_new(full_num_sites, -1);
  int num_sites = 0;
  for (int old_site = 0; old_site < full_num_sites; ++old_site) {
    const bool dangling_open_corner =
        remove_open_corners &&
        (old_site == idxA(0, 0) || old_site == idxB(Nx - 1, Ny - 1));
    if (!dangling_open_corner) old_to_new[old_site] = num_sites++;
  }

  std::vector<detail::Triplet> triplets;
  triplets.reserve(3 * num_sites);
  auto add_edge = [&triplets, &old_to_new, t](int old_i, int old_j) {
    const int i = old_to_new[old_i];
    const int j = old_to_new[old_j];
    if (i >= 0 && j >= 0) detail::add_edge(triplets, i, j, t);
  };

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Intra-cell bond: A -- B
      add_edge(idxA(x, y), idxB(x, y));

      // Inter-cell bond 1: B(x,y) -- A(x+1, y)  (horizontal)
      if (x + 1 < Nx) {
        add_edge(idxB(x, y), idxA(x + 1, y));
      } else if (periodic_x) {
        add_edge(idxB(x, y), idxA(0, y));
      }

      // Inter-cell bond 2: B(x,y) -- A(x, y+1)  (vertical)
      if (y + 1 < Ny) {
        add_edge(idxB(x, y), idxA(x, y + 1));
      } else if (periodic_y) {
        add_edge(idxB(x, y), idxA(x, 0));
      }
    }
  }

  Eigen::SparseMatrix<double> adj(num_sites, num_sites);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();

  EdgeColoring coloring;
  for (const auto& [edge, color] :
       honeycomb_coloring(Nx, Ny, periodic_x, periodic_y)) {
    const int i = old_to_new[edge.first];
    const int j = old_to_new[edge.second];
    if (i < 0 || j < 0) continue;
    const auto mapped_i = static_cast<std::uint64_t>(i);
    const auto mapped_j = static_cast<std::uint64_t>(j);
    coloring[{std::min(mapped_i, mapped_j), std::max(mapped_i, mapped_j)}] =
        color;
  }

  return _with_geometry(std::move(adj), std::move(coloring),
                        std::move(geometry));
}

LatticeGraph LatticeGraph::kagome(std::uint64_t nx, std::uint64_t ny,
                                  bool periodic_x, bool periodic_y, double t,
                                  int coloring_seed, bool dfs_ordering) {
  (void)dfs_ordering;
  auto geometry = std::make_shared<LatticeGeometry>(
      LatticeGeometry::kagome(nx, ny, periodic_x, periodic_y));

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
  return _with_geometry(std::move(adj), std::move(coloring),
                        std::move(geometry));
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

  // For explicit graphs this is a checked projection cache, not a second
  // source of connectivity. It retains legacy factory weights and zero entries.
  nlohmann::json edges = nlohmann::json::array();
  for (int k = 0; k < adjacency_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      edges.push_back({it.row(), it.col(), it.value()});
    }
  }

  nlohmann::json j;
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

  if (_has_connections) {
    j["selected_shells"] = _selected_shells;
    j["connections"] = nlohmann::json::array();
    for (const auto& connection : _connections) {
      const auto& bond = connection.bond_class;
      j["connections"].push_back(
          {{"site_i", connection.site_i},
           {"site_j", connection.site_j},
           {"bond_class",
            {{"shell", bond.shell},
             {"orientation", bond.orientation},
             {"axis", {bond.axis.x(), bond.axis.y()}}}},
           {"displacement",
            {connection.displacement.x(), connection.displacement.y()}},
           {"image_shift", connection.image_shift},
           {"flavor", connection.flavor ? nlohmann::json(*connection.flavor)
                                        : nlohmann::json(nullptr)},
           {"weight", connection.weight}});
    }
  }
  if (_geometry) j["geometry"] = _geometry->to_json();

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

    if (_has_connections) {
      save_vector_to_group(group, "selected_shells",
                           std::vector<std::size_t>(_selected_shells.begin(),
                                                    _selected_shells.end()));
      auto records = group.createGroup("connections");
      const auto count = _connections.size();
      std::vector<std::size_t> site_i(count), site_j(count), shells(count),
          orientations(count), flavors(count);
      std::vector<std::int64_t> images(2 * count);
      std::vector<double> weights(count);
      Eigen::MatrixXd axes(count, 2);
      Eigen::MatrixXd displacements(count, 2);
      for (std::size_t i = 0; i < count; ++i) {
        const auto& connection = _connections[i];
        site_i[i] = connection.site_i;
        site_j[i] = connection.site_j;
        shells[i] = connection.bond_class.shell;
        orientations[i] = connection.bond_class.orientation;
        axes.row(static_cast<Eigen::Index>(i)) = connection.bond_class.axis;
        displacements.row(static_cast<Eigen::Index>(i)) =
            connection.displacement;
        images[2 * i] = connection.image_shift[0];
        images[2 * i + 1] = connection.image_shift[1];
        // UINT64_MAX is outside BondFlavorId, preserving all uint32 labels.
        flavors[i] = connection.flavor
                         ? static_cast<std::uint64_t>(*connection.flavor)
                         : std::numeric_limits<std::uint64_t>::max();
        weights[i] = connection.weight;
      }
      save_vector_to_group(records, "site_i", site_i);
      save_vector_to_group(records, "site_j", site_j);
      save_vector_to_group(records, "shells", shells);
      save_vector_to_group(records, "orientations", orientations);
      save_vector_to_group(records, "flavors", flavors);
      save_matrix_to_group(records, "axes", axes);
      save_matrix_to_group(records, "displacements", displacements);
      save_stl_to_group(records, "weights", weights);
      hsize_t image_dims[2] = {count, 2};
      H5::DataSpace image_space(2, image_dims);
      auto image_dataset = records.createDataSet(
          "image_shifts", H5::PredType::NATIVE_INT64, image_space);
      if (!images.empty()) {
        image_dataset.write(images.data(), H5::PredType::NATIVE_INT64);
      }
    }
    if (_geometry) {
      auto geometry_group = group.createGroup("geometry");
      _geometry->to_hdf5(geometry_group);
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

  if (!j.is_object() || !j.contains("num_sites")) {
    throw std::runtime_error("JSON missing required 'num_sites' field");
  }
  const bool explicit_connections = j.contains("connections");
  if (!explicit_connections && !j.contains("adjacency_sparse")) {
    throw std::runtime_error("JSON missing required 'adjacency_sparse' field");
  }
  if ((!explicit_connections && j.contains("selected_shells")) ||
      (explicit_connections &&
       (j.contains("positions") || j.contains("periods") ||
        j.contains("bond_flavor_definitions")))) {
    throw std::invalid_argument(
        "Incomplete or mixed lattice connection metadata.");
  }

  const auto n = detail::json_integer<std::uint64_t>(j.at("num_sites"));
  if (n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Lattice site count exceeds the sparse index range.");
  }
  const auto n_idx = static_cast<int>(n);

  std::vector<detail::Triplet> triplets;
  if (j.contains("adjacency_sparse")) {
    if (!j.at("adjacency_sparse").is_array()) {
      throw std::invalid_argument("Sparse adjacency must be a triplet array.");
    }
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

  std::shared_ptr<const LatticeGeometry> geometry;
  if (j.contains("geometry")) {
    geometry = std::make_shared<LatticeGeometry>(
        LatticeGeometry::from_json(j.at("geometry")));
  } else if (j.contains("positions")) {
    geometry = std::make_shared<LatticeGeometry>(LatticeGeometry::from_json(j));
  } else if (j.contains("periods")) {
    throw std::invalid_argument("Periodic vectors require lattice positions.");
  }

  if (explicit_connections) {
    const auto read_axis =
        [](const nlohmann::json& value) -> Eigen::RowVector2d {
      if (!value.is_array() || value.size() != 2) {
        throw std::invalid_argument(
            "Connection vectors require two components.");
      }
      return {value[0].get<double>(), value[1].get<double>()};
    };
    std::vector<std::uint64_t> shells;
    if (j.contains("selected_shells")) {
      if (!j.at("selected_shells").is_array()) {
        throw std::invalid_argument(
            "Selected shells must be an integer array.");
      }
      for (const auto& shell : j.at("selected_shells")) {
        shells.push_back(detail::json_integer<std::uint64_t>(shell));
      }
    }
    if (!j.at("connections").is_array()) {
      throw std::invalid_argument("Connections must be an array of records.");
    }
    std::vector<NeighborConnection> connections;
    connections.reserve(j.at("connections").size());
    for (const auto& entry : j.at("connections")) {
      const auto& bond = entry.at("bond_class");
      const auto& image = entry.at("image_shift");
      if (!image.is_array() || image.size() != 2) {
        throw std::invalid_argument(
            "Connection images require two integer shifts.");
      }
      std::optional<BondFlavorId> flavor;
      if (entry.contains("flavor") && !entry.at("flavor").is_null()) {
        flavor = detail::json_integer<BondFlavorId>(entry.at("flavor"));
      }
      connections.push_back(
          {detail::json_integer<std::uint64_t>(entry.at("site_i")),
           detail::json_integer<std::uint64_t>(entry.at("site_j")),
           {detail::json_integer<std::uint64_t>(bond.at("shell")),
            detail::json_integer<std::uint32_t>(bond.at("orientation")),
            read_axis(bond.at("axis"))},
           read_axis(entry.at("displacement")),
           {detail::json_integer<std::int64_t>(image[0]),
            detail::json_integer<std::int64_t>(image[1])},
           flavor,
           entry.at("weight").get<double>()});
    }
    auto graph =
        _from_connections(n, std::move(connections), std::move(geometry),
                          std::move(shells), std::move(coloring));
    if (j.contains("adjacency_sparse")) {
      graph._restore_adjacency(std::move(sparse));
    }
    graph._validate_coloring();
    return graph;
  }

  std::vector<BondFlavorDefinition> bond_flavors;
  if (j.contains("bond_flavor_definitions")) {
    if (!j.at("bond_flavor_definitions").is_array()) {
      throw std::invalid_argument("Bond-flavor definitions must be an array.");
    }
    for (const auto& entry : j.at("bond_flavor_definitions")) {
      if (!entry.is_array() || entry.size() != 4) {
        throw std::invalid_argument("Invalid legacy bond-flavor definition.");
      }
      bond_flavors.push_back({detail::json_integer<std::uint64_t>(entry[0]),
                              {entry[1].get<double>(), entry[2].get<double>()},
                              detail::json_integer<BondFlavorId>(entry[3])});
    }
  }
  if (!geometry && !bond_flavors.empty()) {
    throw std::invalid_argument(
        "Legacy bond flavors require lattice positions.");
  }
  // Legacy adjacency need not agree with any geometric shell. Preserve it
  // without inferring interactions; shell selection is explicit in the new API.
  LatticeGraph graph(std::move(sparse), std::move(coloring));
  if (geometry && geometry->num_sites() != n) {
    throw std::invalid_argument("Graph and geometry site counts must match.");
  }
  graph._geometry = std::move(geometry);
  detail::label_connections(graph._connections, std::move(bond_flavors),
                            1.0e-9);
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
    const auto row_count = [](const H5::DataSet& dataset, int rank,
                              hsize_t columns = 1) {
      const auto space = dataset.getSpace();
      if (space.getSimpleExtentNdims() != rank) {
        throw std::invalid_argument("Invalid lattice dataset rank.");
      }
      hsize_t dimensions[2] = {};
      space.getSimpleExtentDims(dimensions);
      if ((rank == 2 && dimensions[1] != columns) ||
          dimensions[0] > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid lattice dataset dimensions.");
      }
      return dimensions[0];
    };
    const auto read_unsigned = [&](H5::Group& source, const std::string& name) {
      const auto dataset = source.openDataSet(name);
      row_count(dataset, 1);
      if (dataset.getTypeClass() != H5T_INTEGER ||
          dataset.getIntType().getSign() != H5T_SGN_NONE ||
          dataset.getIntType().getSize() > sizeof(std::uint64_t)) {
        throw std::invalid_argument("Expected unsigned integer dataset: " +
                                    name);
      }
      return load_size_vector_from_group(source, name);
    };
    const auto read_matrix = [&](H5::Group& source, const std::string& name) {
      const auto dataset = source.openDataSet(name);
      row_count(dataset, 2, 2);
      if (dataset.getTypeClass() != H5T_FLOAT) {
        throw std::invalid_argument("Expected floating-point matrix: " + name);
      }
      return load_matrix_from_group(source, name);
    };
    const auto read_triplets = [&](const std::string& name) {
      const auto dataset = group.openDataSet(name);
      const auto count = row_count(dataset, 2, 3);
      if (dataset.getTypeClass() != H5T_FLOAT) {
        throw std::invalid_argument("Expected sparse triplet dataset: " + name);
      }
      std::vector<double> buffer(3 * count);
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
    const bool explicit_connections = group.nameExists("connections");
    if ((!explicit_connections && group.nameExists("selected_shells")) ||
        (explicit_connections &&
         (group.nameExists("positions") || group.nameExists("periods") ||
          group.nameExists("bond_flavor_definitions")))) {
      throw std::invalid_argument(
          "Incomplete or mixed lattice connection metadata.");
    }
    if (!explicit_connections && !group.nameExists("adjacency_sparse")) {
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
    std::vector<detail::Triplet> triplets;
    if (group.nameExists("adjacency_sparse")) {
      const auto buffer = read_triplets("adjacency_sparse");
      triplets.reserve(buffer.size() / 3);
      for (std::size_t i = 0; i < buffer.size(); i += 3) {
        triplets.emplace_back(site_index(buffer[i]), site_index(buffer[i + 1]),
                              buffer[i + 2]);
      }
    }
    const auto n_idx = static_cast<Eigen::Index>(n);
    Eigen::SparseMatrix<double> sparse(n_idx, n_idx);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    sparse.makeCompressed();

    std::optional<EdgeColoring> coloring;
    if (group.nameExists("edge_coloring")) {
      coloring.emplace();
      const auto buffer = read_triplets("edge_coloring");
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

    std::shared_ptr<const LatticeGeometry> geometry;
    if (group.nameExists("geometry")) {
      auto geometry_group = group.openGroup("geometry");
      geometry = std::make_shared<LatticeGeometry>(
          LatticeGeometry::from_hdf5(geometry_group));
    } else if (group.nameExists("positions")) {
      geometry =
          std::make_shared<LatticeGeometry>(LatticeGeometry::from_hdf5(group));
    } else if (group.nameExists("periods")) {
      throw std::invalid_argument(
          "Periodic vectors require lattice positions.");
    }

    if (explicit_connections) {
      std::vector<std::uint64_t> selected;
      if (group.nameExists("selected_shells")) {
        const auto shells = read_unsigned(group, "selected_shells");
        selected.assign(shells.begin(), shells.end());
      }
      auto records = group.openGroup("connections");
      const auto site_i = read_unsigned(records, "site_i");
      const auto site_j = read_unsigned(records, "site_j");
      const auto shells = read_unsigned(records, "shells");
      const auto orientations = read_unsigned(records, "orientations");
      const auto flavors = read_unsigned(records, "flavors");
      const auto axes = read_matrix(records, "axes");
      const auto displacements = read_matrix(records, "displacements");
      const auto image_dataset = records.openDataSet("image_shifts");
      const auto weight_dataset = records.openDataSet("weights");
      const auto count = site_i.size();
      if (site_j.size() != count || shells.size() != count ||
          orientations.size() != count || flavors.size() != count ||
          static_cast<std::size_t>(axes.rows()) != count ||
          static_cast<std::size_t>(displacements.rows()) != count ||
          row_count(image_dataset, 2, 2) != count ||
          image_dataset.getTypeClass() != H5T_INTEGER ||
          image_dataset.getIntType().getSign() != H5T_SGN_2 ||
          image_dataset.getIntType().getSize() > sizeof(std::int64_t) ||
          row_count(weight_dataset, 1) != count ||
          weight_dataset.getTypeClass() != H5T_FLOAT) {
        throw std::invalid_argument(
            "Invalid connection dataset shape or type.");
      }
      std::vector<std::int64_t> images(2 * count);
      if (!images.empty()) {
        image_dataset.read(images.data(), H5::PredType::NATIVE_INT64);
      }
      const auto weights =
          load_std_vector_from_group<double>(records, "weights");
      std::vector<NeighborConnection> connections;
      connections.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        if (orientations[i] > std::numeric_limits<std::uint32_t>::max() ||
            (flavors[i] != std::numeric_limits<std::uint64_t>::max() &&
             flavors[i] > std::numeric_limits<BondFlavorId>::max())) {
          throw std::invalid_argument(
              "Invalid connection orientation or flavor ID.");
        }
        std::optional<BondFlavorId> flavor;
        if (flavors[i] != std::numeric_limits<std::uint64_t>::max()) {
          flavor = static_cast<BondFlavorId>(flavors[i]);
        }
        connections.push_back(
            {site_i[i],
             site_j[i],
             {shells[i], static_cast<std::uint32_t>(orientations[i]),
              axes.row(i)},
             displacements.row(i),
             {images[2 * i], images[2 * i + 1]},
             flavor,
             weights[i]});
      }
      auto graph =
          _from_connections(n, std::move(connections), std::move(geometry),
                            std::move(selected), std::move(coloring));
      if (group.nameExists("adjacency_sparse")) {
        graph._restore_adjacency(std::move(sparse));
      }
      graph._validate_coloring();
      return graph;
    }

    std::vector<BondFlavorDefinition> bond_flavors;
    if (group.nameExists("bond_flavor_definitions")) {
      auto definitions = group.openGroup("bond_flavor_definitions");
      const auto shells = read_unsigned(definitions, "shells");
      const auto flavors = read_unsigned(definitions, "flavors");
      const auto axes = read_matrix(definitions, "axes");
      if (flavors.size() != shells.size() ||
          static_cast<std::size_t>(axes.rows()) != shells.size()) {
        throw std::invalid_argument(
            "Invalid legacy bond-flavor dataset shape.");
      }
      for (std::size_t i = 0; i < shells.size(); ++i) {
        if (flavors[i] > std::numeric_limits<BondFlavorId>::max()) {
          throw std::invalid_argument("Legacy bond flavor is out of range.");
        }
        bond_flavors.push_back(
            {shells[i], axes.row(i), static_cast<BondFlavorId>(flavors[i])});
      }
    }
    if (!geometry && !bond_flavors.empty()) {
      throw std::invalid_argument(
          "Legacy bond flavors require lattice positions.");
    }
    LatticeGraph graph(std::move(sparse), std::move(coloring));
    if (geometry && geometry->num_sites() != n) {
      throw std::invalid_argument("Graph and geometry site counts must match.");
    }
    graph._geometry = std::move(geometry);
    detail::label_connections(graph._connections, std::move(bond_flavors),
                              1.0e-9);
    return graph;
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in LatticeGraph::from_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

void LatticeGraph::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, _num_sites);
  // Scalar adjacency consumers observe exact factory weights, even when
  // splitting a subnormal weight among images rounds a connection to zero.
  hash_value(ctx, adjacency_);
  hash_value(ctx, _has_connections);
  if (_has_connections) {
    hash_value(ctx, _selected_shells);
    hash_value(ctx, static_cast<std::uint64_t>(_connections.size()));
    for (const auto& connection : _connections) {
      hash_value(ctx, connection.site_i);
      hash_value(ctx, connection.site_j);
      hash_value(ctx, connection.bond_class.shell);
      hash_value(ctx, connection.bond_class.orientation);
      hash_value(ctx, connection.bond_class.axis);
      hash_value(ctx, connection.displacement);
      for (const auto shift : connection.image_shift) hash_value(ctx, shift);
      hash_value(ctx, connection.flavor);
      hash_value(ctx, connection.weight);
    }
  }
  hash_value(ctx, static_cast<bool>(_geometry));
  if (_geometry) {
    hash_value(ctx, _geometry->content_hash());
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
  if (graph._geometry) {
    result._geometry = std::make_shared<LatticeGeometry>(
        LatticeGeometry::permute(*graph._geometry, path));
  }
  result._has_connections = graph._has_connections;
  result._selected_shells = graph._selected_shells;
  result._connections = graph._connections;
  for (auto& connection : result._connections) {
    connection.site_i = inv_p[connection.site_i];
    connection.site_j = inv_p[connection.site_j];
    detail::canonicalize_connection(connection);
  }
  std::sort(result._connections.begin(), result._connections.end(),
            detail::connection_less);
  return result;
}

}  // namespace qdk::chemistry::data
