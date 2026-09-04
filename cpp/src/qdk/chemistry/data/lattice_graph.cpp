// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <blas.hh>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <lapack.hh>
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
#include <vector>

#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

namespace detail {
using Triplet = Eigen::Triplet<double>;

// Helper: add an undirected edge (i, j) with weight t to the triplet list.
static void add_edge(std::vector<Triplet>& triplets, int i, int j, double t) {
  triplets.emplace_back(i, j, t);
  triplets.emplace_back(j, i, t);
}

template <std::size_t NumBasisSites>
Eigen::MatrixXd lattice_positions(
    int nx, int ny, const Eigen::RowVector2d& a1, const Eigen::RowVector2d& a2,
    const std::array<Eigen::RowVector2d, NumBasisSites>& basis) {
  Eigen::MatrixXd positions(nx * ny * NumBasisSites, 2);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      for (std::size_t s = 0; s < NumBasisSites; ++s) {
        auto i = NumBasisSites * (y * nx + x) + s;
        positions.row(static_cast<Eigen::Index>(i)) =
            x * a1 + y * a2 + basis[s];
      }
    }
  }
  return positions;
}

std::optional<Eigen::MatrixXd> lattice_periods(
    const Eigen::RowVector2d& period_1 = Eigen::RowVector2d::Zero(),
    const Eigen::RowVector2d& period_2 = Eigen::RowVector2d::Zero()) {
  const auto nonzero = [](const Eigen::RowVector2d& vector) {
    return vector.cwiseAbs().maxCoeff() != 0.0;
  };
  const Eigen::Index count = nonzero(period_1) + nonzero(period_2);
  if (count == 0) return std::nullopt;

  Eigen::MatrixXd periods(count, 2);
  Eigen::Index row = 0;
  if (nonzero(period_1)) periods.row(row++) = period_1;
  if (nonzero(period_2)) periods.row(row) = period_2;
  return periods;
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
                           std::optional<EdgeColoring> coloring,
                           std::optional<Eigen::MatrixXd> positions,
                           std::optional<Eigen::MatrixXd> periods,
                           std::vector<BondFlavorDefinition> bond_flavors)
    : _num_sites(static_cast<std::uint64_t>(adjacency.rows())),
      adjacency_(std::move(adjacency)),
      _is_symmetric(_check_symmetry(adjacency_)),
      _edge_coloring(std::move(coloring)),
      _positions(std::move(positions)),
      _periods(std::move(periods)),
      _bond_flavor_definitions(std::move(bond_flavors)) {
  _validate_geometry(_num_sites, _positions, _periods);
  _validate_bond_flavors(_positions, _bond_flavor_definitions, 1.0e-9);
#ifndef NDEBUG
  if (_edge_coloring.has_value()) {
    // Verify every edge in the coloring exists in the adjacency matrix.
    for (const auto& [edge, color] : *_edge_coloring) {
      assert(edge.first < _num_sites && edge.second < _num_sites &&
             "coloring edge vertex out of range");
      assert(edge.first < edge.second && "coloring edge not canonical (i < j)");
      assert(adjacency_.coeff(static_cast<Eigen::Index>(edge.first),
                              static_cast<Eigen::Index>(edge.second)) != 0.0 &&
             "coloring contains edge not in adjacency matrix");
    }
    // Verify every upper-triangular edge in adjacency appears in coloring.
    for (int k = 0; k < adjacency_.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
           ++it) {
        if (it.row() < it.col() && it.value() != 0.0) {
          auto key = std::make_pair(static_cast<std::uint64_t>(it.row()),
                                    static_cast<std::uint64_t>(it.col()));
          assert(_edge_coloring->count(key) != 0 &&
                 "adjacency edge missing from coloring");
        }
      }
    }
  }
#endif
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
  return LatticeGraph(std::move(sym), std::nullopt, graph._positions,
                      graph._periods, graph._bond_flavor_definitions);
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

const std::optional<Eigen::MatrixXd>& LatticeGraph::positions() const {
  return _positions;
}

void LatticeGraph::_validate_geometry(
    std::uint64_t num_sites, const std::optional<Eigen::MatrixXd>& positions,
    const std::optional<Eigen::MatrixXd>& periods) {
  if (!positions.has_value()) {
    if (periods.has_value()) {
      throw std::invalid_argument(
          "Periodic vectors require Cartesian site positions.");
    }
    return;
  }

  auto n = static_cast<Eigen::Index>(num_sites);
  if (positions->rows() != n || positions->cols() != 2 ||
      !positions->allFinite()) {
    throw std::invalid_argument(
        "Lattice positions must be a finite (num_sites, 2) matrix.");
  }

  if (!periods.has_value()) return;
  if ((periods->rows() != 1 && periods->rows() != 2) || periods->cols() != 2 ||
      !periods->allFinite()) {
    throw std::invalid_argument(
        "Periodic vectors must be a finite (1, 2) or (2, 2) matrix.");
  }
  for (Eigen::Index i = 0; i < periods->rows(); ++i) {
    if (periods->row(i).cwiseAbs().maxCoeff() == 0.0) {
      throw std::invalid_argument("Periodic vectors must be nonzero.");
    }
    if (!std::isfinite(std::hypot((*periods)(i, 0), (*periods)(i, 1)))) {
      throw std::invalid_argument("Periodic vector lengths must be finite.");
    }
  }
  if (periods->rows() == 2) {
    const double period_0_length =
        std::hypot((*periods)(0, 0), (*periods)(0, 1));
    const double period_1_length =
        std::hypot((*periods)(1, 0), (*periods)(1, 1));
    const double min_period_length = std::min(period_0_length, period_1_length);
    const double max_period_length = std::max(period_0_length, period_1_length);
    const double position_span =
        std::max(positions->col(0).maxCoeff() - positions->col(0).minCoeff(),
                 positions->col(1).maxCoeff() - positions->col(1).minCoeff());
    const double geometry_scale = std::max(max_period_length, position_span);
    if (min_period_length <
        16.0 * std::numeric_limits<double>::epsilon() * geometry_scale) {
      throw std::invalid_argument(
          "Lattice geometry exceeds the supported numerical condition range.");
    }

    const double p0_scale = periods->row(0).cwiseAbs().maxCoeff();
    const double p1_scale = periods->row(1).cwiseAbs().maxCoeff();
    const Eigen::RowVector2d p0 = periods->row(0) / p0_scale;
    const Eigen::RowVector2d p1 = periods->row(1) / p1_scale;
    const Eigen::RowVector2d u0 = p0 / blas::nrm2(2, p0.data(), 1);
    const Eigen::RowVector2d u1 = p1 / blas::nrm2(2, p1.data(), 1);
    const double sine = std::abs(u0.x() * u1.y() - u0.y() * u1.x());
    if (sine <= 16.0 * std::numeric_limits<double>::epsilon()) {
      throw std::invalid_argument(
          "Two periodic vectors must be linearly independent.");
    }
  }
}

void LatticeGraph::_validate_bond_flavors(
    const std::optional<Eigen::MatrixXd>& positions,
    std::vector<BondFlavorDefinition>& definitions, double tolerance) {
  if (definitions.empty()) return;
  if (!positions.has_value()) {
    throw std::invalid_argument(
        "Bond flavors require Cartesian lattice positions.");
  }
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
  for (std::size_t i = 1; i < definitions.size(); ++i) {
    const Eigen::RowVector2d difference =
        definitions[i - 1].axis - definitions[i].axis;
    if (definitions[i - 1].shell == definitions[i].shell &&
        blas::nrm2(2, difference.data(), 1) <= tolerance) {
      throw std::invalid_argument(
          "Each shell-axis class may have only one bond flavor.");
    }
  }
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
LatticeGraph::mth_nearest_neighbors(std::uint64_t m, double tolerance) const {
  return nearest_neighbor_shells({m}, tolerance).at(m);
}

std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
LatticeGraph::nearest_neighbor_shells(const std::vector<std::uint64_t>& shells,
                                      double tolerance) const {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw std::invalid_argument("Neighbor shell tolerance must be positive.");
  }
  std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
      results;
  for (std::uint64_t shell : shells) {
    if (shell == 0) {
      throw std::invalid_argument("Neighbor shell index m must be > 0.");
    }
    results.try_emplace(shell);
  }
  if (!_positions.has_value()) {
    throw std::runtime_error(
        "Geometric neighbor shells require lattice positions.");
  }
  if (results.empty() || _num_sites < 2) return results;

  const Eigen::Index num_periods = _periods.has_value() ? _periods->rows() : 0;
  Eigen::MatrixXd positions = *_positions;
  Eigen::MatrixXd periods = _periods.value_or(Eigen::MatrixXd(0, 2));
  double geometry_scale = positions.cwiseAbs().maxCoeff();
  if (num_periods != 0) {
    geometry_scale = std::max(geometry_scale, periods.cwiseAbs().maxCoeff());
  }
  if (geometry_scale == 0.0) geometry_scale = 1.0;
  positions /= geometry_scale;
  periods /= geometry_scale;
  const Eigen::RowVector2d origin = positions.row(0);
  positions.rowwise() -= origin;

  Eigen::RowVector2d period_0 = Eigen::RowVector2d::Zero();
  Eigen::RowVector2d period_1 = Eigen::RowVector2d::Zero();
  if (num_periods == 1) {
    period_0 = periods.row(0);
  } else if (num_periods == 2) {
    period_0 = periods.row(0);
    period_1 = periods.row(1);

    // Gauss reduction preserves the period lattice while making the fixed
    // 3x3 closest-image search independent of supercell aspect ratio.
    while (true) {
      const double length_0 = blas::nrm2(2, period_0.data(), 1);
      const double length_1 = blas::nrm2(2, period_1.data(), 1);
      if (length_1 < length_0) {
        std::swap(period_0, period_1);
        continue;
      }
      const Eigen::RowVector2d unit_0 = period_0 / length_0;
      const Eigen::RowVector2d normal_0{-unit_0.y(), unit_0.x()};
      const double parallel = period_1.dot(unit_0);
      if (2.0 * std::abs(parallel) <=
          length_0 * (1.0 + 16.0 * std::numeric_limits<double>::epsilon())) {
        break;
      }
      const double perpendicular = period_1.dot(normal_0);
      const double reduced_parallel = std::remainder(parallel, length_0);
      const Eigen::RowVector2d reduced =
          reduced_parallel * unit_0 + perpendicular * normal_0;
      if (reduced == period_1) {
        throw std::runtime_error(
            "Lattice period reduction made no numerical progress.");
      }
      period_1 = reduced;
    }
  }

  const auto reduce_along =
      [](const Eigen::RowVector2d& vector,
         const Eigen::RowVector2d& period) -> Eigen::RowVector2d {
    const double length = blas::nrm2(2, period.data(), 1);
    const Eigen::RowVector2d unit = period / length;
    const Eigen::RowVector2d normal{-unit.y(), unit.x()};
    const double parallel = vector.dot(unit);
    const double perpendicular = vector.dot(normal);
    return (std::remainder(parallel, length) * unit + perpendicular * normal)
        .eval();
  };

  auto minimum_image_distance = [&](const Eigen::RowVector2d& displacement) {
    if (num_periods == 0) {
      return blas::nrm2(2, displacement.data(), 1);
    }
    if (num_periods == 1) {
      const double period_length = blas::nrm2(2, period_0.data(), 1);
      const Eigen::RowVector2d unit = period_0 / period_length;
      const Eigen::RowVector2d normal{-unit.y(), unit.x()};
      const double parallel = displacement.dot(unit);
      const double perpendicular = displacement.dot(normal);
      return std::hypot(std::remainder(parallel, period_length), perpendicular);
    }

    Eigen::RowVector2d diagonal = period_1;
    if (period_0.dot(period_1) > 0.0) {
      diagonal -= period_0;
    } else {
      diagonal += period_0;
    }
    const std::array<Eigen::RowVector2d, 3> relevant_vectors = {
        period_0, period_1, diagonal};
    Eigen::RowVector2d image = displacement;
    for (int iteration = 0; iteration < 64; ++iteration) {
      bool reduced = false;
      for (const auto& period : relevant_vectors) {
        const Eigen::RowVector2d candidate = reduce_along(image, period);
        const double current_distance = blas::nrm2(2, image.data(), 1);
        const double candidate_distance = blas::nrm2(2, candidate.data(), 1);
        if (candidate_distance <
            current_distance *
                (1.0 - 64.0 * std::numeric_limits<double>::epsilon())) {
          image = candidate;
          reduced = true;
        }
      }
      if (!reduced) return blas::nrm2(2, image.data(), 1);
    }
    throw std::runtime_error("Minimum-image reduction did not converge.");
  };

  struct PairDistance {
    double distance;
    std::uint64_t i;
    std::uint64_t j;
  };
  std::vector<PairDistance> pairs;
  pairs.reserve(static_cast<std::size_t>(_num_sites * (_num_sites - 1) / 2));
  for (std::uint64_t i = 0; i < _num_sites; ++i) {
    for (std::uint64_t j = i + 1; j < _num_sites; ++j) {
      const Eigen::RowVector2d displacement =
          positions.row(static_cast<Eigen::Index>(j)) -
          positions.row(static_cast<Eigen::Index>(i));
      const double distance = minimum_image_distance(displacement);
      if (distance == 0.0) {
        continue;
      }
      pairs.push_back({distance, i, j});
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const PairDistance& lhs, const PairDistance& rhs) {
              if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
              }
              return std::tie(lhs.i, lhs.j) < std::tie(rhs.i, rhs.j);
            });

  const std::uint64_t max_requested_shell = results.rbegin()->first;
  std::uint64_t shell = 0;
  double shell_distance = 0.0;
  for (const auto& pair : pairs) {
    bool same_shell =
        shell != 0 && std::abs(pair.distance - shell_distance) <=
                          tolerance * std::max(std::abs(pair.distance),
                                               std::abs(shell_distance));
    if (!same_shell) {
      ++shell;
      shell_distance = pair.distance;
      if (shell > max_requested_shell) break;
    }
    auto result = results.find(shell);
    if (result != results.end()) {
      result->second.emplace_back(pair.i, pair.j);
    }
  }
  for (auto& [requested_shell, shell_pairs] : results) {
    (void)requested_shell;
    std::sort(shell_pairs.begin(), shell_pairs.end());
  }
  return results;
}

std::vector<NeighborConnection> LatticeGraph::neighbor_connections(
    const std::vector<std::uint64_t>& shells, double tolerance) const {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw std::invalid_argument(
        "Neighbor connection tolerance must be positive.");
  }
  std::set<std::uint64_t> requested_shells;
  for (std::uint64_t shell : shells) {
    if (shell == 0) {
      throw std::invalid_argument("Neighbor shell index must be > 0.");
    }
    requested_shells.insert(shell);
  }
  if (requested_shells.empty() || _num_sites == 0) return {};
  if (!_positions.has_value()) {
    throw std::runtime_error(
        "Geometric neighbor connections require lattice positions.");
  }

  struct Candidate {
    double distance;
    std::uint64_t site_i;
    std::uint64_t site_j;
    std::array<std::int64_t, 2> image_shift;
    Eigen::RowVector2d displacement;
    std::uint64_t shell = 0;
  };

  const Eigen::Index num_periods = _periods.has_value() ? _periods->rows() : 0;
  Eigen::MatrixXd positions = *_positions;
  Eigen::MatrixXd periods = _periods.value_or(Eigen::MatrixXd(0, 2));
  double geometry_scale = positions.cwiseAbs().maxCoeff();
  if (num_periods != 0) {
    geometry_scale = std::max(geometry_scale, periods.cwiseAbs().maxCoeff());
  }
  if (geometry_scale == 0.0) geometry_scale = 1.0;
  positions /= geometry_scale;
  periods /= geometry_scale;
  const Eigen::RowVector2d origin = positions.row(0);
  positions.rowwise() -= origin;

  double minimum_period_scale = std::numeric_limits<double>::infinity();
  if (num_periods == 1) {
    minimum_period_scale =
        blas::nrm2(2, periods.row(0).data(), periods.outerStride());
  } else if (num_periods == 2) {
    Eigen::Matrix2d basis;
    basis.col(0) = periods.row(0).transpose();
    basis.col(1) = periods.row(1).transpose();
    std::array<double, 2> singular_values;
    double unused = 0.0;
    const auto info = lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, 2,
                                    2, basis.data(), 2, singular_values.data(),
                                    &unused, 1, &unused, 1);
    if (info != 0) {
      throw std::runtime_error(
          "Failed to compute the periodic-vector singular values.");
    }
    minimum_period_scale =
        *std::min_element(singular_values.begin(), singular_values.end());
  }
  const Eigen::RowVector2d position_extents{
      positions.col(0).maxCoeff() - positions.col(0).minCoeff(),
      positions.col(1).maxCoeff() - positions.col(1).minCoeff()};
  const double position_span = blas::nrm2(2, position_extents.data(), 1);
  const std::uint64_t max_requested_shell = *requested_shells.rbegin();

  std::vector<Candidate> candidates;
  for (std::int64_t radius = 0;; ++radius) {
    candidates.clear();
    std::set<
        std::tuple<std::uint64_t, std::uint64_t, std::int64_t, std::int64_t>>
        seen;
    const std::int64_t lower_0 = num_periods >= 1 ? -radius : 0;
    const std::int64_t upper_0 = num_periods >= 1 ? radius : 0;
    const std::int64_t lower_1 = num_periods == 2 ? -radius : 0;
    const std::int64_t upper_1 = num_periods == 2 ? radius : 0;
    for (std::uint64_t source = 0; source < _num_sites; ++source) {
      for (std::uint64_t target = 0; target < _num_sites; ++target) {
        for (std::int64_t image_0 = lower_0; image_0 <= upper_0; ++image_0) {
          for (std::int64_t image_1 = lower_1; image_1 <= upper_1; ++image_1) {
            if (source == target && image_0 == 0 && image_1 == 0) continue;

            std::uint64_t site_i = source;
            std::uint64_t site_j = target;
            std::array<std::int64_t, 2> image_shift = {image_0, image_1};
            if (site_i > site_j ||
                (site_i == site_j &&
                 (image_shift[0] < 0 ||
                  (image_shift[0] == 0 && image_shift[1] < 0)))) {
              std::swap(site_i, site_j);
              image_shift[0] = -image_shift[0];
              image_shift[1] = -image_shift[1];
            }
            if (!seen.emplace(site_i, site_j, image_shift[0], image_shift[1])
                     .second) {
              continue;
            }

            Eigen::RowVector2d displacement =
                positions.row(static_cast<Eigen::Index>(site_j)) -
                positions.row(static_cast<Eigen::Index>(site_i));
            if (num_periods >= 1) {
              displacement +=
                  static_cast<double>(image_shift[0]) * periods.row(0);
            }
            if (num_periods == 2) {
              displacement +=
                  static_cast<double>(image_shift[1]) * periods.row(1);
            }
            const double distance = blas::nrm2(2, displacement.data(), 1);
            if (distance != 0.0) {
              candidates.push_back(
                  {distance, site_i, site_j, image_shift, displacement});
            }
          }
        }
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                if (lhs.distance != rhs.distance) {
                  return lhs.distance < rhs.distance;
                }
                return std::tie(lhs.site_i, lhs.site_j, lhs.image_shift) <
                       std::tie(rhs.site_i, rhs.site_j, rhs.image_shift);
              });
    std::uint64_t shell = 0;
    double shell_distance = 0.0;
    double requested_shell_distance = 0.0;
    for (auto& candidate : candidates) {
      const bool same_shell =
          shell != 0 && std::abs(candidate.distance - shell_distance) <=
                            tolerance * std::max(std::abs(candidate.distance),
                                                 std::abs(shell_distance));
      if (!same_shell) {
        ++shell;
        shell_distance = candidate.distance;
      }
      candidate.shell = shell;
      if (shell == max_requested_shell) {
        requested_shell_distance = shell_distance;
      }
    }

    if (num_periods == 0 ||
        (shell >= max_requested_shell &&
         minimum_period_scale * static_cast<double>(radius + 1) -
                 position_span >
             requested_shell_distance * (1.0 + tolerance))) {
      break;
    }
    if (radius == std::numeric_limits<std::int64_t>::max() - 1) {
      throw std::overflow_error(
          "Neighbor connection image radius exceeds the supported range.");
    }
  }

  std::map<std::uint64_t, std::vector<Eigen::RowVector2d>> shell_axes;
  for (const auto& candidate : candidates) {
    if (!requested_shells.contains(candidate.shell)) continue;
    Eigen::RowVector2d axis = candidate.displacement / candidate.distance;
    if (axis.x() < -tolerance ||
        (std::abs(axis.x()) <= tolerance && axis.y() < 0.0)) {
      axis = -axis;
    }
    auto& axes = shell_axes[candidate.shell];
    if (std::none_of(axes.begin(), axes.end(), [&](const auto& existing) {
          const Eigen::RowVector2d difference = existing - axis;
          return blas::nrm2(2, difference.data(), 1) <= tolerance;
        })) {
      axes.push_back(axis);
    }
  }
  for (auto& [shell, axes] : shell_axes) {
    (void)shell;
    std::sort(axes.begin(), axes.end(), [](const auto& lhs, const auto& rhs) {
      return std::atan2(lhs.y(), lhs.x()) < std::atan2(rhs.y(), rhs.x());
    });
  }

  std::vector<NeighborConnection> result;
  for (const auto& candidate : candidates) {
    if (!requested_shells.contains(candidate.shell)) continue;
    Eigen::RowVector2d axis = candidate.displacement / candidate.distance;
    if (axis.x() < -tolerance ||
        (std::abs(axis.x()) <= tolerance && axis.y() < 0.0)) {
      axis = -axis;
    }
    const auto& axes = shell_axes.at(candidate.shell);
    const auto orientation = static_cast<std::uint32_t>(std::distance(
        axes.begin(),
        std::find_if(axes.begin(), axes.end(), [&](const auto& existing) {
          const Eigen::RowVector2d difference = existing - axis;
          return blas::nrm2(2, difference.data(), 1) <= tolerance;
        })));
    std::optional<BondFlavorId> flavor;
    for (const auto& definition : _bond_flavor_definitions) {
      if (definition.shell != candidate.shell) continue;
      const Eigen::RowVector2d difference = definition.axis - axis;
      if (blas::nrm2(2, difference.data(), 1) <= tolerance) {
        flavor = definition.flavor;
        break;
      }
    }
    const Eigen::RowVector2d displacement =
        candidate.displacement * geometry_scale;
    if (!displacement.allFinite()) {
      throw std::overflow_error(
          "Neighbor connection displacement exceeds the supported range.");
    }
    result.push_back({candidate.site_i,
                      candidate.site_j,
                      {candidate.shell, orientation, axes[orientation]},
                      displacement,
                      candidate.image_shift,
                      flavor});
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.bond_class.shell, lhs.bond_class.orientation,
                    lhs.site_i, lhs.site_j, lhs.image_shift) <
           std::tie(rhs.bond_class.shell, rhs.bond_class.orientation,
                    rhs.site_i, rhs.site_j, rhs.image_shift);
  });
  return result;
}

LatticeGraph LatticeGraph::with_bond_flavors(
    const std::vector<BondFlavorDefinition>& definitions,
    double tolerance) const {
  auto normalized = definitions;
  _validate_bond_flavors(_positions, normalized, tolerance);
  return LatticeGraph(adjacency_, _edge_coloring, _positions, _periods,
                      std::move(normalized));
}

const std::vector<BondFlavorDefinition>& LatticeGraph::bond_flavor_definitions()
    const {
  return _bond_flavor_definitions;
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
  const Eigen::RowVector2d a1(1.0, 0.0);
  const Eigen::RowVector2d a2 = Eigen::RowVector2d::Zero();
  const std::array<Eigen::RowVector2d, 1> basis = {Eigen::RowVector2d::Zero()};
  auto positions = detail::lattice_positions(N, 1, a1, a2, basis);
  LatticeGraph g(
      std::move(adj), chain_coloring(static_cast<std::int64_t>(N), periodic),
      std::move(positions), detail::lattice_periods((periodic ? N : 0) * a1));
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
  const Eigen::RowVector2d a1(1.0, 0.0);
  const Eigen::RowVector2d a2(0.0, 1.0);
  const std::array<Eigen::RowVector2d, 1> basis = {Eigen::RowVector2d::Zero()};
  auto positions = detail::lattice_positions(Nx, Ny, a1, a2, basis);
  LatticeGraph g(std::move(adj),
                 square_coloring(Nx, Ny, periodic_x, periodic_y),
                 std::move(positions),
                 detail::lattice_periods((periodic_x ? Nx : 0) * a1,
                                         (periodic_y ? Ny : 0) * a2));
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
  // Guo and Franz, Phys. Rev. B 80, 113102 (2009), define the unit
  // directions u1=(1,0), u2=(1/2,sqrt(3)/2). The equivalent primitive basis
  // a1=u1, a2=u2-u1 matches the upper-right bonds used by this factory.
  const Eigen::RowVector2d a1(1.0, 0.0);
  const Eigen::RowVector2d a2(-0.5, std::sqrt(3.0) / 2.0);
  const std::array<Eigen::RowVector2d, 1> basis = {Eigen::RowVector2d::Zero()};
  auto positions = detail::lattice_positions(Nx, Ny, a1, a2, basis);
  // No known deterministic coloring for triangular lattices with arbitrary
  // periodic boundaries; use greedy with multiple trials instead.
  auto coloring = greedy_edge_coloring(adj, coloring_seed, 32);
  LatticeGraph g(std::move(adj), std::move(coloring), std::move(positions),
                 detail::lattice_periods((periodic_x ? Nx : 0) * a1,
                                         (periodic_y ? Ny : 0) * a2));
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
  return _honeycomb(nx, ny, false, periodic_x, periodic_y, t, dfs_ordering);
}

LatticeGraph LatticeGraph::honeycomb_plaquettes(std::uint64_t nx,
                                                std::uint64_t ny,
                                                bool periodic_x,
                                                bool periodic_y, double t,
                                                bool dfs_ordering) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("honeycomb_plaquettes: nx and ny must be > 0.");
  }
  const auto num_cells_x = nx + static_cast<std::uint64_t>(!periodic_x);
  const auto num_cells_y = ny + static_cast<std::uint64_t>(!periodic_y);
  return _honeycomb(num_cells_x, num_cells_y, !periodic_x && !periodic_y,
                    periodic_x, periodic_y, t, dfs_ordering);
}

LatticeGraph LatticeGraph::_honeycomb(std::uint64_t num_cells_x,
                                      std::uint64_t num_cells_y,
                                      bool remove_open_corners, bool periodic_x,
                                      bool periodic_y, double t,
                                      bool dfs_ordering) {
  (void)dfs_ordering;
  if (num_cells_x == 0 || num_cells_y == 0) {
    throw std::invalid_argument("honeycomb: nx and ny must be > 0.");
  }
  if (periodic_x && num_cells_x < 2) {
    throw std::invalid_argument("honeycomb: periodic_x requires nx > 1.");
  }
  if (periodic_y && num_cells_y < 2) {
    throw std::invalid_argument("honeycomb: periodic_y requires ny > 1.");
  }

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
  // Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009), Eq. (1), in units
  // of the nearest-neighbor distance.
  const Eigen::RowVector2d a1(1.5, std::sqrt(3.0) / 2.0);
  const Eigen::RowVector2d a2(1.5, -std::sqrt(3.0) / 2.0);
  const std::array<Eigen::RowVector2d, 2> basis = {
      Eigen::RowVector2d::Zero(), Eigen::RowVector2d(1.0, 0.0)};
  const auto full_positions = detail::lattice_positions(Nx, Ny, a1, a2, basis);
  Eigen::MatrixXd positions(num_sites, 2);
  for (int old_site = 0; old_site < full_num_sites; ++old_site) {
    if (old_to_new[old_site] >= 0) {
      positions.row(old_to_new[old_site]) = full_positions.row(old_site);
    }
  }

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

  return LatticeGraph(std::move(adj), std::move(coloring), std::move(positions),
                      detail::lattice_periods((periodic_x ? Nx : 0) * a1,
                                              (periodic_y ? Ny : 0) * a2));
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
  // Guo and Franz, Phys. Rev. B 80, 113102 (2009), Fig. 1 and the text
  // following Eq. (2): the Bravais periods are twice their unit directions.
  const Eigen::RowVector2d a1(2.0, 0.0);
  const Eigen::RowVector2d a2(1.0, std::sqrt(3.0));
  const std::array<Eigen::RowVector2d, 3> basis = {
      Eigen::RowVector2d(0.0, 0.0), Eigen::RowVector2d(1.0, 0.0),
      Eigen::RowVector2d(0.5, std::sqrt(3.0) / 2.0)};
  auto positions = detail::lattice_positions(Nx, Ny, a1, a2, basis);
  auto coloring = greedy_edge_coloring(adj, coloring_seed, 32);
  return LatticeGraph(std::move(adj), std::move(coloring), std::move(positions),
                      detail::lattice_periods((periodic_x ? Nx : 0) * a1,
                                              (periodic_y ? Ny : 0) * a2));
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

}  // namespace detail

// Greedy edge coloring: place each edge in the lowest-index color whose
// vertices do not already touch that color.  Optionally retry with shuffled
// edge orders and keep the result with fewest colors.
EdgeColoring greedy_edge_coloring(const Eigen::SparseMatrix<double>& adj,
                                  int seed, int trials) {
  auto edges_in = detail::undirected_edges(adj);
  if (edges_in.empty() || trials < 1) {
    return {};
  }

  // Compute max degree to bound the colour count.
  auto num_vertices = static_cast<std::size_t>(adj.rows());
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

  if (_positions.has_value()) {
    j["positions"] = matrix_to_json(*_positions);
  }
  if (_periods.has_value()) {
    j["periods"] = matrix_to_json(*_periods);
  }
  if (!_bond_flavor_definitions.empty()) {
    nlohmann::json definitions = nlohmann::json::array();
    for (const auto& definition : _bond_flavor_definitions) {
      definitions.push_back({definition.shell, definition.axis.x(),
                             definition.axis.y(),
                             static_cast<std::uint8_t>(definition.flavor)});
    }
    j["bond_flavor_definitions"] = std::move(definitions);
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
    dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);

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
      cds.write(cbuf.data(), H5::PredType::NATIVE_DOUBLE);
    }

    if (_positions.has_value()) {
      save_matrix_to_group(group, "positions", *_positions);
    }
    if (_periods.has_value()) {
      save_matrix_to_group(group, "periods", *_periods);
    }
    if (!_bond_flavor_definitions.empty()) {
      const auto count = static_cast<hsize_t>(_bond_flavor_definitions.size());
      std::vector<std::uint64_t> shells(count);
      std::vector<BondFlavorId> flavors(count);
      Eigen::MatrixXd axes(count, 2);
      for (hsize_t i = 0; i < count; ++i) {
        const auto& definition = _bond_flavor_definitions[i];
        shells[i] = definition.shell;
        axes.row(static_cast<Eigen::Index>(i)) = definition.axis;
        flavors[i] = definition.flavor;
      }
      H5::Group flavor_group = group.createGroup("bond_flavor_definitions");
      hsize_t dims[1] = {count};
      H5::DataSpace space(1, dims);
      auto shell_dataset = flavor_group.createDataSet(
          "shells", H5::PredType::NATIVE_UINT64, space);
      shell_dataset.write(shells.data(), H5::PredType::NATIVE_UINT64);
      save_matrix_to_group(flavor_group, "axes", axes);
      auto flavor_dataset = flavor_group.createDataSet(
          "flavors", H5::PredType::NATIVE_UINT32, space);
      flavor_dataset.write(flavors.data(), H5::PredType::NATIVE_UINT32);
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

  if (!j.contains("num_sites")) {
    throw std::runtime_error("JSON missing required 'num_sites' field");
  }
  if (!j.contains("adjacency_sparse")) {
    throw std::runtime_error("JSON missing required 'adjacency_sparse' field");
  }

  auto n = j["num_sites"].get<std::uint64_t>();
  auto n_idx = static_cast<int>(n);

  std::vector<detail::Triplet> triplets;
  for (const auto& entry : j["adjacency_sparse"]) {
    int row = entry[0].get<int>();
    int col = entry[1].get<int>();
    double val = entry[2].get<double>();
    if (row < 0 || row >= n_idx || col < 0 || col >= n_idx) {
      throw std::runtime_error(
          "Edge (" + std::to_string(row) + ", " + std::to_string(col) +
          ") has index out of range for num_sites=" + std::to_string(n) +
          " in JSON data.");
    }
    triplets.emplace_back(row, col, val);
  }
  Eigen::SparseMatrix<double> sparse(static_cast<Eigen::Index>(n),
                                     static_cast<Eigen::Index>(n));
  sparse.setFromTriplets(triplets.begin(), triplets.end());
  sparse.makeCompressed();

  std::optional<EdgeColoring> coloring;
  if (j.contains("edge_coloring")) {
    EdgeColoring c;
    for (const auto& entry : j["edge_coloring"]) {
      auto i = entry[0].get<std::uint64_t>();
      auto k = entry[1].get<std::uint64_t>();
      auto color = entry[2].get<int>();
      c[{i, k}] = color;
    }
    coloring = std::move(c);
  }

  std::optional<Eigen::MatrixXd> positions;
  if (j.contains("positions")) {
    positions = json_to_matrix(j["positions"]);
  }
  std::optional<Eigen::MatrixXd> periods;
  if (j.contains("periods")) {
    periods = json_to_matrix(j["periods"]);
  }
  std::vector<BondFlavorDefinition> bond_flavors;
  if (j.contains("bond_flavor_definitions")) {
    for (const auto& entry : j["bond_flavor_definitions"]) {
      if (!entry.is_array() || entry.size() != 4 ||
          !entry[0].is_number_unsigned() || !entry[3].is_number_unsigned()) {
        throw std::runtime_error(
            "Bond-flavor definitions require unsigned integer shell and flavor "
            "values.");
      }
      const auto shell = entry[0].get<std::uint64_t>();
      const auto flavor = entry[3].get<std::uint64_t>();
      if (shell == 0 || flavor > std::numeric_limits<BondFlavorId>::max()) {
        throw std::runtime_error(
            "Invalid bond-flavor definition in JSON data.");
      }
      bond_flavors.push_back({shell,
                              {entry[1].get<double>(), entry[2].get<double>()},
                              static_cast<BondFlavorId>(flavor)});
    }
  }

  return LatticeGraph(std::move(sparse), std::move(coloring),
                      std::move(positions), std::move(periods),
                      std::move(bond_flavors));
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
  H5::DataSet dataset = group.openDataSet("adjacency_sparse");

  // Read num_sites from group attribute (required)
  if (!group.attrExists("num_sites")) {
    throw std::runtime_error(
        "HDF5 group missing required 'num_sites' attribute for LatticeGraph.");
  }
  std::uint64_t n = 0;
  H5::Attribute sites_attr = group.openAttribute("num_sites");
  sites_attr.read(H5::PredType::NATIVE_UINT64, &n);

  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[2];
  dataspace.getSimpleExtentDims(dims);
  auto nnz = dims[0];

  std::vector<double> buffer(nnz * 3);
  dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);

  auto n_idx = static_cast<int>(n);

  using T = Eigen::Triplet<double>;
  std::vector<T> triplets;
  triplets.reserve(nnz);
  for (hsize_t i = 0; i < nnz; ++i) {
    int row = static_cast<int>(buffer[i * 3 + 0]);
    int col = static_cast<int>(buffer[i * 3 + 1]);
    double val = buffer[i * 3 + 2];
    if (row < 0 || row >= n_idx || col < 0 || col >= n_idx) {
      throw std::runtime_error(
          "Edge (" + std::to_string(row) + ", " + std::to_string(col) +
          ") has index out of range for num_sites=" + std::to_string(n) +
          " in HDF5 data.");
    }
    triplets.emplace_back(row, col, val);
  }

  Eigen::SparseMatrix<double> sparse(static_cast<Eigen::Index>(n),
                                     static_cast<Eigen::Index>(n));
  sparse.setFromTriplets(triplets.begin(), triplets.end());
  sparse.makeCompressed();

  std::optional<EdgeColoring> coloring;
  if (group.nameExists("edge_coloring")) {
    H5::DataSet cds = group.openDataSet("edge_coloring");
    H5::DataSpace cspace = cds.getSpace();
    hsize_t cdims[2];
    cspace.getSimpleExtentDims(cdims);
    auto nc = cdims[0];
    std::vector<double> cbuf(nc * 3);
    cds.read(cbuf.data(), H5::PredType::NATIVE_DOUBLE);
    EdgeColoring c;
    for (hsize_t ci = 0; ci < nc; ++ci) {
      auto ei = static_cast<std::uint64_t>(cbuf[ci * 3 + 0]);
      auto ej = static_cast<std::uint64_t>(cbuf[ci * 3 + 1]);
      auto color = static_cast<int>(cbuf[ci * 3 + 2]);
      c[{ei, ej}] = color;
    }
    coloring = std::move(c);
  }

  std::optional<Eigen::MatrixXd> positions;
  if (group.nameExists("positions")) {
    positions = load_matrix_from_group(group, "positions");
  }
  std::optional<Eigen::MatrixXd> periods;
  if (group.nameExists("periods")) {
    periods = load_matrix_from_group(group, "periods");
  }
  std::vector<BondFlavorDefinition> bond_flavors;
  if (group.nameExists("bond_flavor_definitions")) {
    H5::Group flavor_group = group.openGroup("bond_flavor_definitions");
    if (!flavor_group.nameExists("shells") ||
        !flavor_group.nameExists("axes") ||
        !flavor_group.nameExists("flavors")) {
      throw std::runtime_error(
          "Bond-flavor definitions require shells, axes, and flavors "
          "datasets.");
    }
    H5::DataSet shell_dataset = flavor_group.openDataSet("shells");
    H5::DataSet axis_dataset = flavor_group.openDataSet("axes");
    H5::DataSet flavor_dataset = flavor_group.openDataSet("flavors");
    H5::DataSpace shell_space = shell_dataset.getSpace();
    H5::DataSpace axis_space = axis_dataset.getSpace();
    H5::DataSpace flavor_space = flavor_dataset.getSpace();
    hsize_t shell_dims[1];
    hsize_t axis_dims[2];
    hsize_t flavor_dims[1];
    if (shell_space.getSimpleExtentNdims() != 1 ||
        axis_space.getSimpleExtentNdims() != 2 ||
        flavor_space.getSimpleExtentNdims() != 1) {
      throw std::runtime_error("Invalid bond-flavor dataset rank.");
    }
    shell_space.getSimpleExtentDims(shell_dims);
    axis_space.getSimpleExtentDims(axis_dims);
    flavor_space.getSimpleExtentDims(flavor_dims);
    if (axis_dims[0] != shell_dims[0] || axis_dims[1] != 2 ||
        flavor_dims[0] != shell_dims[0] ||
        shell_dataset.getTypeClass() != H5T_INTEGER ||
        shell_dataset.getIntType().getSign() != H5T_SGN_NONE ||
        axis_dataset.getTypeClass() != H5T_FLOAT ||
        flavor_dataset.getTypeClass() != H5T_INTEGER ||
        flavor_dataset.getIntType().getSign() != H5T_SGN_NONE) {
      throw std::runtime_error("Invalid bond-flavor dataset shape or type.");
    }
    std::vector<std::uint64_t> shells(shell_dims[0]);
    std::vector<std::uint64_t> flavors(shell_dims[0]);
    Eigen::MatrixXd axes(shell_dims[0], 2);
    shell_dataset.read(shells.data(), H5::PredType::NATIVE_UINT64);
    axis_dataset.read(axes.data(), H5::PredType::NATIVE_DOUBLE);
    flavor_dataset.read(flavors.data(), H5::PredType::NATIVE_UINT64);
    for (hsize_t i = 0; i < shell_dims[0]; ++i) {
      if (shells[i] == 0 ||
          flavors[i] > std::numeric_limits<BondFlavorId>::max()) {
        throw std::runtime_error(
            "Invalid bond-flavor definition in HDF5 data.");
      }
      bond_flavors.push_back(
          {shells[i], axes.row(i), static_cast<BondFlavorId>(flavors[i])});
    }
  }

  return LatticeGraph(std::move(sparse), std::move(coloring),
                      std::move(positions), std::move(periods),
                      std::move(bond_flavors));
}

void LatticeGraph::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, static_cast<uint64_t>(_num_sites));
  hash_value(ctx, adjacency_);
  hash_value(ctx, _is_symmetric);
  hash_value(ctx, _positions.has_value());
  if (_positions.has_value()) {
    hash_value(ctx, *_positions);
  }
  hash_value(ctx, _periods.has_value());
  if (_periods.has_value()) {
    hash_value(ctx, *_periods);
  }
  for (const auto& definition : _bond_flavor_definitions) {
    hash_value(ctx, definition.shell);
    hash_value(ctx, definition.axis);
    hash_value(ctx, definition.flavor);
  }
}

LatticeGraph LatticeGraph::permute(const LatticeGraph& graph,
                                   const std::vector<std::uint64_t>& path) {
  std::uint64_t V = graph.num_sites();
  const auto& adj = graph.sparse_adjacency_matrix();

  // Reorder the adjacency matrix using Eigen's PermutationMatrix
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P(
      static_cast<Eigen::Index>(V));
  for (std::uint64_t i = 0; i < V; ++i) {
    P.indices()[static_cast<Eigen::Index>(i)] = static_cast<int>(path[i]);
  }
  Eigen::SparseMatrix<double> new_adj = P.transpose() * adj * P;
  new_adj.makeCompressed();

  std::vector<std::uint64_t> inv_p(V);
  for (std::uint64_t i = 0; i < V; ++i) {
    inv_p[path[i]] = i;
  }

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

  std::optional<Eigen::MatrixXd> new_positions;
  if (graph._positions.has_value()) {
    new_positions.emplace(static_cast<Eigen::Index>(V), 2);
    for (std::uint64_t i = 0; i < V; ++i) {
      new_positions->row(static_cast<Eigen::Index>(i)) =
          graph._positions->row(static_cast<Eigen::Index>(path[i]));
    }
  }

  return LatticeGraph(std::move(new_adj), std::move(new_coloring),
                      std::move(new_positions), graph._periods,
                      graph._bond_flavor_definitions);
}

}  // namespace qdk::chemistry::data
