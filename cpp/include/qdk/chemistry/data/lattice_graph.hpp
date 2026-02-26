// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <map>
#include <nlohmann/json_fwd.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Weighted graph representing a lattice connectivity structure.
 *
 * Stores the lattice topology as a sparse adjacency matrix and provides
 * static factory methods for common lattice geometries. Used by model
 * Hamiltonian builders to define site connectivity and hopping integrals.
 */
class LatticeGraph : public DataClass {
 public:
  /**
   * @brief Construct a lattice graph from an edge-weight map.
   *
   * Each key is a pair (i, j) of site indices and each value is the
   * corresponding edge weight. Edges are stored exactly as given; use
   * make_bidirectional() to add reverse edges from one-directional input.
   *
   * @param edge_weights Map of (source, target) -> weight.
   * @param num_sites   Total number of sites. If 0, inferred from the
   *                    largest index in edge_weights.
   */
  LatticeGraph(const std::map<std::pair<std::uint64_t, std::uint64_t>, double>&
                   edge_weights,
               std::uint64_t num_sites = 0);

  /**
   * @brief Create a lattice graph from a dense adjacency matrix.
   *
   * @param adjacency_matrix Square dense matrix of edge weights.
   * @return LatticeGraph with the given adjacency.
   * @throws std::invalid_argument If the matrix is not square.
   */
  static LatticeGraph from_dense_matrix(
      const Eigen::MatrixXd& adjacency_matrix);

  /**
   * @brief Create a lattice graph from a sparse adjacency matrix.
   *
   * @param sparse Sparse square matrix of edge weights.
   * @return LatticeGraph with the given adjacency.
   * @throws std::invalid_argument If the matrix is not square.
   */
  static LatticeGraph from_sparse_matrix(
      const Eigen::SparseMatrix<double>& sparse);

  /**
   * @brief Return a new lattice graph with reverse edges added.
   *
   * For each directed edge (i,j) with weight w, ensures (j,i) also exists
   * with the same weight. Computes A_out = A + A^T.
   *
   * @param graph The (possibly directed) lattice graph.
   * @return A new LatticeGraph with bidirectional edges.
   */
  static LatticeGraph make_bidirectional(const LatticeGraph& graph);

  ~LatticeGraph() = default;

  /**
   * @brief Return the number of sites (vertices) in the lattice.
   */
  std::uint64_t num_sites() const;

  /**
   * @brief Return a const reference to the internal sparse adjacency matrix.
   */
  const Eigen::SparseMatrix<double>& sparse_adjacency_matrix() const;

  /**
   * @brief Return a dense copy of the adjacency matrix.
   */
  Eigen::MatrixXd adjacency_matrix() const;

  /**
   * @brief Return whether the adjacency matrix is symmetric.
   */
  bool is_symmetric() const;

  /**
   * @brief Return the edge weight between sites i and j.
   *
   * Returns 0.0 when the sites are not connected.
   *
   * @param i Source site index.
   * @param j Target site index.
   */
  double weight(std::uint64_t i, std::uint64_t j) const;

  /**
   * @brief Check whether sites i and j are connected.
   *
   * Equivalent to weight(i, j) != 0.0.
   *
   * @param i First site index.
   * @param j Second site index.
   * @return True if the edge weight is non-zero.
   */
  bool are_connected(std::uint64_t i, std::uint64_t j) const;

  /**
   * @brief Return the total number of stored non-zero entries in the
   *        sparse adjacency matrix.
   *
   * For a symmetric undirected graph this is twice the number of edges
   * (each edge is stored in both directions).
   */
  std::uint64_t num_nonzeros() const;

  /**
   * @brief Return the number of undirected edges.
   *
   * Counts only upper-triangular entries (row < col) so that each
   * undirected edge is counted once.
   */
  std::uint64_t num_edges() const;

  /**
   * @brief Create a one-dimensional chain lattice.
   *
   * Sites are labelled 0 ... n-1 with nearest-neighbour edges.
   *
   * @code
   *   Example: chain (n=4)
   *
   *     0 --- 1 --- 2 --- 3
   *
   * @endcode
   *
   *   With periodic boundary condition:
   *     - Wrap bond: (n-1) -- 0  e.g. 3 -- 0
   *
   * @param n        Number of sites.
   * @param periodic If true, add an edge between the first and last site
   *                 (ring topology). Requires n > 2. Default: false.
   * @param t        Uniform hopping weight for every edge. Default: 1.0.
   * @throws std::invalid_argument If n == 0.
   */
  static LatticeGraph chain(std::uint64_t n, bool periodic = false,
                            double t = 1.0);

  /**
   * @brief Create a two-dimensional square lattice.
   *
   * Sites are indexed in row-major order: site index = y * nx + x.
   * Total sites: nx * ny.
   *
   * @code
   *   Example: 4x3 square lattice
   *
   *     8 --- 9 ---10 ---11
   *     |     |     |     |
   *     4 --- 5 --- 6 --- 7
   *     |     |     |     |
   *     0 --- 1 --- 2 --- 3
   *
   * @endcode
   *
   * With periodic boundary conditions (using the 4x3 example above):
   *   - periodic_x wraps right to left:  3 -- 0, 7 -- 4, 11 -- 8
   *   - periodic_y wraps top to bottom:  8 -- 0, 9 -- 1, 10 -- 2, 11 -- 3
   *
   * @param nx         Number of sites along the x-axis.
   * @param ny         Number of sites along the y-axis.
   * @param periodic_x If true, apply periodic boundary conditions along x.
   * Requires nx >= 2. Default: false.
   * @param periodic_y If true, apply periodic boundary conditions along y.
   * Requires ny >= 2. Default: false.
   * @param t          Uniform hopping weight. Default: 1.0.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph square(std::uint64_t nx, std::uint64_t ny,
                             bool periodic_x = false, bool periodic_y = false,
                             double t = 1.0);

  /**
   * @brief Create a two-dimensional triangular lattice.
   *
   * Sites are indexed in row-major order: site index = y * nx + x.
   * Total sites: nx * ny. Each site connects to its right and upper
   * square-lattice neighbours plus the upper-right diagonal neighbour,
   * forming a triangulation of the plane.
   *
   * @code
   *   Example: 3x3 triangular lattice
   *
   *      6 --- 7 --- 8
   *      |  /  |  /  |
   *      3 --- 4 --- 5
   *      |  /  |  /  |
   *      0 --- 1 --- 2
   *
   * @endcode
   *
   * With periodic boundary conditions (using the 3x3 example above):
   *   - periodic_x wraps right to left:  2 -- 0, 5 -- 3, 8 -- 6
   *   - periodic_y wraps top to bottom:  6 -- 0, 7 -- 1, 8 -- 2
   *   - Diagonal wraps require both periodic_x and periodic_y: 8 -- 0
   *
   * @param nx         Number of sites along the x-axis.
   * @param ny         Number of sites along the y-axis.
   * @param periodic_x If true, apply periodic boundary conditions along x.
   * Requires nx >= 2. Default: false.
   * @param periodic_y If true, apply periodic boundary conditions along y.
   * Requires ny >= 2. Default: false.
   * @param t          Uniform hopping weight. Default: 1.0.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph triangular(std::uint64_t nx, std::uint64_t ny,
                                 bool periodic_x = false,
                                 bool periodic_y = false, double t = 1.0);

  /**
   * @brief Create a two-dimensional honeycomb lattice.
   *
   * The honeycomb lattice has two sites per unit cell (A and B sublattices).
   * Unit cells are arranged on a rectangular grid of size nx x ny, giving a
   * total of 2 * nx * ny sites.  Sites are indexed as:
   *   - A-sublattice: 2 * (y * nx + x)
   *   - B-sublattice: 2 * (y * nx + x) + 1
   *
   * @code
   *   Example: 3x4 honeycomb
   *
   *               18-19-20-21-22-23
   *                |     |     |
   *            12-13-14-15-16-17
   *             |     |     |
   *          6--7--8--9-10-11
   *          |     |     |
   *       0--1--2--3--4--5
   *
   * @endcode
   *
   * With periodic boundary conditions (using the 3x4 example above):
   *   - periodic_x wraps right to left: 5 -- 0, 11 -- 6, 17 -- 12, 23 -- 18
   *   - periodic_y wraps top to bottom: 19 -- 0, 15 -- 2, 17 -- 4
   *
   * @param nx         Number of unit cells along the x-axis.
   * @param ny         Number of unit cells along the y-axis.
   * @param periodic_x If true, apply periodic boundary conditions along x.
   * Requires nx >= 2. Default: false.
   * @param periodic_y If true, apply periodic boundary conditions along y.
   * Requires ny >= 2. Default: false.
   * @param t          Uniform hopping weight. Default: 1.0.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph honeycomb(std::uint64_t nx, std::uint64_t ny,
                                bool periodic_x = false,
                                bool periodic_y = false, double t = 1.0);

  /**
   * @brief Create a two-dimensional kagome lattice.
   *
   * The kagome lattice has three sites per unit cell, arranged as
   * corner-sharing triangles.  Unit cells are on a rectangular grid of
   * size nx x ny, giving a total of 3 * nx * ny sites.  Sites are indexed
   * as:
   *   - site 0: 3 * (y * nx + x)
   *   - site 1: 3 * (y * nx + x) + 1
   *   - site 2: 3 * (y * nx + x) + 2
   *
   * @code
   *   Unit cell (up-triangle):
   *
   *           2
   *          / \
   *         0---1
   *
   *   Example: 3x2 kagome
   *
   *         11       14       17
   *        /  \     /  \     /  \
   *       9---10--12---13--15---16
   *      /     \  /     \  /
   *     2       5        8
   *    / \     / \      / \
   *   0---1---3---4----6---7
   *
   * @endcode
   *
   * With periodic boundary conditions (using the 3x2 example above):
   *   - periodic_x wraps right to left: 0 -- 7, 9 -- 16, 2 -- 16
   *   - periodic_y wraps top to bottom: 0 -- 11, 3 -- 14, 6 -- 17, 1 -- 14, 4
   * -- 17
   *   - Diagonal wraps (require both periodic_x and periodic_y): 7 -- 11
   *
   * @param nx         Number of unit cells along the x-axis.
   * @param ny         Number of unit cells along the y-axis.
   * @param periodic_x If true, apply periodic boundary conditions along x.
   * Requires nx >= 2. Default: false.
   * @param periodic_y If true, apply periodic boundary conditions along y.
   * Requires ny >= 2. Default: false.
   * @param t          Uniform hopping weight. Default: 1.0.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph kagome(std::uint64_t nx, std::uint64_t ny,
                             bool periodic_x = false, bool periodic_y = false,
                             double t = 1.0);

  /**
   * @brief Get the data type name for this class.
   * @return "lattice_graph"
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(LatticeGraph);
  }

  /**
   * @brief Get a human-readable summary of the lattice graph.
   * @return Multi-line string with site/edge counts and symmetry info.
   */
  std::string get_summary() const override;

  /**
   * @brief Save lattice graph to file in the specified format.
   * @param filename Path to the output file.
   * @param type Format type ("json" or "hdf5").
   * @throws std::invalid_argument If format type is not supported.
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert lattice graph to JSON representation.
   *
   * Stores the sparse adjacency matrix (row-major) and the symmetry flag.
   *
   * @return JSON object containing the serialised data.
   */
  nlohmann::json to_json() const override;

  /** @brief Save lattice graph to a JSON file. */
  void to_json_file(const std::string& filename) const override;

  /** @brief Save lattice graph to an HDF5 group. */
  void to_hdf5(H5::Group& group) const override;

  /** @brief Save lattice graph to an HDF5 file. */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load a lattice graph from file.
   * @param filename Path to the input file.
   * @param type Format type ("json" or "hdf5").
   * @return New LatticeGraph instance.
   */
  static LatticeGraph from_file(const std::string& filename,
                                const std::string& type);

  /** @brief Load a lattice graph from a JSON file. */
  static LatticeGraph from_json_file(const std::string& filename);

  /**
   * @brief Load a lattice graph from a JSON object.
   * @param j JSON object (must contain "adjacency_matrix" and "is_symmetric").
   * @return New LatticeGraph instance.
   */
  static LatticeGraph from_json(const nlohmann::json& j);

  /** @brief Load a lattice graph from an HDF5 file. */
  static LatticeGraph from_hdf5_file(const std::string& filename);

  /**
   * @brief Load a lattice graph from an HDF5 group.
   * @param group HDF5 group to read from.
   * @return New LatticeGraph instance.
   */
  static LatticeGraph from_hdf5(H5::Group& group);

 private:
  /**
   * @brief Private constructor from a sparse adjacency matrix.
   *
   * Used internally by factory methods, deserialization, and
   * make_bidirectional().
   *
   * @param adjacency Sparse square adjacency matrix (moved in).
   */
  explicit LatticeGraph(Eigen::SparseMatrix<double> adjacency);

  /** @brief Check if a sparse matrix is symmetric within a numerical tolerance.
   */
  static bool _check_symmetry(const Eigen::SparseMatrix<double>& mat);

  /// Number of sites (vertices) in the lattice
  std::uint64_t _num_sites;
  /// Sparse adjacency matrix storing edge weights (shape: num_sites x
  /// num_sites)
  Eigen::SparseMatrix<double> adjacency_;
  /// Flag indicating whether the adjacency matrix is symmetric (undirected
  /// graph)
  bool _is_symmetric;
};

static_assert(DataClassCompliant<LatticeGraph>,
              "LatticeGraph must derive from DataClass and implement all "
              "required deserialization methods");

}  //  namespace qdk::chemistry::data
