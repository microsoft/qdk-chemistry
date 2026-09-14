// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <map>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/lattice_geometry.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Edge coloring as a map from ordered (i, j) (with i < j) to a
 *        non-negative integer color label.
 *
 * Two edges sharing the same color have disjoint vertex sets.
 */
using EdgeColoring = std::map<std::pair<std::uint64_t, std::uint64_t>, int>;

/** @brief Optional semantic label for one shell and geometric bond axis. */
struct BondFlavorDefinition {
  std::uint64_t shell;
  Eigen::RowVector2d axis;
  BondFlavorId flavor;
};

// ---- Free coloring functions ------------------------------------------------
// These compute edge colorings for known lattice topologies.  They are
// called by the factory methods to pre-populate the coloring at
// construction time, and can also be called directly by users who need
// a coloring for a topology not covered by the built-in factories.

/**
 * @brief Greedy randomised edge coloring of an arbitrary graph.
 *
 * Shuffles the edge order and assigns each edge the lowest colour not
 * incident to either endpoint.  Repeats for ``trials`` shuffles (with
 * deterministic PRNG seeded by ``seed``) and returns the result with
 * the fewest colours.
 *
 * @param adj   Sparse adjacency matrix of the graph.
 * @param seed  Random seed.  Default: 0.
 * @param trials Number of random-order trials.  Default: 1.
 * @return Edge coloring with the fewest distinct colours found.
 */
EdgeColoring greedy_edge_coloring(const Eigen::SparseMatrix<double>& adj,
                                  int seed = 0, int trials = 1);

/**
 * @brief Deterministic optimal edge coloring for a chain (path / ring).
 *
 * @param n        Number of sites in the chain.
 * @param periodic Whether the chain wraps around (ring topology).
 * @return Edge coloring using 2 colours (open or even-periodic) or 3
 *         colours (odd-periodic).
 */
EdgeColoring chain_coloring(std::int64_t n, bool periodic);

/**
 * @brief Deterministic optimal edge coloring for a square lattice.
 *
 * @param nx         Number of sites along x.
 * @param ny         Number of sites along y.
 * @param periodic_x Whether periodic boundary conditions are applied along x.
 * @param periodic_y Whether periodic boundary conditions are applied along y.
 * @return Edge coloring using 2–4 colours depending on periodicity and parity.
 */
EdgeColoring square_coloring(std::int64_t nx, std::int64_t ny, bool periodic_x,
                             bool periodic_y);

/**
 * @brief Deterministic optimal 3-coloring for a honeycomb lattice.
 *
 * @param nx         Number of unit cells along x.
 * @param ny         Number of unit cells along y.
 * @param periodic_x Whether periodic boundary conditions are applied along x.
 * @param periodic_y Whether periodic boundary conditions are applied along y.
 * @return Edge coloring using exactly 3 colours (one per bond type).
 */
EdgeColoring honeycomb_coloring(std::int64_t nx, std::int64_t ny,
                                bool periodic_x, bool periodic_y);

/**
 * @brief Trivial edge coloring where every edge receives a unique color.
 *
 * Useful as a fallback when no topology-aware coloring is available.
 *
 * @param adj Sparse adjacency matrix of the graph.
 * @return Edge coloring mapping each undirected edge to a distinct colour
 *         label 0, 1, 2, … in iteration order.
 */
EdgeColoring trivial_edge_coloring(const Eigen::SparseMatrix<double>& adj);

/**
 * @brief Weighted graph representing a lattice connectivity structure.
 *
 * Explicit connection graphs retain physical images, shell-axis classes,
 * weights, and optional semantic flavors. Their sparse adjacency is a cached
 * symmetric sum of connection weights; self-image weights contribute once to
 * the diagonal. Adjacency-only graphs retain directed, asymmetric input
 * without inventing geometric labels. Geometry is a separate immutable object.
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
   * @brief Materialize the requested geometric shells exactly once.
   * @param geometry Source geometry, copied into shared immutable storage.
   * @param shells Positive shell indices, sorted and deduplicated on storage.
   * @param definitions Optional shell-axis flavor assignments.
   * @param weight Finite weight assigned to every physical connection.
   * @param tolerance Positive finite distance and axis tolerance.
   * @return Graph retaining selected shells even when they are empty.
   */
  static LatticeGraph from_geometry(
      const LatticeGeometry& geometry,
      const std::vector<std::uint64_t>& shells = {1},
      const std::vector<BondFlavorDefinition>& definitions = {},
      double weight = 1.0, double tolerance = 1.0e-9);

  /**
   * @brief Construct a graph from resolved physical connections.
   *
   * Records are authoritative; geometry is not queried to relabel them.
   * Reversed endpoints and negative self-images are canonicalized together
   * with displacement and image shift. A self-image must have a nonzero image.
   * Duplicate canonical endpoint/image records are rejected across all shells.
   * Orientation IDs are unsigned shell-local labels, not necessarily contiguous
   * in a selected subset. Axes must be finite unit vectors and displacements
   * finite and nonzero. Images may have different weights and flavors.
   *
   * @param num_sites Number of vertices, including isolated sites.
   * @param connections Physical connections with finite weights.
   * @param geometry Optional geometry with the same number of sites.
   * @param selected_shells Additional selected shells, including empty shells.
   * @return Graph with sorted connections and selected/record shells combined.
   * @throws std::invalid_argument If records, shells, or geometry are invalid.
   * @throws std::overflow_error If indices, images, or weights overflow.
   */
  static LatticeGraph from_connections(
      std::uint64_t num_sites, std::vector<NeighborConnection> connections,
      std::shared_ptr<const LatticeGeometry> geometry = nullptr,
      std::vector<std::uint64_t> selected_shells = {});

  /**
   * @brief Return a new lattice graph with reverse edges added.
   *
   * Computes A_out = A + A^T, doubling weights already present symmetrically.
   * Explicit connection weights are doubled, including self-images, and any
   * stored topology coloring is cleared.
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

  /** @brief Shared immutable geometry, or nullptr for geometry-free graphs. */
  const std::shared_ptr<const LatticeGeometry>& geometry() const;

  /** @brief Sorted unique selected shells, including empty shells. */
  const std::vector<std::uint64_t>& selected_shells() const;

  /** @brief Connections ordered by shell, orientation, sites, image. */
  const std::vector<NeighborConnection>& connections() const;

  /**
   * @brief Replace flavors on existing records without discovering connections.
   * @param definitions Shell-axis labels; unmatched records become unlabeled.
   * @param tolerance Positive finite tolerance for normalized axis comparisons.
   * @return Copy with unchanged selection, geometry, weights, and topology.
   * @throws std::invalid_argument If definitions are invalid or axes duplicate.
   */
  LatticeGraph with_bond_flavors(
      const std::vector<BondFlavorDefinition>& definitions,
      double tolerance = 1.0e-9) const;

  /**
   * @brief Greedily color only the supplied active simple support.
   * @param active_pairs Canonical pairs i < j; duplicate pairs are ignored.
   * @param seed Random seed, with the same traversal as greedy_edge_coloring.
   * @param trials Number of trials; fewer than one returns an empty coloring.
   * @return Coloring independent of weights and stored topology colors.
   * @throws std::invalid_argument If any pair is noncanonical or out of bounds.
   */
  EdgeColoring color_edges(
      std::vector<std::pair<std::uint64_t, std::uint64_t>> active_pairs,
      int seed = 0, int trials = 32) const;

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
   * @param dfs_ordering If true, relabel sites in Hamiltonian-path order found
   *                     by depth-first search. Default: false.
   * @throws std::invalid_argument If n == 0.
   */
  static LatticeGraph chain(std::uint64_t n, bool periodic = false,
                            double t = 1.0, bool dfs_ordering = false);

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
   * @param dfs_ordering If true, relabel sites in Hamiltonian-path order found
   *                     by depth-first search. Default: false.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph square(std::uint64_t nx, std::uint64_t ny,
                             bool periodic_x = false, bool periodic_y = false,
                             double t = 1.0, bool dfs_ordering = false);

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
   * @param coloring_seed PRNG seed for greedy edge coloring. Default: 0.
   * @param dfs_ordering If true, relabel sites in Hamiltonian-path order found
   *                     by depth-first search. Default: false.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph triangular(std::uint64_t nx, std::uint64_t ny,
                                 bool periodic_x = false,
                                 bool periodic_y = false, double t = 1.0,
                                 int coloring_seed = 0,
                                 bool dfs_ordering = false);

  /**
   * @brief Create a two-dimensional honeycomb lattice.
   *
   * The honeycomb lattice has two sites per unit cell (A and B sublattices).
   * Unit cells are arranged on a rectangular grid of size nx x ny, giving a
   * total of 2 * nx * ny sites. Sites are indexed as:
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
   * @param nx         Number of unit cells along the x-axis.
   * @param ny         Number of unit cells along the y-axis.
   * @param periodic_x If true, apply periodic boundary conditions along x.
   * Requires nx > 1. Default: false.
   * @param periodic_y If true, apply periodic boundary conditions along y.
   * Requires ny > 1. Default: false.
   * @param t          Uniform hopping weight. Default: 1.0.
   * @param dfs_ordering Reserved for API compatibility; currently ignored.
   *                     Default: false.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph honeycomb(std::uint64_t nx, std::uint64_t ny,
                                bool periodic_x = false,
                                bool periodic_y = false, double t = 1.0,
                                bool dfs_ordering = false);

  /**
   * @brief Create a honeycomb patch sized by complete hexagonal plaquettes.
   *
   * Open directions include the boundary sites needed to complete every
   * requested plaquette. A fully open 1 x 1 patch is one six-site hexagon.
   *
   * @code
   *   1x1 open plaquette patch:
   *
   *       1---2
   *      /     \
   *     0       5
   *      \     /
   *       3---4
   * @endcode
   *
   * @param nx Number of complete plaquettes along x.
   * @param ny Number of complete plaquettes along y.
   * @param periodic_x If true, apply periodic boundary conditions along x.
   * @param periodic_y If true, apply periodic boundary conditions along y.
   * @param t Uniform hopping weight.
   * @param dfs_ordering Reserved for API compatibility; currently ignored.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph honeycomb_plaquettes(std::uint64_t nx, std::uint64_t ny,
                                           bool periodic_x = false,
                                           bool periodic_y = false,
                                           double t = 1.0,
                                           bool dfs_ordering = false);

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
   * @param coloring_seed PRNG seed for greedy edge coloring. Default: 0.
   * @param dfs_ordering Reserved for API compatibility; currently ignored.
   *                     Default: false.
   * @throws std::invalid_argument If nx or ny is 0.
   */
  static LatticeGraph kagome(std::uint64_t nx, std::uint64_t ny,
                             bool periodic_x = false, bool periodic_y = false,
                             double t = 1.0, int coloring_seed = 0,
                             bool dfs_ordering = false);

  /**
   * @brief Edge coloring stored at construction time, if any.
   *
   * Factory methods for recognised topologies pre-populate this field.
   * Returns ``std::nullopt`` for lattices constructed without a coloring.
   *
   * @return Reference to the optional edge coloring.
   */
  const std::optional<EdgeColoring>& edge_coloring() const;

  /**
   * @brief Get the static data type name for this class.
   * @return "lattice_graph"
   */
  static std::string data_type_name() {
    return DATACLASS_TO_SNAKE_CASE(LatticeGraph);
  }

  /**
   * @brief Get the data type name for this instance.
   * @return "lattice_graph"
   */
  std::string get_data_type_name() const override { return data_type_name(); }

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
   * @brief Serialize the graph's connectivity and optional geometry to JSON.
   *
   * Adjacency-only graphs use sparse triplets. Explicit graphs store resolved
   * records, selected shells, nested geometry, and a checked adjacency cache
   * retaining exact factory weights and zero entries. Empty connections remain
   * distinct from absent connection metadata.
   *
   * @return JSON representation of the graph.
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
   * @brief Read resolved records or migrate a legacy adjacency payload.
   * @param j Record-based or legacy adjacency JSON, including num_sites.
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

  /**
   * @brief Permutes the vertices of the lattice graph according to the given
   * path.
   *
   * Reorders the sparse adjacency matrix using Eigen's permutation operations
   * and updates the edge coloring to align with the new vertex indexing.
   *
   * @param graph The source lattice graph to permute.
   * @param path The sequence of original vertex indices representing the target
   * permutation.
   * @return A new LatticeGraph with the permuted adjacency matrix and edge
   * coloring.
   * @throws std::invalid_argument If path omits or repeats a lattice site.
   */
  static LatticeGraph permute(const LatticeGraph& graph,
                              const std::vector<std::uint64_t>& path);

 private:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  explicit LatticeGraph(Eigen::SparseMatrix<double> adjacency,
                        std::optional<EdgeColoring> coloring = std::nullopt);

  // Preserve legacy factory topology while distributing each pair's weight
  // over its shell-one physical images (also used for legacy deserialization).
  static LatticeGraph _with_geometry(
      Eigen::SparseMatrix<double> adjacency,
      std::optional<EdgeColoring> coloring,
      std::shared_ptr<const LatticeGeometry> geometry);
  void _restore_adjacency(Eigen::SparseMatrix<double> adjacency);
  void _validate_coloring() const;

  static LatticeGraph _honeycomb(
      std::uint64_t num_cells_x, std::uint64_t num_cells_y,
      bool remove_open_corners, bool periodic_x, bool periodic_y, double t,
      std::shared_ptr<const LatticeGeometry> geometry);

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
  /// Edge coloring, populated at construction for recognised topologies.
  std::optional<EdgeColoring> _edge_coloring;
  std::shared_ptr<const LatticeGeometry> _geometry;
  std::vector<std::uint64_t> _selected_shells;
  std::vector<NeighborConnection> _connections;
  /// Distinguishes an explicitly empty selection from an adjacency-only graph.
  bool _has_connections = false;
};

static_assert(DataClassCompliant<LatticeGraph>,
              "LatticeGraph must derive from DataClass and implement all "
              "required deserialization methods");

}  //  namespace qdk::chemistry::data
