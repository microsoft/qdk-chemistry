// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cstdint>
#include <map>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <qdk/chemistry/data/data_class.hpp>
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

/** @brief Semantic spin-interaction flavor assigned to a geometric bond class.
 */
enum class BondFlavor : std::uint8_t { X, Y, Z };

/** @brief Interpretation of honeycomb lattice dimensions. */
enum class HoneycombSizeConvention : std::uint8_t {
  UnitCells,
  CompletePlaquettes
};

/** @brief A radial shell and unoriented geometric bond-axis class. */
struct BondClass {
  std::uint64_t shell;
  std::uint32_t orientation;
  Eigen::RowVector2d axis;
};

/** @brief Optional semantic label for one shell and geometric bond axis. */
struct BondFlavorDefinition {
  std::uint64_t shell;
  Eigen::RowVector2d axis;
  BondFlavor flavor;
};

/** @brief One physical lattice connection, including its periodic image. */
struct NeighborConnection {
  std::uint64_t site_i;
  std::uint64_t site_j;
  BondClass bond_class;
  Eigen::RowVector2d displacement;
  std::array<std::int64_t, 2> image_shift;
  std::optional<BondFlavor> flavor;
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
   * @brief Return the Cartesian site positions, if this graph has geometry.
   * @return A (num_sites, 2) position matrix, or std::nullopt.
   */
  const std::optional<Eigen::MatrixXd>& positions() const;

  /**
   * @brief Return all pairs in the m-th geometric neighbor shell.
   *
   * The shells are the distinct positive minimum-image distances between
   * lattice positions, ordered from shortest to longest. Returned pairs are
   * canonical (i < j).
   *
   * @param m One-based shell index.
   * @param tolerance Relative tolerance used to group equal distances.
   * @return Canonical site pairs in the requested shell, or an empty vector if
   *         the finite lattice has fewer than m shells.
   * @throws std::invalid_argument If m is zero or tolerance is not positive.
   * @throws std::runtime_error If this graph has no lattice geometry.
   */
  std::vector<std::pair<std::uint64_t, std::uint64_t>> mth_nearest_neighbors(
      std::uint64_t m, double tolerance = 1.0e-9) const;

  /**
   * @brief Return the pairs in multiple geometric neighbor shells.
   *
   * Pair distances are classified once for all requested shell indices.
   * Returned pairs are canonical (i < j), and unavailable shells map to empty
   * vectors.
   *
   * @param shells One-based shell indices.
   * @param tolerance Relative tolerance used to group equal distances.
   * @return Requested shell indices mapped to their canonical site pairs.
   * @throws std::invalid_argument If any index is zero or tolerance is not
   * positive.
   * @throws std::runtime_error If this graph has no lattice geometry.
   */
  std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
  nearest_neighbor_shells(const std::vector<std::uint64_t>& shells,
                          double tolerance = 1.0e-9) const;

  /**
   * @brief Return physical connections classified by shell and bond axis.
   *
   * Unlike nearest_neighbor_shells(), this method preserves distinct periodic
   * images that collapse onto the same finite-lattice site pair.
   *
   * @param shells One-based shell indices.
   * @param tolerance Relative tolerance used to group distances and axes.
   * @return Connections in deterministic shell, orientation, and site order.
   * @throws std::invalid_argument If any index is zero or tolerance is not
   * positive.
   * @throws std::runtime_error If this graph has no lattice geometry.
   */
  std::vector<NeighborConnection> neighbor_connections(
      const std::vector<std::uint64_t>& shells,
      double tolerance = 1.0e-9) const;

  /**
   * @brief Return a copy with semantic labels for selected geometric classes.
   * @param definitions Shell-axis flavor definitions.
   * @param tolerance Relative tolerance used to detect duplicate axes.
   * @return A graph with unchanged geometry and the supplied flavor metadata.
   */
  LatticeGraph with_bond_flavors(
      const std::vector<BondFlavorDefinition>& definitions,
      double tolerance = 1.0e-9) const;

  /** @brief Return semantic bond-flavor definitions, if any. */
  const std::vector<BondFlavorDefinition>& bond_flavor_definitions() const;

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
   * @brief Create a honeycomb lattice with an explicit size convention.
   *
   * CompletePlaquettes adds the boundary cells required along open axes and,
   * for a fully open patch, removes the two dangling corner sites. UnitCells
   * uses two sites per unit cell.
   *
   * @param nx Number of unit cells or complete plaquettes along x.
   * @param ny Number of unit cells or complete plaquettes along y.
   * @param size_convention Interpretation of nx and ny.
   * @param periodic_x If true, apply periodic boundary conditions along x.
   * @param periodic_y If true, apply periodic boundary conditions along y.
   * @param t Uniform hopping weight.
   * @param dfs_ordering Reserved for API compatibility; currently ignored.
   */
  static LatticeGraph honeycomb(std::uint64_t nx, std::uint64_t ny,
                                HoneycombSizeConvention size_convention,
                                bool periodic_x = false,
                                bool periodic_y = false, double t = 1.0,
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
   */
  static LatticeGraph permute(const LatticeGraph& graph,
                              const std::vector<std::uint64_t>& path);

 private:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /**
   * @brief Private constructor from a sparse adjacency matrix.
   *
   * Used internally by factory methods, deserialization, and
   * make_bidirectional().
   *
   * @param adjacency Sparse square adjacency matrix (moved in).
   * @param coloring  Optional edge coloring (moved in).
   * @param positions Optional Cartesian site positions (moved in).
   * @param periods Optional periodic supercell vectors (moved in).
   * @param bond_flavors Optional semantic shell-axis labels (moved in).
   */
  explicit LatticeGraph(Eigen::SparseMatrix<double> adjacency,
                        std::optional<EdgeColoring> coloring = std::nullopt,
                        std::optional<Eigen::MatrixXd> positions = std::nullopt,
                        std::optional<Eigen::MatrixXd> periods = std::nullopt,
                        std::vector<BondFlavorDefinition> bond_flavors = {});

  static void _validate_geometry(
      std::uint64_t num_sites, const std::optional<Eigen::MatrixXd>& positions,
      const std::optional<Eigen::MatrixXd>& periods);
  static void _validate_bond_flavors(
      const std::optional<Eigen::MatrixXd>& positions,
      std::vector<BondFlavorDefinition>& definitions, double tolerance);

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
  /// Cartesian site positions used to identify geometric neighbor shells.
  std::optional<Eigen::MatrixXd> _positions;
  /// Periodic supercell vectors, with one vector per row.
  std::optional<Eigen::MatrixXd> _periods;
  /// Optional semantic labels for selected shell-axis classes.
  std::vector<BondFlavorDefinition> _bond_flavor_definitions;
};

static_assert(DataClassCompliant<LatticeGraph>,
              "LatticeGraph must derive from DataClass and implement all "
              "required deserialization methods");

}  //  namespace qdk::chemistry::data
