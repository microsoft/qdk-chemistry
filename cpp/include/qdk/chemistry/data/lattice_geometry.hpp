// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <map>
#include <optional>
#include <qdk/chemistry/data/data_class.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/** @brief Opaque semantic label assigned to a geometric bond class. */
using BondFlavorId = std::uint32_t;

/**
 * @brief A radial shell and unoriented geometric bond-axis class.
 *
 * The axis is a unit vector with one component per spatial dimension.
 */
struct BondClass {
  std::uint64_t shell;
  std::uint32_t orientation;
  Eigen::RowVectorXd axis;
};

/**
 * @brief One physical lattice connection, including its periodic image.
 *
 * The displacement and image shift have one entry per spatial dimension.
 * Image coefficients follow periodic-vector order, padded with zeros.
 */
struct NeighborConnection {
  std::uint64_t site_i;
  std::uint64_t site_j;
  BondClass bond_class;
  Eigen::RowVectorXd displacement;
  std::vector<std::int64_t> image_shift;
  std::optional<BondFlavorId> flavor;
  double weight = 1.0;
};

/**
 * @brief Immutable Cartesian lattice geometry, independent of connectivity.
 *
 * Stores site positions in any positive spatial dimension and optional
 * periodic supercell vectors. Geometric queries retain distinct periodic
 * images; they do not assign interaction weights, semantic flavors, or edge
 * colors. Neighbor searches currently support two-dimensional geometries only.
 */
class LatticeGeometry : public DataClass {
 public:
  /**
   * @brief Construct geometry from Cartesian positions and supercell vectors.
   * @param positions Finite (num_sites, d) matrix with d > 0, possibly empty.
   * @param periods Finite, nonzero, independent (k, d) vectors with k <= d.
   * @throws std::invalid_argument If the geometry is invalid.
   */
  explicit LatticeGeometry(
      Eigen::MatrixXd positions,
      std::optional<Eigen::MatrixXd> periods = std::nullopt);

  /** @brief Cartesian positions in site-index order. */
  const Eigen::MatrixXd& positions() const;

  /** @brief Periodic supercell vectors in image-shift order, if present. */
  const std::optional<Eigen::MatrixXd>& periods() const;

  /** @brief Number of sites, including isolated or coincident sites. */
  std::uint64_t num_sites() const;

  /** @brief Number of Cartesian components per position. */
  std::uint64_t dimension() const;

  /**
   * @brief Return physical connections by positive distance shell and axis.
   *
   * Connections are ordered by shell, orientation, endpoints, and image shift.
   * Endpoints satisfy site_i <= site_j; self-image connections have a positive
   * first nonzero image component. Distinct images are not merged. Flavor is
   * absent and weight is one. Unavailable finite shells contribute no entries.
   *
   * @param shells One-based shell indices; duplicate requests are ignored.
   * @param tolerance Positive finite relative distance and axis tolerance.
   * @return Canonical physical connections in the requested shells.
   * @throws std::invalid_argument If a shell or tolerance is invalid.
   * @throws std::runtime_error If the geometry is not two-dimensional.
   * @throws std::overflow_error If an image or displacement is out of range.
   */
  std::vector<NeighborConnection> neighbor_connections(
      const std::vector<std::uint64_t>& shells,
      double tolerance = 1.0e-9) const;

  /**
   * @brief Project open-lattice connections to sorted, unique site pairs.
   * @param shells One-based shell indices; unavailable shells map to empties.
   * @param tolerance Positive finite relative distance and axis tolerance.
   * @return Requested shells mapped to canonical pairs with i < j.
   * @throws std::runtime_error If periodic vectors are present or the geometry
   * is not two-dimensional.
   */
  std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
  nearest_neighbor_shells(const std::vector<std::uint64_t>& shells,
                          double tolerance = 1.0e-9) const;

  /**
   * @brief Return the sorted, unique pairs in one open-lattice shell.
   * @param m One-based shell index.
   * @param tolerance Positive finite relative distance and axis tolerance.
   * @return Canonical site pairs, or empty if the shell is unavailable.
   * @throws std::runtime_error If periodic vectors are present or the geometry
   * is not two-dimensional.
   */
  std::vector<std::pair<std::uint64_t, std::uint64_t>> mth_nearest_neighbors(
      std::uint64_t m, double tolerance = 1.0e-9) const;

  /**
   * @brief Unit-spaced chain with positions (i, 0), for 0 <= i < n.
   * @param n Positive number of sites.
   * @param periodic Whether to include the supercell vector (n, 0).
   */
  static LatticeGeometry chain(std::uint64_t n, bool periodic = false);

  /**
   * @brief Square lattice with site index y * nx + x and unit spacing.
   * @param nx Positive number of sites along x.
   * @param ny Positive number of sites along y.
   * @param periodic_x Periodic x direction; requires nx > 1.
   * @param periodic_y Periodic y direction; requires ny > 1.
   */
  static LatticeGeometry square(std::uint64_t nx, std::uint64_t ny,
                                bool periodic_x = false,
                                bool periodic_y = false);

  /**
   * @brief Triangular lattice with site index y * nx + x and unit bond length.
   * @param nx Positive number of sites along the first primitive vector.
   * @param ny Positive number of sites along the second primitive vector.
   * @param periodic_x Periodic first direction; requires nx > 1.
   * @param periodic_y Periodic second direction; requires ny > 1.
   */
  static LatticeGeometry triangular(std::uint64_t nx, std::uint64_t ny,
                                    bool periodic_x = false,
                                    bool periodic_y = false);

  /**
   * @brief Honeycomb geometry sized by unit cells, with unit bond length.
   *
   * The A and B sites of cell (x, y) have indices 2 * (y * nx + x) and
   * 2 * (y * nx + x) + 1, respectively.
   *
   * @param nx Positive number of unit cells along the first primitive vector.
   * @param ny Positive number of unit cells along the second primitive vector.
   * @param periodic_x Periodic first direction; requires nx > 1.
   * @param periodic_y Periodic second direction; requires ny > 1.
   */
  static LatticeGeometry honeycomb(std::uint64_t nx, std::uint64_t ny,
                                   bool periodic_x = false,
                                   bool periodic_y = false);

  /**
   * @brief Honeycomb patch sized by complete hexagonal plaquettes.
   *
   * Open directions include an extra boundary cell. Only fully open patches
   * omit the first A and last B corner sites, retaining the remaining order.
   * A fully open 1 x 1 patch therefore contains six sites.
   *
   * @param nx Positive number of plaquettes along the first primitive vector.
   * @param ny Positive number of plaquettes along the second primitive vector.
   * @param periodic_x Periodic first direction; requires nx > 1.
   * @param periodic_y Periodic second direction; requires ny > 1.
   */
  static LatticeGeometry honeycomb_plaquettes(std::uint64_t nx,
                                              std::uint64_t ny,
                                              bool periodic_x = false,
                                              bool periodic_y = false);

  /**
   * @brief Kagome geometry with three sites per unit cell and unit bonds.
   * @param nx Positive number of unit cells along the first primitive vector.
   * @param ny Positive number of unit cells along the second primitive vector.
   * @param periodic_x Periodic first direction; requires nx > 1.
   * @param periodic_y Periodic second direction; requires ny > 1.
   */
  static LatticeGeometry kagome(std::uint64_t nx, std::uint64_t ny,
                                bool periodic_x = false,
                                bool periodic_y = false);

  /**
   * @brief Relabel positions and retained integer coordinates together.
   * @param geometry Source geometry.
   * @param path Original site indices in the desired new order.
   * @return New geometry with unchanged periodic vectors.
   * @throws std::invalid_argument If path is not a permutation of all sites.
   */
  static LatticeGeometry permute(const LatticeGeometry& geometry,
                                 const std::vector<std::uint64_t>& path);

  /** @brief Wire-format identifier: lattice_geometry. */
  static std::string data_type_name() {
    return DATACLASS_TO_SNAKE_CASE(LatticeGeometry);
  }

  std::string get_data_type_name() const override { return data_type_name(); }
  std::string get_summary() const override;
  void to_file(const std::string& filename,
               const std::string& type) const override;
  nlohmann::json to_json() const override;
  void to_json_file(const std::string& filename) const override;
  void to_hdf5(H5::Group& group) const override;
  void to_hdf5_file(const std::string& filename) const override;

  static LatticeGeometry from_file(const std::string& filename,
                                   const std::string& type);
  static LatticeGeometry from_json(const nlohmann::json& j);
  static LatticeGeometry from_json_file(const std::string& filename);
  static LatticeGeometry from_hdf5(H5::Group& group);
  static LatticeGeometry from_hdf5_file(const std::string& filename);

 private:
  struct IntegerEmbedding {
    int nx;
    int ny;
    Eigen::Matrix2d primitive_vectors;
    Eigen::MatrixXd basis;
    std::vector<int> site_by_coordinate;
    bool periodic_x;
    bool periodic_y;
  };

  static LatticeGeometry _bravais(std::uint64_t nx, std::uint64_t ny,
                                  const Eigen::RowVector2d& a1,
                                  const Eigen::RowVector2d& a2,
                                  Eigen::MatrixXd basis, bool periodic_x,
                                  bool periodic_y,
                                  bool remove_open_corners = false);
  std::vector<NeighborConnection> _integer_neighbor_connections(
      const std::set<std::uint64_t>& shells, double tolerance) const;
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  Eigen::MatrixXd _positions;
  std::optional<Eigen::MatrixXd> _periods;
  std::optional<IntegerEmbedding> _integer_embedding;
};

static_assert(DataClassCompliant<LatticeGeometry>);

}  // namespace qdk::chemistry::data
