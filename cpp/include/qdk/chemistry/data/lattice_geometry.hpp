// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <qdk/chemistry/data/data_class.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

class LatticeGraph;

/**
 * @brief Immutable two-dimensional geometry of a built-in lattice.
 *
 * Stores the Cartesian site positions and optional periodic supercell vectors
 * of a factory lattice. LatticeGraph::from_geometry turns its distance shells
 * into labelled edges; the geometry assigns no weights, flavors, or colors.
 */
class LatticeGeometry : public DataClass {
 public:
  /** @brief Cartesian positions (num_sites, 2) in site-index order. */
  const Eigen::MatrixXd& positions() const;

  /** @brief Periodic supercell vectors, if present. */
  const std::optional<Eigen::MatrixXd>& periods() const;

  /** @brief Number of sites. */
  std::uint64_t num_sites() const;

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
  friend class LatticeGraph;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /** @brief One bond of a distance shell, with its canonical unoriented axis.
   */
  struct ShellBond {
    std::uint64_t site_i;
    std::uint64_t site_j;
    std::uint64_t shell;
    Eigen::RowVector2d axis;
  };

  struct IntegerEmbedding {
    int nx;
    int ny;
    Eigen::Matrix2d primitive_vectors;
    Eigen::MatrixXd basis;
    std::vector<int> site_by_coordinate;
    bool periodic_x;
    bool periodic_y;
  };

  LatticeGeometry(Eigen::MatrixXd positions,
                  std::optional<Eigen::MatrixXd> periods,
                  IntegerEmbedding embedding);

  static LatticeGeometry _bravais(std::uint64_t nx, std::uint64_t ny,
                                  const Eigen::RowVector2d& a1,
                                  const Eigen::RowVector2d& a2,
                                  Eigen::MatrixXd basis, bool periodic_x,
                                  bool periodic_y,
                                  bool remove_open_corners = false);

  /**
   * @brief Return one record per physical bond in the requested shells.
   *
   * Shells rank the distinct distances present on this lattice, including
   * periodic images. Each image is a separate record with site_i <= site_j.
   *
   * @throws std::invalid_argument If a shell or the tolerance is invalid.
   * @throws std::overflow_error If the stencil exceeds the integer range.
   */
  std::vector<ShellBond> _shell_bonds(const std::vector<std::uint64_t>& shells,
                                      double tolerance) const;
  static LatticeGeometry _from_integer_embedding(
      int nx, int ny, const Eigen::MatrixXd& primitive_vectors,
      Eigen::MatrixXd basis, std::vector<int> site_by_coordinate,
      bool periodic_x, bool periodic_y);
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  Eigen::MatrixXd _positions;
  std::optional<Eigen::MatrixXd> _periods;
  IntegerEmbedding _embedding;
};

static_assert(DataClassCompliant<LatticeGeometry>);

}  // namespace qdk::chemistry::data
