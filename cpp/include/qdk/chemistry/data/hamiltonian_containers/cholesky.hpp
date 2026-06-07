// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class CholeskyHamiltonianContainer
 * @brief Contains a molecular Hamiltonian expressed using three-center
 * integrals.
 *
 * In addition to those contained in HamiltonianContainer, this subclass also
 * contains:
 * - Three-center two-electron integrals (electron-electron repulsion) in MO
 * representation.
 * - Optionally, AO Cholesky vectors for potential reuse in further
 * transformations.
 *
 */
class CholeskyHamiltonianContainer : public HamiltonianContainer {
 public:
  /**
   * @brief Constructor for restricted Cholesky Hamiltonian.
   */
  CholeskyHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals,
      const Eigen::MatrixXd& three_center_integrals,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix,
      std::optional<Eigen::MatrixXd> ao_cholesky_vectors = std::nullopt,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Constructor for unrestricted Cholesky Hamiltonian.
   */
  CholeskyHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals_alpha,
      const Eigen::MatrixXd& one_body_integrals_beta,
      const Eigen::MatrixXd& three_center_integrals_aa,
      const Eigen::MatrixXd& three_center_integrals_bb,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix_alpha,
      const Eigen::MatrixXd& inactive_fock_matrix_beta,
      std::optional<Eigen::MatrixXd> ao_cholesky_vectors = std::nullopt,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief SymmetryBlockedTensor constructor for Cholesky Hamiltonian.
   * @param one_body One-body integrals as rank-2 SymmetryBlockedTensor.
   * @param three_center Three-center integrals as a rank-3
   *   SymmetryBlockedTensor with slots @c [MO_row, MO_col, auxiliary]. Each
   *   block is a dense @c [norb_row*norb_col, naux] @c MatrixXd.
   * @param orbitals Shared pointer to molecular orbital data.
   * @param core_energy Core energy.
   * @param inactive_fock Inactive Fock matrix as rank-2 SymmetryBlockedTensor.
   * @param ao_cholesky_vectors Optional AO Cholesky vectors.
   * @param type Hamiltonian type.
   */
  CholeskyHamiltonianContainer(
      SymmetryBlockedTensor<2> one_body, SymmetryBlockedTensor<3> three_center,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      std::shared_ptr<const SymmetryBlockedTensor<2>> inactive_fock,
      std::optional<Eigen::MatrixXd> ao_cholesky_vectors = std::nullopt,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Destructor
   */
  ~CholeskyHamiltonianContainer() override = default;

  /**
   * @brief Create a deep copy of this container
   * @return Unique pointer to a cloned container
   */
  std::unique_ptr<HamiltonianContainer> clone() const override final;

  /**
   * @brief Get the type of the underlying container
   * @return String identifying the container type (e.g.,
   * "cholesky")
   */
  std::string get_container_type() const override final;

  /**
   * @brief Get four-center two-electron integrals in MO basis for all spin
   * channels
   * @return Tuple of references to (aaaa, aabb, bbbb) four-center two-electron
   * integrals vectors
   * @throws std::runtime_error if integrals are not set
   */
  std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
             const Eigen::VectorXd&>
  get_two_body_integrals() const override;

  /**
   * @brief Get three-center integrals in MO basis for all spin channels
   * @return Pair of dense @c [norb^2, naux] @c MatrixXd for (alpha, beta).
   */
  std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
  get_three_center_integrals() const;

  /**
   * @brief Three-center integrals as a rank-3 symmetry-blocked tensor.
   * Slots are @c [MO_row, MO_col, auxiliary]; each block is a dense
   * @c [norb_row*norb_col, naux] @c MatrixXd.
   */
  const SymmetryBlockedTensor<3>& three_center() const;

  /**
   * @brief Get the optional AO Cholesky vectors
   * @return Const reference to the optional AO Cholesky vectors matrix
   * [nao^2 x nchol]. Contains std::nullopt if AO Cholesky vectors were not
   * provided at construction.
   */
  const std::optional<Eigen::MatrixXd>& get_ao_cholesky_vectors() const;

  /**
   * @brief Get specific four-center two-electron integral element
   * @param i First orbital index
   * @param j Second orbital index
   * @param k Third orbital index
   * @param l Fourth orbital index
   * @param channel Spin channel to query (aaaa, aabb, or bbbb), defaults to
   * aaaa
   * @return Four-center two-electron integral (ij|kl)
   * @throws std::out_of_range if indices are invalid
   */
  double get_two_body_element(
      unsigned i, unsigned j, unsigned k, unsigned l,
      SpinChannel channel = SpinChannel::aaaa) const override;

  /**
   * @brief Check if two-body integrals are available
   * @return True if two-body integrals are set
   */
  bool has_two_body_integrals() const override;

  /**
   * @brief Check if the Hamiltonian is restricted
   * @return True if alpha and beta integrals are identical
   */
  bool is_restricted() const override final;

  /**
   * @brief Convert Hamiltonian to JSON
   * @return JSON object containing Hamiltonian data
   */
  nlohmann::json to_json() const override final;

  /**
   * @brief Serialize Hamiltonian data to HDF5 group
   * @param group HDF5 group to write data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override final;

  /**
   * @brief Deserialize Hamiltonian data from HDF5 group
   * @param group HDF5 group to read data from
   * @return Unique pointer to const CholeskyHamiltonianContainer loaded from
   * group
   * @throws std::runtime_error if I/O error occurs
   */
  static std::unique_ptr<CholeskyHamiltonianContainer> from_hdf5(
      H5::Group& group);

  /**
   * @brief Load Hamiltonian from JSON
   * @param j JSON object containing Hamiltonian data
   * @return Unique pointer to const CholeskyHamiltonianContainer loaded from
   * JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<CholeskyHamiltonianContainer> from_json(
      const nlohmann::json& j);

  /**
   * @brief Check if the Hamiltonian data is complete and consistent
   * @return True if all required data is set and dimensions are consistent
   */
  bool is_valid() const override final;

 private:
  /// Three-center integrals (rank-3, slots @c [MO_row, MO_col, auxiliary];
  /// blocks are dense @c [orb_pair, naux] @c MatrixXd).
  std::shared_ptr<const SymmetryBlockedTensor<3>> _three_center;

  /**
   * Lazily computed four-center integrals cache (built on first access).
   * Stores (aaaa, aabb, bbbb) as flattened arrays [norb^4].
   * Uses shared_ptr so restricted case can share the same data for all
   * channels.
   */
  mutable std::tuple<std::shared_ptr<Eigen::VectorXd>,
                     std::shared_ptr<Eigen::VectorXd>,
                     std::shared_ptr<Eigen::VectorXd>>
      _cached_four_center_integrals;

  /** Build four-center integrals from three-center integrals and cache them */
  void _build_four_center_cache() const;

  /** Validation helper for integral dimensions */
  void validate_integral_dimensions() const override final;

  /** Optional AO Cholesky vectors for potential reuse */
  const std::optional<Eigen::MatrixXd> _ao_cholesky_vectors;

  /** Serialization version */
  static constexpr const char* SERIALIZATION_VERSION = "0.2.0";
};

}  // namespace qdk::chemistry::data
