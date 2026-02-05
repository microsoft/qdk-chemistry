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
 * @class DensityFittedHamiltonianContainer
 * @brief Contains a molecular Hamiltonian using three center
 * integrals.
 *
 * In addition to those contained in HamiltonianContainer, this subclass also
 * contains:
 * - Three-center two-electron integrals (electron-electron repulsion) in MO
 * representation.
 *
 */
class DensityFittedHamiltonianContainer : public HamiltonianContainer {
 public:
  /**
   * @brief Constructor for active space Hamiltonian with three center integrals
   * (Q|ij)
   *
   * @param one_body_integrals One-electron integrals in MO basis [norb x norb]
   * @param three_center_integrals Three-center two-electron integrals in MO
   * basis [naux x (norb x norb)]
   * @param orbitals Shared pointer to molecular orbital data for the system
   * @param core_energy Core energy (nuclear repulsion + inactive orbital
   * energy)
   * @param inactive_fock_matrix Inactive Fock matrix for the selected active
   * space
   * @param type Type of Hamiltonian (Hermitian by default)
   *
   * @throws std::invalid_argument if orbitals pointer is nullptr
   */
  DensityFittedHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals,
      const Eigen::MatrixXd& three_center_integrals,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Constructor for active space Hamiltonian with three center integrals
   * using separate spin components
   *
   * @param one_body_integrals_alpha One-electron integrals for alpha spin in MO
   * basis
   * @param one_body_integrals_beta One-electron integrals for beta spin in MO
   * basis
   * @param three_center_integrals_aa Three-center two-electron alpha-alpha
   * integrals
   * @param three_center_integrals_bb Three-center two-electron beta-beta
   * integrals
   * @param orbitals Shared pointer to molecular orbital data for the system
   * @param core_energy Core energy (nuclear repulsion + inactive orbital
   * energy)
   * @param inactive_fock_matrix_alpha Inactive Fock matrix for alpha spin in
   * the selected active space
   * @param inactive_fock_matrix_beta Inactive Fock matrix for beta spin in the
   * selected active space
   * @param type Type of Hamiltonian (Hermitian by default)
   *
   * @throws std::invalid_argument if orbitals pointer is nullptr
   */
  DensityFittedHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals_alpha,
      const Eigen::MatrixXd& one_body_integrals_beta,
      const Eigen::MatrixXd& three_center_integrals_aa,
      const Eigen::MatrixXd& three_center_integrals_bb,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix_alpha,
      const Eigen::MatrixXd& inactive_fock_matrix_beta,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Destructor
   */
  ~DensityFittedHamiltonianContainer() override = default;

  /**
   * @brief Create a deep copy of this container
   * @return Unique pointer to a cloned container
   */
  std::unique_ptr<HamiltonianContainer> clone() const override final;

  /**
   * @brief Get the type of the underlying container
   * @return String identifying the container type (e.g.,
   * "canonical_four_center", "density_fitted")
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
   * @return Pair of references to (aa, bb) three-center two-electron
   * integrals matrices
   * @throws std::runtime_error if integrals are not set
   */
  std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
  get_three_center_integrals() const;

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
   * @return Unique pointer to const Hamiltonian loaded from group
   * @throws std::runtime_error if I/O error occurs
   */
  static std::unique_ptr<DensityFittedHamiltonianContainer> from_hdf5(
      H5::Group& group);

  /**
   * @brief Load Hamiltonian from JSON
   * @param j JSON object containing Hamiltonian data
   * @return Unique pointer to const Hamiltonian loaded from JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<DensityFittedHamiltonianContainer> from_json(
      const nlohmann::json& j);

  /**
   * @brief Save Hamiltonian to an FCIDUMP file
   * @param filename Path to FCIDUMP file to create/overwrite
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @throws std::runtime_error if I/O error occurs
   */
  void to_fcidump_file(const std::string& filename, size_t nalpha,
                       size_t nbeta) const override final;

  /**
   * @brief Check if the Hamiltonian data is complete and consistent
   * @return True if all required data is set and dimensions are consistent
   */
  bool is_valid() const override final;

 private:
  /**
   * Three-center integrals in MO basis, stored as matrices [naux x n_geminals]
   * where n_geminals = norb * norb for each spin channel
   */
  const std::pair<std::shared_ptr<Eigen::MatrixXd>,
                  std::shared_ptr<Eigen::MatrixXd>>
      _three_center_integrals;

  /**
   * Lazily computed four-center integrals cache (built on first access).
   * Stores (aaaa, aabb, bbbb) as flattened arrays [norb^4].
   * Uses shared_ptr so restricted case can share the same data for all
   * channels.
   */
  mutable std::optional<std::tuple<std::shared_ptr<Eigen::VectorXd>,
                                   std::shared_ptr<Eigen::VectorXd>,
                                   std::shared_ptr<Eigen::VectorXd>>>
      _cached_four_center_integrals;

  /** Build four-center integrals from three-center integrals and cache them */
  void _build_four_center_cache() const;

  /** Validation helper for integral dimensions */
  void validate_integral_dimensions() const override final;

  size_t _get_orb_pair_index(size_t i, size_t j) const;

  double _get_two_body_element(const Eigen::MatrixXd& A, unsigned ij,
                               const Eigen::MatrixXd& B, unsigned kl) const;

  static std::pair<std::shared_ptr<Eigen::MatrixXd>,
                   std::shared_ptr<Eigen::MatrixXd>>
  make_restricted_three_center_integrals(const Eigen::MatrixXd& integrals);

  /**
   * @brief Save FCIDUMP file without filename validation (internal use)
   * @param filename Path to FCIDUMP file to create/overwrite
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_fcidump_file(const std::string& filename, size_t nalpha,
                        size_t nbeta) const;

  /** Serialization version */
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data
