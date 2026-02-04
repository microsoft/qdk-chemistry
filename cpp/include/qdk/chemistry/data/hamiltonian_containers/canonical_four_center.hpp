// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class CanonicalFourCenterHamiltonianContainer
 * @brief Contains a molecular Hamiltonian using canonical four center
 * integrals, implemented as a subclass of HamiltonianContainer.
 *
 * This class stores molecular Hamiltonian data for quantum chemistry
 * calculations, specifically designed for active space methods. It contains:
 * - One-electron integrals (kinetic + nuclear attraction) in MO representation
 * - Two-electron integrals (electron-electron repulsion) in MO representation
 * - Molecular orbital information for the active space
 * - Core energy contributions from inactive orbitals and nuclear repulsion
 *
 * This class implies that all inactive orbitals are fully occupied for the
 * purpose of computing the core energy and inactive Fock matrix.
 *
 * The Hamiltonian is immutable after construction, meaning all data must be
 * provided during construction and cannot be modified afterwards. The
 * Hamiltonian supports both restricted and unrestricted calculations and
 * integrates with the broader quantum chemistry framework for active space
 * methods.
 */
class CanonicalFourCenterHamiltonianContainer : public HamiltonianContainer {
 public:
  /**
   * @brief Constructor for active space Hamiltonian with four center integrals
   *
   * @param one_body_integrals One-electron integrals in MO basis [norb x norb]
   * @param two_body_integrals Two-electron integrals in MO basis [norb x norb x
   * norb x norb]
   * @param orbitals Shared pointer to molecular orbital data for the system
   * @param core_energy Core energy (nuclear repulsion + inactive orbital
   * energy)
   * @param inactive_fock_matrix Inactive Fock matrix for the selected active
   * space
   * @param type Type of Hamiltonian (Hermitian by default)
   *
   * @throws std::invalid_argument if orbitals pointer is nullptr
   */
  CanonicalFourCenterHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals,
      const Eigen::VectorXd& two_body_integrals,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Constructor for active space Hamiltonian with four center integrals
   * using separate spin components
   *
   * @param one_body_integrals_alpha One-electron integrals for alpha spin in MO
   * basis
   * @param one_body_integrals_beta One-electron integrals for beta spin in MO
   * basis
   * @param two_body_integrals_aaaa Two-electron alpha-alpha-alpha-alpha
   * integrals
   * @param two_body_integrals_aabb Two-electron alpha-alpha-beta-beta integrals
   * @param two_body_integrals_bbbb Two-electron beta-beta-beta-beta integrals
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
  CanonicalFourCenterHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals_alpha,
      const Eigen::MatrixXd& one_body_integrals_beta,
      const Eigen::VectorXd& two_body_integrals_aaaa,
      const Eigen::VectorXd& two_body_integrals_aabb,
      const Eigen::VectorXd& two_body_integrals_bbbb,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix_alpha,
      const Eigen::MatrixXd& inactive_fock_matrix_beta,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Destructor
   */
  ~CanonicalFourCenterHamiltonianContainer() = default;

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
   * @brief Get two-electron integrals in MO basis for all spin channels
   * @return Tuple of references to (aaaa, aabb, bbbb) two-electron integrals
   * vectors
   * @throws std::runtime_error if integrals are not set
   */
  std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
             const Eigen::VectorXd&>
  get_two_body_integrals() const override final;

  /**
   * @brief Get specific two-electron integral element
   * @param i First orbital index
   * @param j Second orbital index
   * @param k Third orbital index
   * @param l Fourth orbital index
   * @param channel Spin channel to query (aaaa, aabb, or bbbb), defaults to
   * aaaa
   * @return Two-electron integral (ij|kl)
   * @throws std::out_of_range if indices are invalid
   */
  double get_two_body_element(
      unsigned i, unsigned j, unsigned k, unsigned l,
      SpinChannel channel = SpinChannel::aaaa) const override final;

  /**
   * @brief Check if two-body integrals are available
   * @return True if two-body integrals are set
   */
  bool has_two_body_integrals() const override final;

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
   * @return Unique pointer to Hamiltonian loaded from group
   * @throws std::runtime_error if I/O error occurs
   */
  static std::unique_ptr<CanonicalFourCenterHamiltonianContainer> from_hdf5(
      H5::Group& group);

  /**
   * @brief Load Hamiltonian from JSON
   * @param j JSON object containing Hamiltonian data
   * @return Shared pointer to Hamiltonian loaded from JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<CanonicalFourCenterHamiltonianContainer> from_json(
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
  /// Two-electron integrals in MO basis, stored as flattened arrays [norb^4]
  /// Access pattern: V[i*norb^3 + j*norb^2 + k*norb + l] = (ij|kl)
  const std::tuple<std::shared_ptr<Eigen::VectorXd>,
                   std::shared_ptr<Eigen::VectorXd>,
                   std::shared_ptr<Eigen::VectorXd>>
      _two_body_integrals;

  /// Validation helpers
  void validate_integral_dimensions() const override final;

  size_t get_two_body_index(size_t i, size_t j, size_t k, size_t l) const;

  static std::tuple<std::shared_ptr<Eigen::VectorXd>,
                    std::shared_ptr<Eigen::VectorXd>,
                    std::shared_ptr<Eigen::VectorXd>>
  make_restricted_two_body_integrals(const Eigen::VectorXd& integrals);

  /**
   * @brief Save FCIDUMP file without filename validation (internal use)
   * @param filename Path to FCIDUMP file to create/overwrite
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_fcidump_file(const std::string& filename, size_t nalpha,
                        size_t nbeta) const;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data
