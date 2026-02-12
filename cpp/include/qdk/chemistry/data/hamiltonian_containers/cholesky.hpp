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

#include "canonical_four_center.hpp"

namespace qdk::chemistry::data {

/**
 * @class CholeskyHamiltonianContainer
 * @brief Contains a molecular Hamiltonian and AO Cholesky vectors
 *
 * This class stores molecular Hamiltonian data for quantum chemistry
 * calculations, specifically designed for active space methods. It contains:
 * - One-electron integrals (kinetic + nuclear attraction) in MO representation
 * - Two-electron integrals (electron-electron repulsion) in MO representation
 * - Cholesky vectors in AO representation for reconstructing integrals
 * - Molecular orbital information for the active space
 * - Inactive Fock matrix
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
// TODO: Change Cholesky container to only store MO cholesky vectors and
// optionally AO cholesky vecs. Make two body ints evaluated on request
class CholeskyHamiltonianContainer
    : public CanonicalFourCenterHamiltonianContainer {
 public:
  /**
   * @brief Constructor for restricted Cholesky Hamiltonian
   *
   * Creates a restricted Hamiltonian where alpha and beta spin components
   * share the same integrals.
   *
   * @param one_body_integrals One-electron integrals in MO basis [norb x norb]
   * @param two_body_integrals Two-electron integrals in MO basis [norb x norb x
   * norb x norb]
   * @param orbitals Shared pointer to molecular orbital data for the system
   * @param core_energy Core energy (nuclear repulsion + inactive orbital
   * energy)
   * @param inactive_fock_matrix Inactive Fock matrix for the selected active
   * space
   * @param L_ao AO Cholesky vectors for reconstructing integrals
   * @param type Type of Hamiltonian (Hermitian by default)
   *
   * @throws std::invalid_argument if orbitals pointer is nullptr
   */
  CholeskyHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals,
      const Eigen::VectorXd& two_body_integrals,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix, const Eigen::MatrixXd& L_ao,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Constructor for unrestricted Cholesky Hamiltonian
   *
   * Creates an unrestricted Hamiltonian with separate alpha and beta spin
   * components.
   *
   * @param one_body_integrals_alpha One-electron integrals for alpha spin in MO
   * basis
   * @param one_body_integrals_beta One-electron integrals for beta spin in MO
   * basis
   * @param two_body_integrals_aaaa Two-electron alpha-alpha-alpha-alpha
   * integrals
   * @param two_body_integrals_aabb Two-electron alpha-beta-alpha-beta integrals
   * @param two_body_integrals_bbbb Two-electron beta-beta-beta-beta integrals
   * @param orbitals Shared pointer to molecular orbital data for the system
   * @param core_energy Core energy (nuclear repulsion + inactive orbital
   * energy)
   * @param inactive_fock_matrix_alpha Inactive Fock matrix for alpha spin
   * @param inactive_fock_matrix_beta Inactive Fock matrix for beta spin
   * @param L_ao AO Cholesky vectors for reconstructing integrals
   * @param type Type of Hamiltonian (Hermitian by default)
   *
   * @throws std::invalid_argument if orbitals pointer is nullptr
   */
  CholeskyHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals_alpha,
      const Eigen::MatrixXd& one_body_integrals_beta,
      const Eigen::VectorXd& two_body_integrals_aaaa,
      const Eigen::VectorXd& two_body_integrals_aabb,
      const Eigen::VectorXd& two_body_integrals_bbbb,
      std::shared_ptr<Orbitals> orbitals, double core_energy,
      const Eigen::MatrixXd& inactive_fock_matrix_alpha,
      const Eigen::MatrixXd& inactive_fock_matrix_beta,
      const Eigen::MatrixXd& L_ao,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Destructor
   */
  ~CholeskyHamiltonianContainer() = default;

  /**
   * @brief Create a deep copy of this container
   * @return Unique pointer to a cloned container
   */
  std::unique_ptr<HamiltonianContainer> clone() const override final;

  /**
   * @brief Get the type of the underlying container
   * @return String "cholesky" identifying this container type
   */
  std::string get_container_type() const override final;

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
   * @return Unique pointer to CholeskyHamiltonianContainer loaded from group
   * @throws std::runtime_error if I/O error occurs or data is malformed
   */
  static std::unique_ptr<CholeskyHamiltonianContainer> from_hdf5(
      H5::Group& group);

  /**
   * @brief Load Hamiltonian from JSON
   * @param j JSON object containing Hamiltonian data
   * @return Unique pointer to CholeskyHamiltonianContainer loaded from JSON
   * @throws std::runtime_error if JSON is malformed or missing required fields
   */
  static std::unique_ptr<CholeskyHamiltonianContainer> from_json(
      const nlohmann::json& j);

 private:
  // AO Cholesky vectors for integral reconstruction
  const std::shared_ptr<Eigen::MatrixXd> _ao_cholesky_vectors;
};

}  // namespace qdk::chemistry::data
