// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {
/**
 * @brief Transforms AO Cholesky vectors to MO basis.
 *
 * L_{ij}^k = \sum_{pq} C_{pi} C_{qj} L_{pq}^k
 *
 * @param ao_cholesky_vectors The AO Cholesky vectors (rows = N_ao*N_ao, cols =
 * N_vectors).
 * @param mo_coeffs The MO coefficient matrix (rows = N_ao, cols = N_mo).
 * @return The MO Cholesky vectors (rows = N_mo*N_mo, cols = N_vectors).
 */
Eigen::MatrixXd transform_cholesky_to_mo(
    const Eigen::MatrixXd& ao_cholesky_vectors,
    const Eigen::MatrixXd& mo_coeffs);

/**
 * @brief Builds the Coulomb (J) matrix from AO Cholesky vectors and a density
 * matrix.
 * @param ao_cholesky_vectors The AO Cholesky vectors (rows = N_ao*N_ao, cols =
 * N_vectors).
 * @param density The density matrix (rows = N_ao, cols = N_ao).
 * @return The Coulomb (J) matrix.
 */
Eigen::MatrixXd build_J_from_cholesky(
    const Eigen::MatrixXd& ao_cholesky_vectors, const Eigen::MatrixXd& density);

/**
 * @brief Builds the Exchange (K) matrix from AO Cholesky vectors and MO
 * coefficients.
 * @param ao_cholesky_vectors The AO Cholesky vectors (rows = N_ao*N_ao, cols =
 * N_vectors).
 * @param coeffs The MO coefficient matrix (rows = N_ao, cols = N_mo).
 * @param occ_orb_ind The indices of the occupied orbitals.
 * @return The Exchange (K) matrix.
 */
Eigen::MatrixXd build_K_from_cholesky(
    const Eigen::MatrixXd& ao_cholesky_vectors, const Eigen::MatrixXd& coeffs,
    const std::vector<size_t>& occ_orb_ind);

}  // namespace detail

class CholeskyHamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  CholeskyHamiltonianSettings() {
    set_default("scf_type", "auto");
    set_default("cholesky_tolerance", 1e-6);
    set_default("store_cholesky_vectors", true);
  }
  ~CholeskyHamiltonianSettings() override = default;
};

class CholeskyHamiltonianConstructor
    : public qdk::chemistry::algorithms::HamiltonianConstructor {
 public:
  CholeskyHamiltonianConstructor() {
    _settings = std::make_unique<CholeskyHamiltonianSettings>();
  };
  ~CholeskyHamiltonianConstructor() override = default;

  virtual std::string name() const final { return "qdk_cholesky"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
