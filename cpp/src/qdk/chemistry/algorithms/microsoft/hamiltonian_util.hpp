// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace qdk::chemistry::scf {
class BasisSet;
}

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

/**
 * @brief Validate orbital indices (active or inactive) for bounds, uniqueness,
 * sortedness
 * @param indices The indices to validate
 * @param label Label for error messages (e.g., "Alpha active", "Beta inactive")
 * @param num_molecular_orbitals Total number of molecular orbitals
 * @return true if the indices are contiguous, false otherwise
 */
bool validate_contiguous_indices(const std::vector<size_t>& indices,
                                 const std::string& label,
                                 size_t num_molecular_orbitals);

/**
 * @brief Validate active orbital indices (legacy name, delegates to
 * validate_contiguous_indices)
 */
inline bool validate_active_contiguous_indices(
    const std::vector<size_t>& indices, const std::string& spin_label,
    size_t num_molecular_orbitals) {
  return validate_contiguous_indices(indices, spin_label + " active",
                                     num_molecular_orbitals);
}

/**
 * @brief Transforms AO 3-center vectors to MO basis.
 *
 * L_{ij}^k = \sum_{pq} C_{pi} C_{qj} L_{pq}^k
 *
 * @param ao_three_center_vectors The AO 3-center vectors (rows = N_ao*N_ao,
 * cols = N_vectors).
 * @param mo_coeffs The MO coefficient matrix (rows = N_ao, cols = N_mo).
 * @return The MO 3-center vectors (rows = N_mo*N_mo, cols = N_vectors).
 */
Eigen::MatrixXd transform_three_center_ao_to_mo(
    const Eigen::MatrixXd& ao_three_center_vectors,
    const Eigen::MatrixXd& mo_coeffs);

/**
 * @brief Builds the Coulomb (J) matrix from AO 3-center vectors and a density
 * matrix.
 * @param ao_three_center_vectors The AO 3-center vectors (rows = N_ao*N_ao,
 * cols = N_vectors).
 * @param density The density matrix (rows = N_ao, cols = N_ao).
 * @return The Coulomb (J) matrix.
 */
Eigen::MatrixXd build_J_from_three_center(
    const Eigen::MatrixXd& ao_three_center_vectors,
    const Eigen::MatrixXd& density);

/**
 * @brief Builds the Exchange (K) matrix from AO 3-center vectors and MO
 * coefficients.
 * @param ao_three_center_vectors The AO 3-center vectors (rows = N_ao*N_ao,
 * cols = N_vectors).
 * @param coeffs The MO coefficient matrix (rows = N_ao, cols = N_mo).
 * @param occ_orb_ind The indices of the occupied orbitals.
 * @return The Exchange (K) matrix.
 */
Eigen::MatrixXd build_K_from_three_center(
    const Eigen::MatrixXd& ao_three_center_vectors,
    const Eigen::MatrixXd& coeffs, const std::vector<size_t>& occ_orb_ind);

/**
 * @brief Build an active-space Hamiltonian from AO three-center vectors.
 *
 * Shared orchestration for Cholesky and density-fitted Hamiltonian
 * constructors. Transforms AO three-center vectors to MO basis, constructs
 * the inactive Fock matrix (restricted or unrestricted), computes the
 * inactive energy, and returns a ThreeCenterHamiltonianContainer wrapped
 * in a Hamiltonian.
 *
 * @param B_ao AO three-center vectors [nao^2 x nvec]
 * @param H_full Core Hamiltonian in AO basis [nao x nao]
 * @param Ca Alpha MO coefficient matrix [nao x nmo]
 * @param Cb Beta MO coefficient matrix [nao x nmo]
 * @param orbitals Orbitals object (contains active/inactive indices)
 * @param structure Molecular structure (for nuclear repulsion energy)
 * @param is_restricted_calc Whether to perform restricted calculation
 * @param store_ao_vectors Whether to store AO three-center vectors in the
 *        resulting container
 * @return Shared pointer to the constructed Hamiltonian
 */
std::shared_ptr<data::Hamiltonian>
build_active_space_hamiltonian_from_three_center(
    const Eigen::Ref<const Eigen::MatrixXd>& B_ao,
    const Eigen::MatrixXd& H_full, const Eigen::MatrixXd& Ca,
    const Eigen::MatrixXd& Cb, std::shared_ptr<data::Orbitals> orbitals,
    std::shared_ptr<data::Structure> structure, bool is_restricted_calc,
    bool store_ao_vectors);

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
