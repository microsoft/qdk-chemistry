// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <tuple>
#include <vector>

namespace qdk::chemistry::scf {
class BasisSet;
}

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

/**
 * @brief Validate active orbital indices
 * @param indices The indices to validate
 * @param spin_label Label for error messages (e.g., "Alpha", "Beta")
 * @param num_molecular_orbitals Total number of molecular orbitals
 * @return true if the indices are contiguous, false otherwise
 */
bool validate_active_contiguous_indices(const std::vector<size_t>& indices,
                                        const std::string& spin_label,
                                        size_t num_molecular_orbitals);

}  // namespace detail

namespace detail_three_center {
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

}  // namespace detail_three_center

}  // namespace qdk::chemistry::algorithms::microsoft
