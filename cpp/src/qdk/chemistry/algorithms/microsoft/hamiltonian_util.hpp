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
 * @brief Check whether a sorted index vector is contiguous.
 */
bool is_indices_contiguous(const std::vector<size_t>& indices);

/**
 * @brief Build inactive density matrix with contiguity optimization.
 */
Eigen::MatrixXd build_inactive_density(const Eigen::MatrixXd& C,
                                       const std::vector<size_t>& indices,
                                       size_t n_ao);

/**
 * @brief Results of inactive Fock construction for one spin channel.
 */
struct InactiveFockResult {
  Eigen::MatrixXd F_inactive;  // Full MO-basis inactive Fock
  Eigen::MatrixXd H_active;    // Active-space one-body Hamiltonian
  double E_inactive;           // Inactive energy contribution
};

/**
 * @brief Compute inactive Fock, active H, and inactive energy (restricted).
 *
 * Given J and K in AO basis, builds F_inactive = H + 2J - K, transforms to MO,
 * computes inactive energy, and extracts active-space H.
 */
InactiveFockResult compute_restricted_inactive(
    const Eigen::MatrixXd& J_ao, const Eigen::MatrixXd& K_ao,
    const Eigen::MatrixXd& H_full, const Eigen::MatrixXd& Ca,
    const std::vector<size_t>& inactive_indices,
    const std::vector<size_t>& active_indices);

/**
 * @brief Compute inactive Fock, active H, and inactive energy (unrestricted).
 *
 * Given alpha/beta J and K in AO basis, builds both spin Fock matrices,
 * computes inactive energy, and extracts active-space H for each spin.
 */
struct UnrestrictedInactiveFockResult {
  Eigen::MatrixXd F_inactive_alpha;
  Eigen::MatrixXd F_inactive_beta;
  Eigen::MatrixXd H_active_alpha;
  Eigen::MatrixXd H_active_beta;
  double E_inactive;
};

UnrestrictedInactiveFockResult compute_unrestricted_inactive(
    const Eigen::MatrixXd& J_alpha_ao, const Eigen::MatrixXd& K_alpha_ao,
    const Eigen::MatrixXd& J_beta_ao, const Eigen::MatrixXd& K_beta_ao,
    const Eigen::MatrixXd& H_full, const Eigen::MatrixXd& Ca,
    const Eigen::MatrixXd& Cb,
    const std::vector<size_t>& inactive_indices_alpha,
    const std::vector<size_t>& inactive_indices_beta,
    const std::vector<size_t>& active_indices_alpha,
    const std::vector<size_t>& active_indices_beta);

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
