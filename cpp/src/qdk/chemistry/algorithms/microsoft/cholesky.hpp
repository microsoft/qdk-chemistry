// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <qdk/chemistry/data/basis_set.hpp>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Performs pivoted Cholesky decomposition on a matrix.
 * @param matrix The input square matrix to decompose.
 * @param tolerance The error tolerance for the decomposition.
 * @return A matrix where each column is a Cholesky vector (rows = dim, cols =
 * rank).
 */
Eigen::MatrixXd pivoted_cholesky_decomposition(
    const Eigen::MatrixXd& matrix,
    double tolerance = std::numeric_limits<double>::epsilon());

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
}  // namespace qdk::chemistry::algorithms::microsoft
