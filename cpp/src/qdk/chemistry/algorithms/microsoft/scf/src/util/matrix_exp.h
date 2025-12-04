// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace qdk::chemistry::scf {

/**
 * @brief Compute the matrix exponential using the scaling-and-squaring
 * algorithm
 *
 * This function computes exp(M) for a square matrix M using the
 * scaling-and-squaring method combined with Padé approximation.
 * Reference:
 * Higham, Nicholas J. "The scaling and squaring method for the matrix
 * exponential revisited." SIAM Journal on Matrix Analysis and Applications 26,
 * no. 4 (2005): 1179-1193. DOI. 10.1137/04061101X
 *
 * @param m Input matrix (row-major storage, size × size)
 * @param exp_m Output matrix exponential (row-major storage, size × size)
 * @param size Dimension of the square matrix (number of rows/columns)
 *
 * @details
 * Algorithm steps:
 * 1. Compute matrix 1-norm to determine scaling factor s
 * 2. Scale matrix: A' = A / 2^s where 2^s reduces matrix 1-norm below the
 * threshold for order 13 Padé approximation: theta13 5.371920351148152e0
 * 3. Apply 13th-order Padé approximation to compute exp(A')
 * 4. Square the result s times: exp(A) = (exp(A'))^(2^s)
 *
 * The scaling threshold θ₁₃ ≈ 5.37 ensures the Padé approximation converges
 * within machine precision for IEEE 754 double precision arithmetic.
 *
 * @warning Input and output matrices must not overlap in memory
 *
 * @throws std::runtime_error if matrix norm is too large (s > 30)
 *
 * @see pade_approximation for the core approximation routine
 */
void matrix_exp(const double *m, double *exp_m, int size);

/**
 * @brief Compute matrix exponential using 13th-order Padé approximation
 *
 * This function implements the (13,13) Padé rational approximation for the
 * matrix exponential. It is designed to work with pre-scaled matrices where
 * the matrix norm is sufficiently small to ensure convergence.
 *
 * @param x Input matrix (row-major storage, size × size)
 * @param exp_x Output matrix exponential (row-major storage, size × size)
 * @param size Dimension of the square matrix (number of rows/columns)
 *
 * @details
 * Implementation steps:
 * 1. Form numerator U and denominator V using Padé coefficients
 * 2. Solve linear system (V - U) * exp_x = (V + U)
 * 3. Handle row-major to column-major conversion for LAPACK
 *
 * The 13th-order approximation provides accuracy ~2.22e-16 for ||X|| ≤ 5.37.
 *
 * @warning Input matrix norm should be ≤ θ₁₃ ≈ 5.37 for optimal accuracy
 * @warning Input and output matrices must not overlap in memory
 * @warning Matrix must be non-singular (solvable linear system)
 *
 * @see matrix_exp for the complete scaling-and-squaring algorithm
 */
void pade_approximation(const double *x, double *exp_x, int size);

}  // namespace qdk::chemistry::scf
