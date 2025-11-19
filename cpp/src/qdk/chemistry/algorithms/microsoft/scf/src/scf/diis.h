// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/types.h>

#include <deque>

namespace qdk::chemistry::scf {
/**
 * @brief Direct Inversion in the Iterative Subspace (DIIS) convergence
 * accelerator
 *
 * Implements the DIIS algorithm for accelerating SCF convergence by
 * extrapolating from previous Fock matrices and error vectors. DIIS builds
 * a linear combination of previous trial vectors that minimizes the norm
 * of the error vector.
 *
 * The algorithm maintains a subspace of recent Fock matrices and their
 * corresponding error vectors, then solves a linear system to find optimal
 * extrapolation coefficients.
 *
 * Reference: P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
 */
class DIIS {
 public:
  /**
   * @brief Construct a DIIS accelerator
   *
   * @param subspace_size Maximum number of vectors to retain in the DIIS
   */
  explicit DIIS(size_t subspace_size = 8);

  /**
   * @brief Perform DIIS extrapolation to generate improved Fock matrix
   *
   * Takes the current Fock matrix and error vector, adds them to the history,
   * and computes an extrapolated Fock matrix that should have reduced error.
   * The extrapolation is computed by solving:
   *
   *   min ||sum_i c_i * e_i||^2  subject to sum_i c_i = 1
   *
   * where e_i are the error vectors and c_i are the coefficients applied
   * to the Fock matrices F_i to produce F_diis = sum_i c_i * F_i.
   *
   * @param x Current Fock matrix to add to history
   * @param error Current error vector
   * @param x_diis Output extrapolated Fock matrix with reduced error
   */
  void extrapolate(const RowMajorMatrix& x, const RowMajorMatrix& error,
                   RowMajorMatrix* x_diis);

 private:
  /**
   * @brief Remove oldest vector pair when subspace is full
   *
   * Deletes the oldest Fock matrix and error vector from the history
   * to maintain the subspace size constraint.
   */
  void delete_oldest_();

  size_t subspace_size_;  ///< Maximum number of vectors in DIIS subspace

  std::deque<RowMajorMatrix>
      hist_;  ///< History of Fock matrices {F_1, F_2, ..., F_n}
  std::deque<RowMajorMatrix>
      errors_;        ///< History of error vectors {e_1, e_2, ..., e_n}
  RowMajorMatrix B_;  ///< DIIS B matrix: B_ij = <e_i|e_j> used in extrapolation
};
}  // namespace qdk::chemistry::scf
