// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <cstdint>
#include <optional>
#include <vector>

namespace qdk::chemistry::utils {

/**
 * @brief Givens representation @f$U = D L_{m-1} \cdots L_0@f$ of a real
 * orthogonal matrix, where each layer applies
 * @f$G(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\
 * \sin\theta&\cos\theta\end{bmatrix}@f$ to disjoint adjacent basis states and
 * @f$D@f$ is a diagonal sign matrix.
 */
struct GivensDecomposition {
  /** @brief Rotation angles for each parallel layer. */
  std::vector<std::vector<double>> layer_angles;

  /** @brief One when the corresponding layer acts on odd-starting pairs. */
  std::vector<std::uint8_t> layer_shifted;

  /** @brief One when the corresponding basis state receives a minus sign. */
  std::vector<std::uint8_t> phases;
};

/**
 * @brief Circuit data of a dense real isometry @f$M@f$ with @f$2^k\chi@f$ rows,
 * whose row @f$q\chi+b@f$ is added-qubit state @f$q@f$ and input state @f$b@f$,
 * realized as @f$B\,\mathrm{UCR}_0\,V@f$ for @f$k=1@f$,
 * @f$B\,\mathrm{UCR}_2\,W_1\,\mathrm{CNOT}\,\mathrm{UCR}_1\,W_0\,
 * \mathrm{CNOT}\,\mathrm{UCR}_0\,V@f$ for @f$k=2@f$, and @f$B@f$ for @f$k=0@f$
 * (Rupprecht and Wölk, arXiv:2605.28489).
 */
struct DenseIsometrySynthesis {
  /** @brief Ry angles of each @f$\mathrm{UCR}_j@f$ in @f$(-2\pi,2\pi]@f$, one
   * per input state, with the Givens signs folded in. */
  std::vector<std::vector<double>> rotation_angles;

  /** @brief Givens data of @f$W_0@f$ and @f$W_1@f$; empty unless two qubits
   * are added. */
  std::vector<GivensDecomposition> mixing_givens;

  /** @brief Merged Givens data of the block-diagonal unitary @f$B@f$. */
  GivensDecomposition block_givens;

  /** @brief Right factor @f$V@f$ on the isometry columns, reported even when
   * omitted and the identity when no qubit is added. */
  Eigen::MatrixXd right_factor;

  /** @brief Givens data of @f$V\oplus I@f$ up to row signs folded into
   * @f$\mathrm{UCR}_0@f$, or empty when @f$V@f$ is omitted or no qubit is
   * added. */
  std::optional<GivensDecomposition> right_factor_givens;
};

/**
 * @brief Unitary completion @f$U=P_r\,B\,P_c@f$ of a sparse real isometry,
 * applied right to left, whose first columns equal the isometry.
 */
struct SparseIsometrySynthesis {
  /** @brief @f$\pi_c@f$: entry @f$c@f$ is the column of @f$B@f$ holding
   * target column @f$c@f$. */
  std::vector<Eigen::Index> column_permutation;

  /** @brief @f$\pi_r@f$: entry @f$v@f$ is the target row held by row @f$v@f$
   * of @f$B@f$. */
  std::vector<Eigen::Index> row_permutation;

  /** @brief Merged Givens data of @f$B@f$, largest diagonal block first. */
  GivensDecomposition block_givens;
};

/**
 * @brief Synthesize a dense real isometry with cosine-sine decompositions.
 *
 * @param isometry Matrix with @f$2^k\chi@f$ rows, @f$k\in\{0,1,2\}@f$, and at
 * most @f$\chi@f$ orthonormal columns, realized as its polar factor.
 * @param num_input_states Dimension @f$\chi@f$ of the input register.
 * @param omit_right_factor Whether to leave @f$V@f$ out, so the circuit
 * realizes @f$M V^T@f$ and the caller applies @f$V@f$ first.
 * @throws std::invalid_argument If the shape is unsupported or the matrix is
 * not a finite isometry.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
DenseIsometrySynthesis dense_isometry_synthesis(const Eigen::MatrixXd& isometry,
                                                Eigen::Index num_input_states,
                                                bool omit_right_factor = false);

/**
 * @brief Right factor @f$V@f$ that @ref dense_isometry_synthesis omits, equal
 * to DenseIsometrySynthesis::right_factor.
 *
 * @param isometry See @ref dense_isometry_synthesis.
 * @param num_input_states See @ref dense_isometry_synthesis.
 * @return Orthogonal matrix on the isometry columns, the identity when no
 * qubit is added.
 * @throws std::invalid_argument If the shape is unsupported or the matrix is
 * not a finite isometry.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
Eigen::MatrixXd dense_isometry_right_factor(const Eigen::MatrixXd& isometry,
                                            Eigen::Index num_input_states);

/**
 * @brief Decompose a sparse real isometry into row and column permutations
 * around a block-diagonal orthogonal matrix, joining columns that share a
 * nonzero row.
 *
 * @param isometry Matrix with orthonormal columns and at least as many rows as
 * columns, whose duplicate entries are summed.
 * @throws std::invalid_argument If the matrix is empty, wider than tall, or
 * not a finite isometry.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
SparseIsometrySynthesis block_sparse_isometry_synthesis(
    const Eigen::SparseMatrix<double>& isometry);

/** @copydoc block_sparse_isometry_synthesis(const Eigen::SparseMatrix<double>&)
 */
SparseIsometrySynthesis block_sparse_isometry_synthesis(
    const Eigen::MatrixXd& isometry);

}  // namespace qdk::chemistry::utils
