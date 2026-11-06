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
 * @brief Resulting Givens representation of a real orthogonal matrix.
 *
 * The decomposition represents an orthogonal matrix as
 * @f$U = D L_{m-1} \cdots L_0@f$. Each @f$L_j@f$ contains non-overlapping
 * rotations @f$G(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\
 * \sin\theta&\cos\theta\end{bmatrix}@f$ on adjacent basis states, and
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
 * @brief Circuit data of a dense real isometry.
 *
 * The isometry @f$M@f$ has @f$2^k\chi@f$ rows and at most @f$\chi@f$
 * orthonormal columns, with @f$k\in\{0,1,2\}@f$. Row @f$q\chi+b@f$ holds the
 * output state with @f$k@f$ added qubits in state @f$q@f$ and input register
 * state @f$b@f$; column @f$a@f$ is input state @f$a@f$ with the added qubits
 * in zero. The circuit realizes a unitary @f$U@f$ with
 * @f$U|0\rangle_q|a\rangle=\sum_{q,b}M_{q\chi+b,\,a}|q\rangle|b\rangle@f$.
 *
 * For @f$k=1@f$ the circuit is @f$U=B\,\mathrm{UCR}_0\,V@f$, the two-block
 * cosine-sine decomposition of Eq. 6 of Rupprecht and Wölk
 * (arXiv:2605.28489). For @f$k=2@f$ it is
 * @f$U=B\,\mathrm{UCR}_2\,W_1\,\mathrm{CNOT}\,\mathrm{UCR}_1\,W_0\,
 * \mathrm{CNOT}\,\mathrm{UCR}_0\,V@f$ (Fig. 5 of the same reference), in
 * operator order. Each @f$\mathrm{UCR}_j@f$ is an Ry rotation of an added
 * qubit multiplexed by the input register, and @f$B@f$ is block diagonal over
 * the added qubits. For @f$k=0@f$ the isometry is completed to an orthogonal
 * matrix synthesized as @f$B@f$ alone.
 */
struct DenseIsometrySynthesis {
  /**
   * @brief Ry angles @f$2\arctan(D'_b / D_b)@f$, one per input state. One
   * rotation for one added qubit, three for two, none for zero.
   */
  std::vector<std::vector<double>> rotation_angles;

  /** @brief Givens data of @f$W_0@f$ and @f$W_1@f$; empty unless two qubits
   * are added. */
  std::vector<GivensDecomposition> mixing_givens;

  /** @brief Merged Givens data of the block-diagonal unitary @f$B@f$. */
  GivensDecomposition block_givens;

  /**
   * @brief Right factor @f$V@f$, the input-side factor of the first
   * cosine-sine decomposition, of size equal to the column count. It is the
   * identity when no qubit is added. It is reported even when omitted from
   * the circuit and equals @ref dense_isometry_right_factor.
   */
  Eigen::MatrixXd right_factor;

  /**
   * @brief Givens data of @f$V\oplus I@f$ on the input register, or empty
   * when the right factor is omitted or no qubit is added.
   */
  std::optional<GivensDecomposition> right_factor_givens;
};

/**
 * @brief Permutations and block synthesis data of a sparse real isometry.
 *
 * Represents a unitary completion of the isometry as @f$U=P_r\,B\,P_c@f$,
 * applied right to left: @f$P_c|c\rangle=|\pi_c(c)\rangle@f$ gathers the
 * columns of each connected block, the block-diagonal orthogonal matrix
 * @f$B@f$ described by @c block_givens acts on the gathered blocks, and
 * @f$P_r|v\rangle=|\pi_r(v)\rangle@f$ scatters the rows back. The first
 * columns of @f$U@f$ equal the isometry.
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
 * most @f$\chi@f$ columns, orthonormal within
 * @f$\lVert M^T M-I\rVert_F\leq 10^{-8}\,n_\mathrm{cols}@f$.
 * @param num_input_states Dimension @f$\chi@f$ of the input register.
 * @param omit_right_factor Whether to leave @f$V@f$ out of the circuit. The
 * circuit then realizes @f$M V^T@f$ on the first columns, and the caller
 * applies @f$V@f$, from @ref dense_isometry_right_factor, beforehand.
 * @throws std::invalid_argument If the shape is unsupported or the matrix is
 * not a finite isometry.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
DenseIsometrySynthesis dense_isometry_synthesis(const Eigen::MatrixXd& isometry,
                                                Eigen::Index num_input_states,
                                                bool omit_right_factor = false);

/**
 * @brief Right factor @f$V@f$ that @ref dense_isometry_synthesis omits.
 *
 * Runs the same factorization, so the result is identical to
 * DenseIsometrySynthesis::right_factor for the same arguments.
 *
 * @param isometry See @ref dense_isometry_synthesis.
 * @param num_input_states See @ref dense_isometry_synthesis.
 * @return Orthogonal matrix of size equal to the column count, the identity
 * when no qubit is added.
 * @throws std::invalid_argument If the shape is unsupported or the matrix is
 * not a finite isometry.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
Eigen::MatrixXd dense_isometry_right_factor(const Eigen::MatrixXd& isometry,
                                            Eigen::Index num_input_states);

/**
 * @brief Decompose a sparse real isometry into row and column permutations
 * around a block-diagonal orthogonal matrix.
 *
 * Columns that share a nonzero row are joined into one block. Only stored
 * nonzero entries are read.
 *
 * @param isometry Matrix with at least as many rows as columns, orthonormal
 * within @f$\lVert M^T M-I\rVert_F\leq 10^{-8}\,n_\mathrm{cols}@f$.
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
