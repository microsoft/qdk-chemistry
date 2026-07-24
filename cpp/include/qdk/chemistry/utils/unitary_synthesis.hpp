// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <vector>

namespace qdk::chemistry::utils::detail {

/**
 * @brief Resulting Givens representation of a real orthogonal matrix.
 *
 * The decomposition represents an orthogonal matrix as
 * @f$U = D L_{k-1} \cdots L_0@f$. Each @f$L_j@f$ contains non-overlapping
 * rotations
 * @f$G(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\
 * \sin\theta&\cos\theta\end{bmatrix}@f$ on adjacent basis states, and
 * @f$D@f$ is a diagonal sign matrix.
 *
 * Layer @c j uses pairs @f$(0,1),(2,3),\ldots@f$ when
 * @c layer_shifted[j] is zero and @f$(1,2),(3,4),\ldots@f$ when it is one.
 * Angles are ordered by increasing pair index. A nonzero @c phases[i]
 * represents @f$D_{ii}=-1@f$.
 */
struct GivensDecomposition {
  /** @brief Rotation angles for each parallel layer and adjacent-pair slot. */
  std::vector<std::vector<double>> layer_angles;

  /** @brief One when the corresponding layer acts on odd-starting pairs. */
  std::vector<std::uint8_t> layer_shifted;

  /** @brief One when the corresponding basis state receives a minus sign. */
  std::vector<std::uint8_t> phases;
};

/**
 * @brief Cosine-sine factors for a vertically stacked two-block isometry.
 *
 * Represents @f$[A;B]=\operatorname{diag}(U_1,U_2)[D_1;D_2]V@f$, where
 * @f$D_1@f$ and @f$D_2@f$ are diagonal vectors satisfying
 * @f$D_1^2+D_2^2=I@f$.
 */
struct TwoBlockCsd {
  Eigen::MatrixXd u_1;
  Eigen::MatrixXd u_2;
  Eigen::VectorXd d_1;
  Eigen::VectorXd d_2;
  Eigen::MatrixXd v;
};

/** @brief Matrix factors produced by the three-step MPS site CSD peel. */
struct SiteCsd {
  std::array<Eigen::MatrixXd, 4> u;
  std::array<Eigen::VectorXd, 3> d_prime;
  Eigen::MatrixXd w_0;
  Eigen::MatrixXd w_1;
  Eigen::MatrixXd v;
};

/** @brief Permutations and block synthesis data for one sparse MPS site. */
struct SparseSiteSynthesis {
  std::vector<Eigen::Index> column_permutation;
  std::vector<Eigen::Index> row_permutation;
  GivensDecomposition block_givens;
};

/**
 * @brief Decompose two equally sized blocks whose vertical stack is an
 * isometry.
 *
 * @param a Upper block with shape @f$m\times k@f$.
 * @param b Lower block with the same shape as @p a.
 * @return Complete left orthogonal factors, CS diagonals, and the shared right
 * orthogonal factor.
 * @throws std::invalid_argument If dimensions, entries, or the isometry
 * precondition are invalid.
 */
TwoBlockCsd decompose_2d(const Eigen::Ref<const Eigen::MatrixXd>& a,
                         const Eigen::Ref<const Eigen::MatrixXd>& b);

/**
 * @brief Compute the three-step CSD peel of a packed MPS site isometry.
 *
 * @param matrix Physical-major packed isometry with shape
 * @f$(4\chi)\times\chi_L@f$.
 * @param ancilla_dim Padded bond dimension @f$\chi@f$.
 * @return Four terminal unitaries, three sine diagonals, two mixing unitaries,
 * and the right factor propagated to the preceding site.
 * @throws std::invalid_argument If the matrix shape or isometry precondition is
 * invalid.
 */
SiteCsd decompose_site_csd(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                           Eigen::Index ancilla_dim);

/**
 * @brief Decompose a real orthogonal matrix into parallel Givens layers.
 *
 * Uses the Clements double-sided elimination schedule. Alternating sweeps
 * eliminate entries with right column rotations and left row rotations. The
 * left rotations are subsequently commuted through the final diagonal sign
 * matrix so every returned layer can be applied from the right. Applying the
 * returned layers in increasing index order, followed by the phase signs,
 * reconstructs @p matrix as @f$D L_{k-1}\cdots L_0@f$.
 *
 * Numerically empty layers are omitted. The input dimension need not be a
 * power of two; sparse unitary synthesis decomposes individual symmetry blocks
 * of arbitrary positive dimension.
 *
 * @param matrix Nonempty, finite, square real matrix satisfying
 * @f$\lVert U^T U-I\rVert_F \leq 10^{-8}n@f$, where @f$n@f$ is its dimension.
 * @return Layer angles, shifted-layer flags, and diagonal phase signs.
 * @throws std::invalid_argument If @p matrix is empty, nonsquare, nonfinite,
 * or not orthogonal within the accepted tolerance.
 */
GivensDecomposition decompose_unitary_to_givens(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix);

/**
 * @brief Decompose orthogonal diagonal blocks and merge compatible rotations
 * into global adjacent-pair layers.
 *
 * @param blocks Nonempty list of nonempty real orthogonal square matrices.
 * @return A Givens decomposition of the corresponding block-diagonal matrix.
 * @throws std::invalid_argument If the block list or any block is invalid.
 */
GivensDecomposition decompose_block_diagonal_to_givens(
    const std::vector<Eigen::MatrixXd>& blocks);

/**
 * @brief Decompose a sparse-pattern site isometry into row and column
 * permutations around a block-diagonal orthogonal matrix.
 *
 * @param target Physical-major packed isometry. Its row dimension is the full
 * active register dimension and its columns are left-bond states.
 * @return Row and column permutations plus merged Givens data for the completed
 * diagonal blocks.
 * @throws std::invalid_argument If @p target is empty, nonfinite, wider than
 * tall, or not an isometry within tolerance.
 */
SparseSiteSynthesis decompose_sparse_site(
    const Eigen::Ref<const Eigen::MatrixXd>& target);

}  // namespace qdk::chemistry::utils::detail