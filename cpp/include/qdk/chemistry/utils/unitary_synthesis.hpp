// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <vector>

namespace qdk::chemistry::utils::detail {

/**
 * @brief Resulting Givens representation of a real orthogonal matrix.
 *
 * The decomposition represents an orthogonal matrix as
 * @f$U = D L_{m-1} \cdots L_0@f$. Each @f$L_j@f$ contains non-overlapping
 * rotations
 * @f$G(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\
 * \sin\theta&\cos\theta\end{bmatrix}@f$ on adjacent basis states, and
 * @f$D@f$ is a diagonal sign matrix.
 *
 * An @f$n \times n@f$ matrix needs at most @f$m = n@f$ layers (one when
 * @f$n = 2@f$), which together hold @f$n(n-1)/2@f$ rotations in general.
 * Layers whose angles all vanish are omitted. The blocks of a block-diagonal
 * matrix act on disjoint basis states, so layer @c j of every block is applied
 * in one shared layer: four @f$16 \times 16@f$ blocks need 16 layers rather
 * than the 64 of a general @f$64 \times 64@f$ matrix. Blocks that start at
 * indices of different parity can add one layer.
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
 * @brief Permutations and block synthesis data for one sparse MPS site.
 *
 * Represents the completed site unitary @f$U@f$ as
 * @f$U_{r c}=B_{\pi_r^{-1}(r),\,\pi_c(c)}@f$, where @f$B@f$ is the
 * block-diagonal orthogonal matrix described by @c block_givens.
 */
struct SparseSiteSynthesis {
  /** @brief Entry @f$c@f$ is the column of @f$B@f$ holding target column
   * @f$c@f$. */
  std::vector<Eigen::Index> column_permutation;

  /** @brief Entry @f$v@f$ is the target row held by row @f$v@f$ of @f$B@f$. */
  std::vector<Eigen::Index> row_permutation;

  /** @brief Merged Givens data of @f$B@f$, largest diagonal block first. */
  GivensDecomposition block_givens;
};

/**
 * @brief Circuit data for one dense site unitary of the sequential MPS
 * preparation.
 *
 * A site with four physical states is applied as
 * @f$\mathrm{UCR}_0\,\mathrm{CNOT}\,W_0\,\mathrm{UCR}_1\,\mathrm{CNOT}\,W_1\,
 * \mathrm{UCR}_2\,U@f$ (Fig. 5 of Rupprecht and Wölk, arXiv:2605.28489). A
 * site with two physical states is applied as @f$\mathrm{UCR}_0\,U@f$. The
 * right factor @f$V@f$ of the decomposition is not part of the site circuit;
 * it is absorbed into the preceding site or the initial state.
 */
struct DenseSiteSynthesis {
  /**
   * @brief Ry angles of each uniformly controlled rotation, one per bond
   * state: three rotations for four physical states, one for two.
   */
  std::vector<std::vector<double>> rotation_angles;

  /** @brief Givens data of @f$W_0@f$ and @f$W_1@f$; empty for two states. */
  std::vector<GivensDecomposition> mixing_givens;

  /** @brief Merged Givens data of the block-diagonal terminal unitary. */
  GivensDecomposition block_givens;

  /** @brief Right factor @f$V@f$ acting on the incoming bond. */
  Eigen::MatrixXd right_factor;
};

/**
 * @brief Synthesize the dense site unitaries of the sequential MPS preparation
 * for consecutive MPS sites.
 *
 * For site @f$i@f$ with tensor @f$M^p_{ab}@f$, the returned unitary
 * @f$U_i@f$ satisfies
 * @f$U_i(|0\rangle_p\otimes V_i|a\rangle)=\sum_{p,b}(V_{i+1}(M^{p})^{T})_{b a}
 * |p\rangle|b\rangle@f$ for every left-bond state @f$a@f$, where @f$V_i@f$ is
 * its returned right factor and @f$V_{i+1}@f$ that of the following site, or
 * the identity for the last site. Each site therefore absorbs the right factor
 * of its successor, and only the right factor of the first site remains for
 * the caller to absorb into the preceding site or the initial state. The bond
 * register holds @f$\chi@f$ states, and right-bond states beyond a site's
 * right bond have zero amplitude.
 *
 * Four physical states use a three-step peel: two QR factorizations split the
 * packed isometry into three independent two-block CSDs. Two physical states
 * use a single two-block CSD. Each cosine-sine pair @f$(D, D')@f$ becomes Ry
 * angles @f$2\arctan(D'_b / D_b)@f$, zero-padded to @p ancilla_dim, and each
 * orthogonal factor is decomposed into Givens layers.
 *
 * Absorbing @f$V_{i+1}@f$ rotates only the right-bond rows of every physical
 * block, which leaves the QR factors' triangular parts, the cosine-sine
 * angles, the mixing unitaries, and @f$V_i@f$ unchanged and left-multiplies
 * each terminal diagonal block by @f$V_{i+1}@f$. All sites are therefore
 * factored concurrently and the rotation is applied to the terminal blocks
 * afterwards. The factors of all sites are held in memory at once.
 *
 * @param sites Real right-orthonormal sites with two or four physical states,
 * both bond dimensions at most @p ancilla_dim, and each right bond dimension
 * equal to the left bond dimension of the following site.
 * @param ancilla_dim Dimension @f$\chi@f$ of the bond register.
 * @return Rotation angles, Givens data, and the right factor of each site, in
 * input order.
 * @throws std::invalid_argument If a site is complex, does not fit the bond
 * register, has nonfinite entries, is not an isometry, or the bonds of
 * consecutive sites do not match.
 */
std::vector<DenseSiteSynthesis> decompose_dense_sites(
    const std::vector<data::MPSSite>& sites, Eigen::Index ancilla_dim);

/**
 * @brief Decompose sparse MPS sites into row and column permutations around
 * block-diagonal orthogonal matrices.
 *
 * The packed isometry of a site has row @f$p\chi+b@f$ and column @f$a@f$ equal
 * to @f$M^p_{ab}@f$, with zero rows for right-bond states beyond the site's
 * right bond. Its row index equals the little-endian value of a register
 * holding the bond in its low qubits and the physical state in its high
 * qubits.
 *
 * The decomposition reads the nonzero entries of the stored symmetry blocks
 * directly; absent blocks are structural zeros and the dense isometry is never
 * formed. Left-bond columns that share a nonzero row are grouped into
 * connected components, which need not be contiguous, so the components have
 * disjoint row supports. Within a component, rows are ordered by the first
 * column that reaches them. Each component, restricted to its rows, is
 * completed to an orthogonal block, and every unused row becomes a
 * @f$1\times 1@f$ identity block. The result depends only on the nonzero
 * pattern and values, not on how they are distributed over symmetry blocks.
 * Sites are independent, and the blocks of all sites are decomposed
 * concurrently.
 *
 * @param sites Real right-orthonormal sites with both bond dimensions at most
 * @p ancilla_dim.
 * @param ancilla_dim Dimension @f$\chi@f$ of the bond register.
 * @return Row and column permutations plus merged Givens data for the completed
 * diagonal blocks of each site, in input order.
 * @throws std::invalid_argument If a site is complex, does not fit the bond
 * register, has nonfinite entries, or is not an isometry within
 * @f$\lVert M^T M-I\rVert_F\leq 10^{-8}\chi_L@f$.
 */
std::vector<SparseSiteSynthesis> decompose_sparse_sites(
    const std::vector<data::MPSSite>& sites, Eigen::Index ancilla_dim);
}  // namespace qdk::chemistry::utils::detail
