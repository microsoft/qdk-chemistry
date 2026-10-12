// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <string_view>
#include <variant>
#include <vector>

namespace qdk::chemistry::utils::detail {

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
 * @brief Permutations and block synthesis data for one sparse MPS site.
 *
 * Represents the site unitary as @f$U=P_r\,B\,P_c@f$, applied right to left:
 * @f$P_c|c\rangle=|\pi_c(c)\rangle@f$ gathers the columns of each symmetry
 * block, the block-diagonal orthogonal matrix @f$B@f$ described by
 * @c block_givens acts on the gathered blocks, and
 * @f$P_r|v\rangle=|\pi_r(v)\rangle@f$ scatters the rows back.
 */
struct SparseSiteSynthesis {
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
 * @brief Circuit data for one dense site unitary of the sequential MPS
 * preparation.
 *
 * A site with four physical states is applied as
 * @f$\mathrm{UCR}_0\,\mathrm{CNOT}\,W_0\,\mathrm{UCR}_1\,\mathrm{CNOT}\,W_1\,
 * \mathrm{UCR}_2\,U@f$ (Fig. 5 of Rupprecht and Wölk, arXiv:2605.28489). A
 * site with two physical states is applied as @f$\mathrm{UCR}_0\,U@f$, the
 * two-block cosine-sine decomposition of Eq. 6 in the same reference. Each
 * @f$\mathrm{UCR}_k@f$ is a uniformly controlled rotation. The right
 * factor @f$V@f$ of the decomposition is absorbed into the preceding site.
 */
struct DenseSiteSynthesis {
  /**
   * @brief Ry angles @f$2\arctan(D'_b / D_b)@f$, one per bond state.
   * Three rotations for four physical states, one for two.
   */
  std::vector<std::vector<double>> rotation_angles;

  /** @brief Givens data of @f$W_0@f$ and @f$W_1@f$; empty for two states. */
  std::vector<GivensDecomposition> mixing_givens;

  /** @brief Merged Givens data of the block-diagonal unitary. */
  GivensDecomposition block_givens;

  /** @brief Right factor @f$V@f$ acting on the incoming bond. */
  Eigen::MatrixXd right_factor;
};

/**
 * @brief Synthesize one validated MPS site with a dense decomposition.
 *
 * For site @f$i@f$ with tensor @f$M^p_{ab}@f$, the returned unitary
 * @f$U_i@f$ satisfies
 * @f$U_i(|0\rangle_p\otimes V_i|a\rangle)=\sum_{p,b}(V_{i+1}(M^{p})^{T})_{b a}
 * |p\rangle|b\rangle@f$. The bond register holds @f$\chi@f$ states.
 *
 * @param site Validated real MPS site with two or four physical states.
 * @param ancilla_dim Dimension of the bond register.
 * @param following_right_factor Successor's right factor. Its dimension must
 * equal this site's right bond.
 * @throws std::invalid_argument If the site is complex, has an unsupported
 * physical dimension, is nonisometric, or is incompatible with the bond
 * register or successor factor.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
DenseSiteSynthesis dense_unitary_synthesis(
    const data::MPSSite& site, Eigen::Index ancilla_dim,
    const Eigen::MatrixXd& following_right_factor = Eigen::MatrixXd{});

/**
 * @brief Decompose one blocked MPS tensor into row and column permutations
 * around block-diagonal orthogonal matrices.
 *
 * @param site Validated real MPS site; its stored sector offsets are reused.
 * @param ancilla_dim Dimension of the bond register.
 * @throws std::invalid_argument If the site is complex, does not fit the bond
 * register, or is not an isometry within
 * @f$\lVert M^T M-I\rVert_F\leq 10^{-8}\chi_L@f$.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
SparseSiteSynthesis block_sparse_unitary_synthesis(const data::MPSSite& site,
                                                   Eigen::Index ancilla_dim);

using MPSSynthesis = std::variant<std::vector<DenseSiteSynthesis>,
                                  std::vector<SparseSiteSynthesis>>;

/**
 * @brief Synthesize the sites after site zero of a validated MPS container.
 *
 * Site zero is prepared separately as the initial state. Dense synthesis
 * absorbs each successor's right factor into its predecessor, processing sites
 * from right to left. Block-sparse sites are independent.
 *
 * @param mps Container whose construction validates bond compatibility.
 * @param ancilla_dim Dimension of the bond register.
 * @param unitary_synthesis "dense" or "block_sparse".
 * @throws std::invalid_argument If the method is unknown, the MPS is complex,
 * a bond does not fit the register, or a synthesized site is not an isometry.
 * @throws std::runtime_error If a numerical factorization breaks down.
 */
MPSSynthesis matrix_product_state_synthesis(
    const data::MPSContainer& mps, Eigen::Index ancilla_dim,
    std::string_view unitary_synthesis = "dense");
}  // namespace qdk::chemistry::utils::detail
