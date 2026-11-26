// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <tuple>
#include <vector>

namespace qdk::chemistry::utils {

/**
 * @brief Synthesize a dense real isometry with cosine-sine decompositions.
 *
 * Row q * chi + b of the isometry M is added-qubit state q and input state b.
 * The circuit is B for no added qubit, B UCR_0 V for one, and
 * B UCR_2 W_1 CNOT UCR_1 W_0 CNOT UCR_0 V for two (Rupprecht and Wölk,
 * arXiv:2605.28489). Each orthogonal factor is returned as Givens layers
 * D L[m-1] ... L[0]: entry (j, k) of its angle matrix is the angle with which
 * L[j] applies G = [[cos, -sin], [sin, cos]] to the pair (k, k + 1), zero if
 * idle, and D is the diagonal of its sign vector.
 *
 * @param isometry Matrix with 1, 2 or 4 times chi rows and at most chi
 *   orthonormal columns, realized as its polar factor.
 * @param num_input_states Dimension chi of the input register.
 * @param omit_right_factor Whether to leave V out, so that the circuit
 *   realizes M V^T.
 * @return Ry angles of UCR_0, UCR_1 and UCR_2 by row; Givens angles and signs
 *   of B, W_0, W_1 and V, padded to chi, as present; and V.
 * @throws std::invalid_argument If the shape is unsupported or the matrix is
 *   not a finite isometry.
 */
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>,
           std::vector<Eigen::VectorXd>, Eigen::MatrixXd>
dense_isometry_synthesis(const Eigen::MatrixXd& isometry,
                         Eigen::Index num_input_states,
                         bool omit_right_factor = false);

/**
 * @brief Decompose a sparse real isometry into P_r B P_c, where B is block
 * diagonal over the groups of columns that share a nonzero row.
 *
 * @param isometry Matrix with orthonormal columns and at least as many rows
 *   as columns. Duplicate entries are summed and each block is realized as
 *   its polar factor.
 * @return Target row held by each row of B, column of B holding each target
 *   column, and the Givens angles and signs of B as in
 *   dense_isometry_synthesis.
 * @throws std::invalid_argument If the matrix is empty, wider than tall, or
 *   not a finite isometry.
 */
std::tuple<std::vector<Eigen::Index>, std::vector<Eigen::Index>,
           Eigen::MatrixXd, Eigen::VectorXd>
block_sparse_isometry_synthesis(const Eigen::SparseMatrix<double>& isometry);

}  // namespace qdk::chemistry::utils
