// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <functional>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <utility>
#include <vector>

namespace qdk::chemistry::utils::detail {

inline constexpr double orthogonality_tolerance = 1.0e-8;

// Frobenius norm of M^T M - I.
inline double isometry_residual(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return (matrix.transpose() * matrix -
          Eigen::MatrixXd::Identity(matrix.cols(), matrix.cols()))
      .norm();
}

// The tolerance scales with the column count. The comparison rejects NaN.
inline bool is_isometry(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return isometry_residual(matrix) <=
         orthogonality_tolerance * static_cast<double>(matrix.cols());
}

struct FullSvd {
  Eigen::MatrixXd u;
  Eigen::VectorXd singular_values;
  Eigen::MatrixXd v;
};

// [A; B] = diag(U_1, U_2) [D_1; D_2] V, with D_1^2 + D_2^2 = I.
struct TwoBlockCsd {
  Eigen::MatrixXd u_1;
  Eigen::MatrixXd u_2;
  Eigen::VectorXd d_1;
  Eigen::VectorXd d_2;
  Eigen::MatrixXd v;
};

// Cosine-sine decompositions of a validated dense isometry, shared by the
// synthesis and the right-factor helper so both see the same V.
struct DenseIsometryFactors {
  Eigen::Index num_added_qubits = 0;
  std::vector<TwoBlockCsd> csds;
};

FullSvd singular_value_decomposition(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> qr_decomposition(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix, Eigen::Index num_columns);

TwoBlockCsd cosine_sine_decomposition(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b);

std::vector<double> rotation_angles(const Eigen::VectorXd& d,
                                    const Eigen::VectorXd& d_prime,
                                    Eigen::Index ancilla_dim);

std::vector<GivensDecomposition> givens_decomposition(
    const std::vector<std::reference_wrapper<const Eigen::MatrixXd>>& matrices);

GivensDecomposition merge_block_givens(
    const std::vector<GivensDecomposition>& decompositions);

DenseIsometryFactors dense_isometry_factors(const Eigen::MatrixXd& isometry,
                                            Eigen::Index num_input_states);

}  // namespace qdk::chemistry::utils::detail
