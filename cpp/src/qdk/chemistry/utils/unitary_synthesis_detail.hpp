// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <exception>
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

FullSvd decompose_svd(const Eigen::Ref<const Eigen::MatrixXd>& matrix);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> decompose_qr(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix, Eigen::Index num_columns);

TwoBlockCsd decompose_csd(const Eigen::Ref<const Eigen::MatrixXd>& a,
                          const Eigen::Ref<const Eigen::MatrixXd>& b);

std::vector<double> rotation_angles(const Eigen::VectorXd& d,
                                    const Eigen::VectorXd& d_prime,
                                    Eigen::Index ancilla_dim);

std::vector<GivensDecomposition> decompose_unitaries_to_givens(
    const std::vector<std::reference_wrapper<const Eigen::MatrixXd>>& matrices);

GivensDecomposition merge_block_givens(
    const std::vector<GivensDecomposition>& decompositions);

void validate_site(const data::MPSSite& site, Eigen::Index ancilla_dim);

// Runs task(0), ..., task(count - 1) concurrently when OpenMP is available and
// rethrows the first exception once every task has finished.
template <typename Task>
void run_tasks(std::ptrdiff_t count, const Task& task) {
  std::exception_ptr error;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    try {
      task(index);
    } catch (...) {
#ifdef _OPENMP
#pragma omp critical(qdk_unitary_synthesis_error)
#endif
      {
        if (!error) {
          error = std::current_exception();
        }
      }
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

}  // namespace qdk::chemistry::utils::detail
