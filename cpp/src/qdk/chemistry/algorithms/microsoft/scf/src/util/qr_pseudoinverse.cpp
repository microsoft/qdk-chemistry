// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/qr_pseudoinverse.h>

#include <cstdint>
#include <lapack.hh>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::scf {

RowMajorMatrix qr_pseudoinverse(const RowMajorMatrix& coefficients) {
  if (coefficients.cols() == 0 || coefficients.rows() < coefficients.cols() ||
      !coefficients.allFinite()) {
    throw std::invalid_argument(
        "qr_pseudoinverse requires a finite, nonempty tall or square matrix.");
  }

  const int64_t num_rows = coefficients.rows();
  const int64_t num_columns = coefficients.cols();
  Eigen::MatrixXd qr = coefficients;
  std::vector<int64_t> pivots(num_columns, 0);
  std::vector<double> tau(num_columns);
  const auto qr_info = lapack::geqp3(num_rows, num_columns, qr.data(), num_rows,
                                     pivots.data(), tau.data());
  if (qr_info != 0) {
    throw std::runtime_error("qr_pseudoinverse QR factorization failed (info=" +
                             std::to_string(qr_info) + ")");
  }

  const Eigen::MatrixXd R = qr.topRows(num_columns);
  const auto diagonal = R.diagonal().cwiseAbs().eval();
  const double singular_tolerance =
      std::numeric_limits<double>::epsilon() * num_rows * diagonal.maxCoeff();
  if (!R.allFinite() || (diagonal.array() <= singular_tolerance).any()) {
    throw std::runtime_error(
        "qr_pseudoinverse requires numerically full column rank.");
  }

  const auto q_info = lapack::orgqr(num_rows, num_columns, num_columns,
                                    qr.data(), num_rows, tau.data());
  if (q_info != 0) {
    throw std::runtime_error("qr_pseudoinverse Q generation failed (info=" +
                             std::to_string(q_info) + ")");
  }

  Eigen::MatrixXd inverse = qr.transpose();
  const auto solve_info =
      lapack::trtrs(lapack::Uplo::Upper, lapack::Op::NoTrans,
                    lapack::Diag::NonUnit, num_columns, num_rows, R.data(),
                    num_columns, inverse.data(), num_columns);
  if (solve_info != 0) {
    throw std::runtime_error("qr_pseudoinverse triangular solve failed (info=" +
                             std::to_string(solve_info) + ")");
  }

  // C * P = Q * R, so C_plus = P * R^{-1} * Q^T; LAPACK pivots are 1-based.
  RowMajorMatrix result(num_columns, num_rows);
  for (int64_t column = 0; column < num_columns; ++column) {
    result.row(pivots[column] - 1) = inverse.row(column);
  }
  if (!result.allFinite()) {
    throw std::runtime_error(
        "qr_pseudoinverse produced nonfinite coefficients.");
  }
  return result;
}

}  // namespace qdk::chemistry::scf
