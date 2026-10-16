// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "similarity_transform.h"

#include <limits>
#include <stdexcept>

namespace qdk::chemistry::scf {

RowMajorMatrix qr_pseudoinverse(const RowMajorMatrix& coefficients) {
  if (coefficients.cols() == 0 || coefficients.rows() < coefficients.cols() ||
      !coefficients.allFinite()) {
    throw std::invalid_argument(
        "qr_pseudoinverse requires a finite, nonempty tall or square matrix.");
  }

  const auto num_columns = coefficients.cols();
  Eigen::HouseholderQR<RowMajorMatrix> qr(coefficients);
  const auto R = qr.matrixQR().topRows(num_columns);
  const auto diagonal = R.diagonal().cwiseAbs().eval();
  const double singular_tolerance = std::numeric_limits<double>::epsilon() *
                                    coefficients.rows() * diagonal.maxCoeff();
  if (!R.allFinite() || (diagonal.array() <= singular_tolerance).any()) {
    throw std::runtime_error(
        "qr_pseudoinverse requires numerically full column rank.");
  }

  RowMajorMatrix Q = qr.householderQ() *
                     RowMajorMatrix::Identity(coefficients.rows(), num_columns);
  RowMajorMatrix result = R.triangularView<Eigen::Upper>().solve(Q.transpose());
  if (!result.allFinite()) {
    throw std::runtime_error(
        "qr_pseudoinverse produced nonfinite coefficients.");
  }
  return result;
}

void similarity_transform(blas::Layout layout, int64_t N, int64_t K,
                          double ALPHA, const double* A, int64_t LDA,
                          const double* B, int64_t LDB, double BETA, double* C,
                          int64_t LDC, std::vector<double>* workspace) {
  if (A == nullptr || B == nullptr || C == nullptr) {
    throw std::invalid_argument("similarity_transform: null matrix pointer.");
  }
  if (N < 0 || K < 0) {
    throw std::invalid_argument("similarity_transform: negative dimensions.");
  }
  if (N == 0) {
    return;
  }
  if (K == 0) {
    // A**H * B * A vanishes; reduce to C = BETA * C
    if (layout == blas::Layout::RowMajor) {
      for (int64_t row = 0; row < N; ++row)
        for (int64_t col = 0; col < N; ++col) C[row * LDC + col] *= BETA;
    } else {
      for (int64_t col = 0; col < N; ++col)
        for (int64_t row = 0; row < N; ++row) C[col * LDC + row] *= BETA;
    }
    return;
  }

  const size_t required_workspace_size = static_cast<size_t>(K) * N;
  std::vector<double> internal_workspace;
  double* temp_ptr;
  if (workspace == nullptr) {
    internal_workspace.resize(required_workspace_size);
    temp_ptr = internal_workspace.data();
  } else {
    if (workspace->size() < required_workspace_size) {
      workspace->resize(required_workspace_size);
    }
    temp_ptr = workspace->data();
  }

  // LD for the K x N temporary buffer
  const int64_t ld_temp = (layout == blas::Layout::RowMajor) ? N : K;

  // temp = B * A  (K x K * K x N -> K x N)
  blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, K, N, K, 1.0, B, LDB,
             A, LDA, 0.0, temp_ptr, ld_temp);

  // C = ALPHA * A**H * temp + BETA * C  (N x K * K x N -> N x N)
  blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, N, N, K, ALPHA, A, LDA,
             temp_ptr, ld_temp, BETA, C, LDC);
}

}  // namespace qdk::chemistry::scf
