// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf/diis.h"

namespace qdk::chemistry::scf {
DIIS::DIIS(size_t subspace_size) : subspace_size_(subspace_size) {}

void DIIS::extrapolate(const RowMajorMatrix& x, const RowMajorMatrix& error,
                       RowMajorMatrix* x_diis) {
  *x_diis = x;
  if (hist_.size() == subspace_size_) delete_oldest_();
  hist_.push_back(x);
  errors_.push_back(error);

  size_t n = hist_.size();
  RowMajorMatrix B_old = B_;
  B_ = RowMajorMatrix::Zero(n, n);
  B_.block(0, 0, n - 1, n - 1) = B_old;

  // Build overlap matrix of error vectors
  for (size_t i = 0; i < n; i++) {
    B_(i, n - 1) = B_(n - 1, i) = errors_[i].cwiseProduct(errors_[n - 1]).sum();
  }

  for (;;) {
    size_t rank = hist_.size() + 1;
    // Set up the DIIS linear system: A c = rhs
    RowMajorMatrix A(rank, rank);
    A.col(0).setConstant(-1.0);
    A.row(0).setConstant(-1.0);
    A(0, 0) = 0.0;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rank);
    rhs[0] = -1.0;

    double b_max = B_.maxCoeff();
    if (b_max == 0.0) {
      // Fallback: just return the input x without extrapolation
      *x_diis = x;
      return;
    }
    A.block(1, 1, rank - 1, rank - 1) =
        B_.block(0, 0, rank - 1, rank - 1) / b_max;

    Eigen::ColPivHouseholderQR<RowMajorMatrix> qr = A.colPivHouseholderQr();
    Eigen::VectorXd c = qr.solve(rhs);
    double absdet = qr.absDeterminant();

    const double diis_linear_dependence_threshold = 1e-12;
    if (absdet < diis_linear_dependence_threshold) {
      delete_oldest_();
    } else {
      x_diis->setZero();
      for (size_t i = 0; i < hist_.size(); i++) {
        *x_diis += c[i + 1] * hist_[i];
      }
      break;
    }
  }
}

void DIIS::delete_oldest_() {
  hist_.pop_front();
  errors_.pop_front();
  size_t sz = B_.rows();
  B_ = B_.block(1, 1, sz - 1, sz - 1);
}
}  // namespace qdk::chemistry::scf
