// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "matrix_exp.h"

#include <Eigen/Dense>
#include <algorithm>
#include <blas.hh>
#include <cmath>
#include <iostream>
#include <lapack.hh>
#include <memory>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

namespace qdk::chemistry::scf {

void matrix_exp(const double *m, double *exp_m, int size) {
  QDK_LOG_TRACE_ENTERING();
  // confirm the scaling factor 2^s
  double first_norm = 0.0;  // maximum column sum
  for (int row = 0; row < size; row++) {
    double abssum_row = 0.0;
    for (int col = 0; col < size; col++) {
      double abs_entry =
          std::fabs(m[row * size + col]);  // input m is row-major
      abssum_row += abs_entry;
    }
    if (abssum_row > first_norm) first_norm = abssum_row;
  }

  // matrix 1-norm threshold for order 13 Padé approximation
  // Reference: Nicholas J. Higham (2005) DOI:10.1137/04061101X
  static constexpr double theta13 = 5.371920351148152e0;
  int s = first_norm > theta13
              ? std::max(0, (int)(std::log2(first_norm / theta13)) + 1)
              : 0;
  QDK_LOGGER().debug(
      "computing exp(m), first norm of m {}, scaling factor s {}", first_norm,
      s);
  // scale the matrix by 2^s
  double scale = 1.0 / std::pow(2.0, s);
  auto scaled_m = std::make_unique<double[]>(size * size);
  for (int i = 0; i < size * size; i++) {
    scaled_m[i] = m[i] * scale;
  }

  // Pade approximation, currently const order = 13
  auto exp_scaled_m = std::make_unique<double[]>(size * size);
  pade_approximation(scaled_m.get(), exp_scaled_m.get(), size);

  // square exp_scaled_m matrix s times. (x*x)^T = x^T * x^T,  so it is fine to
  // call gemm to compute (x*x)^T
  auto temp = std::make_unique<double[]>(size * size);
  for (int order = 0; order < s; order++) {
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               size, size, size, 1.0, exp_scaled_m.get(), size,
               exp_scaled_m.get(), size, 0.0, temp.get(), size);
    std::copy(temp.get(), temp.get() + size * size, exp_scaled_m.get());
  }
  std::copy(exp_scaled_m.get(), exp_scaled_m.get() + size * size, exp_m);
}

void pade_approximation(const double *x, double *exp_x, int size) {
  QDK_LOG_TRACE_ENTERING();
  // b is equal to b[i] / b[0] for i = 0 to 13, b[i] are the coefficients for
  // degree 13 Pade approximant.
  // Reference: Nicholas J. Higham (2005) DOI:10.1137/04061101X
  static constexpr double b[14] = {1.,
                                   0.5,
                                   0.12,
                                   0.0183333333,
                                   0.00199275362,
                                   1.63043478E-4,
                                   1.03519669E-5,
                                   5.17598344E-7,
                                   2.04315136E-8,
                                   6.30602271E-10,
                                   1.48377005E-11,
                                   2.52915349E-13,
                                   2.81017055E-15,
                                   1.54404975E-17};

  // Map input and output arrays to Eigen matrices
  Eigen::Map<const Eigen::MatrixXd> x_matrix(x, size, size);
  Eigen::Map<Eigen::MatrixXd> exp_x_matrix(exp_x, size, size);

  // Create matrices u and v. u starts from zero matrix, v starts from x^0 = I
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(size, size);
  Eigen::MatrixXd v = Eigen::MatrixXd::Identity(size, size);

  Eigen::MatrixXd x_power = Eigen::MatrixXd::Identity(size, size);  // x^0 = I
  Eigen::MatrixXd temp(size, size);
  for (int order = 1; order < 14; order++) {
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               size, size, size, 1.0, x_matrix.data(), size, x_power.data(),
               size, 0.0, temp.data(), size);
    x_power = temp;
    if (order % 2 == 1) {       // odd order, add to u
      u += b[order] * x_power;  // u += b[order] * x^order
    } else {                    // even order, add to v
      v += b[order] * x_power;  // v += b[order] * x^order
    }
  }

  // transpose u and v to call lapack, since current u and v are row-major
  // making lapack see them as u^T and v^T
  u.transposeInPlace();
  v.transposeInPlace();

  // prepare for solving (V - U) exp_x = (V + U)
  exp_x_matrix = v + u;
  v = v - u;

  auto ipiv = std::make_unique<int64_t[]>(size);  // pivot indices for lapack
  // LU factorization of v, v = L * U
  lapack::getrf(size, size, v.data(), size, ipiv.get());
  // solve the linear equation v * exp_x = u, so exp_x = v^{-1} * u
  lapack::getrs(lapack::Op::NoTrans, size, size, v.data(), size, ipiv.get(),
                exp_x_matrix.data(), size);

  // transpose exp_x to be row-major
  exp_x_matrix.transposeInPlace();
}

}  // namespace qdk::chemistry::scf
