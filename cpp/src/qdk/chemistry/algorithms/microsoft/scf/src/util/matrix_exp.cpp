// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "matrix_exp.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "util/blas.h"
#include "util/lapack.h"

namespace qdk::chemistry::scf {

void matrix_exp(const double *m, double *exp_m, int size) {
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
  static constexpr double theta13 =
      5.371920351148152e0;  // matrix 1-norm threshold for order 13
                            // Padé approximation, reference
                            // DOI. 10.1137/04061101X
  int s = first_norm > theta13
              ? std::max(0, (int)(std::log2(first_norm / theta13)) + 1)
              : 0;
  spdlog::debug("computing exp(m), first norm of m {}, scaling factor s {}",
                first_norm, s);
  // scale the matrix by 2^s
  double scale = 1.0 / std::pow(2.0, s);
  std::unique_ptr<double[]> scaled_m(new double[size * size]);
  for (int i = 0; i < size * size; i++) {
    scaled_m[i] = m[i] * scale;
  }

  // Pade approximation, currently const order = 13
  std::unique_ptr<double[]> exp_scaled_m(new double[size * size]);
  pade_approximation(scaled_m.get(), exp_scaled_m.get(), size);

  // square exp_scaled_m matrix s times
  std::unique_ptr<double[]> temp(
      new double[size *
                 size]);  // although blas is column-major, (x*x)^T = x^T * x^T,
                          // so it is fine to call gemm to compute (x*x)^T
  const char NORMAL = 'N';
  for (int order = 0; order < s; order++) {
    blas::gemm(&NORMAL, &NORMAL, size, size, size, 1.0, exp_scaled_m.get(),
               size, exp_scaled_m.get(), size, 0.0, temp.get(), size);
    std::copy(temp.get(), temp.get() + size * size, exp_scaled_m.get());
  }
  std::copy(exp_scaled_m.get(), exp_scaled_m.get() + size * size, exp_m);
}

void pade_approximation(const double *x, double *exp_x, int size) {
  // b is equal to b[i] / b[0] for i = 0 to 13, b[i] are the coefficients for
  // degree 13 Pade approximant, reference DOI. 10.1137/04061101X
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

  std::unique_ptr<double[]> u(new double[size * size]);
  std::unique_ptr<double[]> v(new double[size * size]);
  std::fill_n(u.get(), size * size, 0.0);  // u starts from zero matrix
  std::fill_n(v.get(), size * size, 0.0);  // v starts from x^0 = I
  for (int r_c = 0; r_c < size; r_c++) {   // I matrix
    v[r_c * size + r_c] = 1.0;
  }

  std::unique_ptr<double[]> x_power(new double[size * size]);
  std::copy(v.get(), v.get() + size * size, x_power.get());  // x^0 = I
  const char NORMAL = 'N';
  std::unique_ptr<double[]> temp(new double[size * size]);
  for (int order = 1; order < 14; order++) {
    blas::gemm(&NORMAL, &NORMAL, size, size, size, 1.0, x, size, x_power.get(),
               size, 0.0, temp.get(), size);
    std::copy(temp.get(), temp.get() + size * size, x_power.get());
    if (order % 2 == 1) {  // odd order, add to u
      for (int i = 0; i < size * size; i++) {
        u[i] += b[order] * x_power[i];  // u += b[order] * x^order
      }
    } else {  // even order, add to v
      for (int i = 0; i < size * size; i++) {
        v[i] += b[order] * x_power[i];  // v += b[order] * x^order
      }
    }
  }

  for (int row = 0; row < size;
       row++) {  // transpose u and v to call lapack, since current u and v are
                 // row-major making lapack see them as u^T and v^T
    for (int col = 0; col < row; col++) {
      int lentry_index = row * size + col;
      int uentry_index = col * size + row;
      std::swap(u[lentry_index], u[uentry_index]);
      std::swap(v[lentry_index], v[uentry_index]);
    }
  }

  for (int i = 0; i < size * size;
       i++) {  // prepare for solving (V - U) exp_x = (V + U)
    exp_x[i] = v[i] + u[i];
    v[i] = v[i] - u[i];
  }

  std::unique_ptr<int[]> ipiv(new int[size]);  // pivot indices for lapack
  lapack::getrf(size, size, v.get(), size,
                ipiv.get());  // LU factorization of v, v = L * U
  lapack::getrs(
      &NORMAL, size, size, v.get(), size, ipiv.get(), exp_x,
      size);  // solve the linear equation v * exp_x = u, so exp_x = v^{-1} * u

  for (int row = 0; row < size; row++) {  // transpose exp_x to be row-major
    for (int col = 0; col < row; col++) {
      int lentry_index = row * size + col;
      int uentry_index = col * size + row;
      std::swap(exp_x[lentry_index], exp_x[uentry_index]);
    }
  }
}

}  // namespace qdk::chemistry::scf
