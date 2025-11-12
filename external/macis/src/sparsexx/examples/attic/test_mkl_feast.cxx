/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/io/read_rb.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/util/submatrix.hpp>
#include <sparsexx/wrappers/mkl_dss_solver.hpp>
#include <sparsexx/wrappers/mkl_sparse_matrix.hpp>

#include "mkl.h"

int main(int argc, char** argv) {
  assert(argc == 2);

  using spmat_type = sparsexx::csr_matrix<double, MKL_INT>;
  auto A = sparsexx::read_mm<spmat_type>(std::string(argv[1]));
  const MKL_INT N = A.m();

  auto* nzval = A.nzval().data();
  auto* colind = A.colind().data();
  auto* rowptr = A.rowptr().data();

  std::array<MKL_INT, 128> fpm;
  feastinit(fpm.data());
  fpm[0] = 1;
  fpm[2] = 8;

  double epsout;
  MKL_INT loop, m_true, info;

  double emin = 25.;
  double emax = 25.1;
  MKL_INT m0 = 400;

  std::vector<double> W(m0), X(N * m0), RES(m0);
  auto feast_st = std::chrono::high_resolution_clock::now();
  dfeast_scsrev("F", &N, nzval, rowptr, colind, fpm.data(), &epsout, &loop,
                &emin, &emax, &m0, W.data(), X.data(), &m_true, RES.data(),
                &info);
  auto feast_en = std::chrono::high_resolution_clock::now();
  if (info) throw std::runtime_error("FEAST DIED");

  std::cout << std::chrono::duration<double>(feast_en - feast_st).count()
            << std::endl;
  std::cout << "W" << std::scientific << std::setprecision(10) << std::endl;
  for (auto w : W) std::cout << w << std::endl;

  return 0;
}
