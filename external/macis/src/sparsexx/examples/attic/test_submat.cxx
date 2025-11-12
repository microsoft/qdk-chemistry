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
  const int N = A.m();

  auto A_sub = sparsexx::extract_submatrix(A, {1, 2}, {4, 4});
  const int M_sub = A_sub.m();
  const int N_sub = A_sub.n();

  std::vector<double> A_dense(N * N);
  std::vector<double> A_sub_dense(N_sub * M_sub);

  sparsexx::convert_to_dense(A, A_dense.data(), N);
  sparsexx::convert_to_dense(A_sub, A_sub_dense.data(), M_sub);

  std::cout << "A Matrix" << std::endl;
  for (auto i = 0; i < A.m(); ++i) {
    for (auto j = 0; j < A.n(); ++j)
      std::cout << std::setw(5) << A_dense[i + j * A.m()];
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "A_sub Matrix" << std::endl;
  for (auto i = 0; i < A_sub.m(); ++i) {
    for (auto j = 0; j < A_sub.n(); ++j)
      std::cout << std::setw(5) << A_sub_dense[i + j * A_sub.m()];
    std::cout << std::endl;
  }

  auto D = sparsexx::extract_diagonal_elements(A);
  std::cout << "D" << std::endl;
  for (auto d : D) std::cout << d << std::endl;

  return 0;
}
