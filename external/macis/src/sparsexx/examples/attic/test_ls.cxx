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
  const int K = 400;

  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&]() { return dist(gen); };

  std::vector<double> B(N * K), X_d(N * K), X_s(N * K);
  std::generate(B.begin(), B.end(), rand_gen);

  // Solve dense
  if (0) {
    std::vector<double> A_dense(N * N);
    sparsexx::convert_to_dense(A, A_dense.data(), N);
    int INFO;
    std::vector<int> IPIV(N);
    std::copy(B.begin(), B.end(), X_d.begin());
    int LWORK = -1;
    std::vector<double> WORK(1);

    dsysv("U", &N, &K, A_dense.data(), &N, IPIV.data(), X_d.data(), &N,
          WORK.data(), &LWORK, &INFO);
    if (INFO) throw std::runtime_error("GESV-LWORK FAILED");

    LWORK = WORK[0];
    WORK.resize(LWORK);
    dsysv("U", &N, &K, A_dense.data(), &N, IPIV.data(), X_d.data(), &N,
          WORK.data(), &LWORK, &INFO);
    if (INFO) throw std::runtime_error("GESV FAILED");
  }

  // Solve sparse
  {
    auto U = sparsexx::extract_upper_triangle(A);
    sparsexx::spsolve::bunch_kaufman<decltype(A)> bk_solver(
        sparsexx::spsolve::create_mkl_bunch_kaufman_solver<decltype(A)>(U));

    std::cout << "BEFORE FACT" << std::endl;
    auto fact_st = std::chrono::high_resolution_clock::now();
    bk_solver.factorize(U);
    auto fact_en = std::chrono::high_resolution_clock::now();
    std::cout << "BEFORE SOLVE" << std::endl;
    auto solve_st = std::chrono::high_resolution_clock::now();
    bk_solver.solve(K, B.data(), N, X_s.data(), N);
    auto solve_en = std::chrono::high_resolution_clock::now();

    std::cout << "Factor "
              << std::chrono::duration<double>(fact_st - fact_en).count()
              << std::endl;
    std::cout << "SOLVE  "
              << std::chrono::duration<double>(solve_st - solve_en).count()
              << std::endl;

    auto [p, n, z] = bk_solver.get_inertia();
    std::cout << "p = " << p << ", n = " << n << ", z = " << z << std::endl;
  }

  /*
    std::cout << std::scientific << std::setprecision(6);
    //for( auto i = 0; i < N; ++i ) {
    //for( auto k = 0; k < K; ++k )
    //  std::cout << std::abs( X_s[i + k*N] - X_d[i + k*N] ) << " ";
    //std::cout << std::endl;
    //}

    for( auto i = 0; i < N*K; ++i )
      X_s[i] = std::abs( X_s[i] - X_d[i] );

    MKL_INT sz = N*K; MKL_INT inc = 1;
    std::cout << dnrm2( &sz, X_s.data(), &inc ) << std::endl;
  */

  return 0;
}
