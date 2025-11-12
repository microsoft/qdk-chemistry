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
#include <sparsexx/wrappers/mkl_dss_solver.hpp>
#include <sparsexx/wrappers/mkl_sparse_matrix.hpp>

#include "mkl.h"

std::string get_mkl_error_string(sparse_status_t s) {
  switch (s) {
    case SPARSE_STATUS_SUCCESS:
      return "The operation was successful.";

    case SPARSE_STATUS_NOT_INITIALIZED:
      return "The routine encountered an empty handle or matrix array.";

    case SPARSE_STATUS_ALLOC_FAILED:
      return "Internal memory allocation failed.";

    case SPARSE_STATUS_INVALID_VALUE:
      return "The input parameters contain an invalid value.";

    case SPARSE_STATUS_EXECUTION_FAILED:
      return "Execution failed.";

    case SPARSE_STATUS_INTERNAL_ERROR:
      return "An error in algorithm implementation occurred.";

    case SPARSE_STATUS_NOT_SUPPORTED:
      return "NOT SUPPORTED";

    default:
      return "UNKNOWN";
  }
}

template <typename T>
void comma_sep(std::ostream& out, const T& v) {
  out << v;
}

template <typename T, typename... Args>
void comma_sep(std::ostream& out, const T& v, Args&&... args) {
  out << v << ", ";
  comma_sep(out, std::forward<Args>(args)...);
}

template <typename T>
MKL_INT cholqr(MKL_INT N, MKL_INT K, T* V, MKL_INT LDV, T* R, MKL_INT LDR) {
  T one = 1., zero = 0.;
  MKL_INT info;
  dgemm("T", "N", &K, &K, &N, &one, V, &LDV, V, &LDV, &zero, R, &LDR);
  dpotrf("U", &K, R, &LDR, &info);
  if (info) throw std::runtime_error("DEAD");

  dtrsm("R", "U", "N", "N", &N, &K, &one, R, &LDR, V, &LDV);
  return info;
}

int main(int argc, char** argv) {
#if 0
  const int N = 10000000;
  int K = 10;
  sparsexx::csr_matrix<double, int32_t> A( N, N, N + 2*(N-1) );
  static_assert( sparsexx::detail::is_csr_matrix_v<decltype(A)>, "This should be a CSR Matrix" );
  static_assert( !sparsexx::detail::mkl::is_mkl_sparse_matrix_v<decltype(A)>, "This should not be an MKL matrix" );

  sparsexx::mkl_csr_matrix<double, int32_t> A_mkl( N, N, N + 2*(N-1) );
  static_assert( sparsexx::detail::is_csr_matrix_v<decltype(A_mkl)>, "This should be a CSR Matrix" );
  static_assert( sparsexx::detail::mkl::is_mkl_sparse_matrix_v<decltype(A_mkl)>, "This should be an MKL matrix" );

  size_t cnt = 0;
  A.nzval()[cnt] = 2; A.nzval()[cnt+1] = -1;
  A.colind()[cnt] = 1; A.colind()[cnt+1] = 2;
  A.rowptr()[0] = 1; A.rowptr()[1] = 3;
  cnt += 2;
  for( int i = 1; i < N-1; ++i ) {
    A.nzval()[cnt+0] = -1.; A.colind()[cnt+0] = i;
    A.nzval()[cnt+1] = 2; A.colind()[cnt+1] = i + 1;
    A.nzval()[cnt+2] = -1.; A.colind()[cnt+2] = i + 2;
    cnt += 3;
    A.rowptr()[i+1] = A.rowptr()[i] + 3;
  }
  A.nzval()[cnt] = -1; A.nzval()[cnt+1] = 2.;
  A.colind()[cnt] = N-1; A.colind()[cnt+1] = N;
  A.rowptr()[N] = A.rowptr()[N-1] + 2;
  cnt += 2;
  assert( cnt ==  A.nnz() );
#else
  assert(argc == 2);
  auto A = sparsexx::read_mm<double, sparsexx::detail::mkl::int_type>(
      std::string(argv[1]));
  sparsexx::mkl_csr_matrix<double, int32_t> A_mkl(A.m(), A.n(), A.nnz(),
                                                  A.indexing());

  const int N = A.m();
  const int K = 10;
#endif

  // std::vector<double> A_dense(N*N);
  // sparsexx::convert_to_dense( A, A_dense.data(), N );

  sparsexx::spsolve::bunch_kaufman<decltype(A)> bk_solver(
      sparsexx::spsolve::create_mkl_bunch_kaufman_solver<decltype(A)>(A));

  comma_sep(std::cout, A.m(), A.n(), A.nnz());
  std::cout << std::endl;
  comma_sep(std::cout, A_mkl.m(), A_mkl.n(), A_mkl.nnz());
  std::cout << std::endl;

  std::copy(A.rowptr().begin(), A.rowptr().end(), A_mkl.rowptr().begin());
  std::copy(A.colind().begin(), A.colind().end(), A_mkl.colind().begin());
  std::copy(A.nzval().begin(), A.nzval().end(), A_mkl.nzval().begin());

  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&]() { return dist(gen); };

  std::vector<double> V(N * K), AV(V.size()), R(K * K);
  std::generate(V.begin(), V.end(), rand_gen);

  // A_mkl.optimize();

  // Power iteration
  auto power_st = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    if (i % 100 == 0) std::cout << "x " << std::flush;
    cholqr(N, K, V.data(), N, R.data(), K);
    sparsexx::spblas::gespmbv(K, 1., A_mkl, V.data(), N, 0., AV.data(), N);
    std::copy(AV.begin(), AV.end(), V.begin());
  }
  std::cout << std::endl;
  cholqr(N, K, V.data(), N, R.data(), K);
  auto power_en = std::chrono::high_resolution_clock::now();

  std::cout
      << "POWER DUR = "
      << std::chrono::duration<double, std::milli>(power_en - power_st).count()
      << std::endl;

  // Rayleigh-Ritz
  sparsexx::spblas::gespmbv(K, 1., A, V.data(), N, 0., AV.data(), N);

  double one = 1., zero = 0.;
  dgemm("T", "N", &K, &K, &N, &one, V.data(), &N, AV.data(), &N, &zero,
        R.data(), &K);

  int LWORK = -1;
  double dummy = 1;
  std::vector<double> W(K);
  int info;
  dsyev("V", "L", &K, R.data(), &K, W.data(), &dummy, &LWORK, &info);
  LWORK = dummy;
  std::vector<double> WORK(LWORK);
  dsyev("V", "L", &K, R.data(), &K, W.data(), WORK.data(), &LWORK, &info);

  std::cout << std::scientific << std::setprecision(10);
  for (auto w : W) std::cout << w << std::endl;

  return 0;
}
