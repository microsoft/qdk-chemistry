/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <mkl_rci.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <lobpcgxx/lobpcg.hpp>
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

template <typename Op>
double time_op(const Op& op) {
  auto st = std::chrono::high_resolution_clock::now();
  op();
  auto en = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double>(en - st).count();
}

int main(int argc, char** argv) {
  auto A = sparsexx::read_mm<double, sparsexx::detail::mkl::int_type>(
      std::string(argv[1]));
  const int N = A.m();
  const int K = 10;
  const int NR = 1;

  auto D = sparsexx::extract_diagonal_elements(A);

  for (auto i = 0; i < N; ++i) {
    auto b = A.nzval().begin() + A.rowptr()[i] - A.indexing();
    auto e = A.nzval().begin() + A.rowptr()[i + 1] - A.indexing();
    double row_sum = std::accumulate(
        b, e, 0., [](auto x, auto y) { return x + std::abs(y); });
    if (std::abs(D[i]) < (row_sum - std::abs(D[i])))
      std::cout << "Row " << i << " is not diagoanlly dominant" << std::endl;
    std::cout << "Row ratio " << i << " = "
              << std::abs(D[i] / (row_sum - std::abs(D[i]))) << std::endl;
  }

  // Aop
  lobpcgxx::operator_action_type<double> Aop =
      [&](int64_t n, int64_t k, const double* x, int64_t ldx, double* y,
          int64_t ldy) -> void {
    sparsexx::spblas::gespmbv(k, 1., A, x, ldx, 0., y, ldy);
  };

  lobpcgxx::operator_action_type<double> Kop =
      [&](int64_t n, int64_t k, const double* x, int64_t ldx, double* y,
          int64_t ldy) -> void {

  // for( auto j = 0; j < k; ++j )
  // for( auto i = 0; i < n; ++i )
  //   y[ i + j*ldy ] = x[ i + j*ldx ] / D[i];

#if 0
      MKL_INT n_mkl = n;
      MKL_INT k_mkl = k;
      MKL_INT cg_method = 1;
      MKL_INT rci_request;

      lapack::lacpy( lapack::MatrixType::General, n, k, x, ldx, y, ldy );

      std::vector<MKL_INT> ipar( 128 + 2*k );
      std::vector<double>  dpar( 128 + 2*k );
      std::vector<double>  tmp ( n*(3+k)   );
      dcgmrhs_init( &n_mkl, y, &k_mkl, x, &cg_method, &rci_request,
        ipar.data(), dpar.data(), tmp.data() );

      if( rci_request != 0 )
        throw std::runtime_error("CG init failed");

      dcgmrhs_check( &n_mkl, y, &k_mkl, x, &rci_request, ipar.data(),
        dpar.data(), tmp.data() );

      if( rci_request != 0 )
        throw std::runtime_error("CG check failed");
#endif
    std::vector<double> x_cpy(n * k, 0.);

    assert(ldy == n);
    assert(ldx == n);

    for (auto it = 0; it < 10; ++it) {
      // Y = B - A*X
      lapack::lacpy(lapack::MatrixType::General, n, k, x, ldx, y, ldy);
      sparsexx::spblas::gespmbv(k, -1., A, x_cpy.data(), n, 1., y, ldy);

      std::cout << "JIt = " << it << ", nrm =";
      std::cout << "  " << blas::nrm2(n * k, y, 1) << std::endl;

// Y <- Y + D*X = B - A*X + D*X
//              = B - (A - D)*X
//              = B - (L + U)*X
#pragma omp parallel for collapse(2)
      for (auto j = 0; j < k; ++j)
        for (auto i = 0; i < n; ++i) y[i + j * ldy] += x_cpy[i + j * n] * D[i];

// X <- D**-1 * Y = D**-1 * ( B - (L+U)*X )
#pragma omp parallel for collapse(2)
      for (auto j = 0; j < k; ++j)
        for (auto i = 0; i < n; ++i) x_cpy[i + j * n] = y[i + j * ldy] / D[i];
    }

    lapack::lacpy(lapack::MatrixType::General, n, k, x_cpy.data(), n, y, ldy);
  };

  // starting vectors
  std::vector<double> X(N * K);

// Random vectors
#if 1
  std::default_random_engine gen;
  std::normal_distribution<> dist(0., 1.);
  auto rand_gen = [&]() { return dist(gen); };
  std::generate(X.begin(), X.end(), rand_gen);
  lobpcgxx::cholqr(N, K, X.data(), N);  // Orthogonalize
#else
  std::vector<int64_t> ind(N);
  std::iota(ind.begin(), ind.end(), 0);
  std::stable_sort(ind.begin(), ind.end(), [&](auto i, auto j) {
    return std::abs(D[i]) < std::abs(D[j]);
  });
  std::fill(X.begin(), X.end(), 0.);
  for (auto i = 0; i < K; ++i) X[ind[i] + i * N] = 1.;
#endif

  lobpcgxx::lobpcg_settings settings;
  settings.conv_tol = 1e-6;
  settings.maxiter = 1000;
  settings.track_convergence = true;
  lobpcgxx::lobpcg_operator<double> lob_op(Aop);
  // lobpcgxx::lobpcg_operator<double> lob_op( Aop, Kop );

  lobpcgxx::lobpcg_convergence<double> conv;

  std::vector<double> lam(K), res(K);
  auto lobpcg_dur = time_op([&]() {
    lobpcgxx::lobpcg(settings, N, K, NR, lob_op, lam.data(), X.data(), N,
                     res.data(), conv);
  });

  std::cout << "LOBPCG Duration = " << lobpcg_dur << " s ("
            << conv.conv_data.size() << " iterations)" << std::endl;

  std::cout << std::scientific << std::setprecision(10) << std::endl;
  for (int64_t i = 0; i < NR; ++i) {
    std::cout << "  lam[" << i << "] = " << lam[i] << ", res[" << i
              << "] = " << res[i] << std::endl;
  }

  return 0;
}
