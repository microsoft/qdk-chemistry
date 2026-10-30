/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <macis/csr_hamiltonian.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/solvers/davidson.hpp>
#include <macis/util/fcidump.hpp>
#include <numeric>
#include <sparsexx/util/submatrix.hpp>

#include "ut_common.hpp"

TEST_CASE("Davidson") {
  ROOT_ONLY(MPI_COMM_WORLD);

  if (!spdlog::get("davidson")) {
    spdlog::null_logger_mt("davidson");
  }

  size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t num_occupied_orbitals = 5;

  std::vector<double> T(norb * norb);
  std::vector<double> V(norb * norb * norb * norb);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  using wfn_type = macis::wfn_t<64>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_type = macis::DoubleLoopHamiltonianGenerator<wfn_type>;

  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  // Generate configuration space
  const auto hf_det = wfn_traits::canonical_hf_determinant(
      num_occupied_orbitals, num_occupied_orbitals);
  auto dets = macis::generate_cisd_hilbert_space(norb, hf_det);
  auto E0_ref = -7.623197835987e+01;

  // Generate CSR Hamiltonian
  auto H = macis::make_csr_hamiltonian<int32_t>(dets.begin(), dets.end(),
                                                ham_gen, 1e-16);

  // Obtain lowest eigenvalue
  SECTION("With Vectors") {
    std::vector<double> X(H.n());
    macis::diagonal_guess(H.n(), H, X.data());
    auto D = sparsexx::extract_diagonal_elements(H);
    auto [niter, E0] = macis::davidson(
        H.n(), 15, macis::SparseMatrixOperator(H), D.data(), 1e-8, X.data());

    REQUIRE_THAT(E0 + E_core, Catch::Matchers::WithinAbs(
                                  E0_ref, testing::davidson_tolerance));
    REQUIRE_THAT(
        blas::nrm2(X.size(), X.data(), 1),
        Catch::Matchers::WithinAbs(1.0, testing::numerical_zero_tolerance));

    std::vector<double> AX(X.size());
    sparsexx::spblas::gespmbv(1, 1., H, X.data(), H.n(), 0., AX.data(), H.n());
    REQUIRE_THAT(
        blas::dot(X.size(), X.data(), 1, AX.data(), 1),
        Catch::Matchers::WithinAbs(E0, testing::numerical_zero_tolerance));
  }

  SECTION("Restarted") {
    auto D = sparsexx::extract_diagonal_elements(H);
    // Small search spaces force many restarts; with max_m == 2 a restart
    // keeps only the current Ritz vector
    for (int64_t max_m : {2, 3, 4}) {
      DYNAMIC_SECTION("max_m = " << max_m) {
        std::vector<double> X(H.n());
        macis::diagonal_guess(H.n(), H, X.data());
        auto [niter, E0] = macis::restarted_davidson(
            H.n(), max_m, 500, macis::SparseMatrixOperator(H), D.data(), 1e-8,
            X.data());

        REQUIRE(niter > 4);
        REQUIRE_THAT(E0 + E_core, Catch::Matchers::WithinAbs(
                                      E0_ref, testing::davidson_tolerance));
        REQUIRE_THAT(
            blas::nrm2(X.size(), X.data(), 1),
            Catch::Matchers::WithinAbs(1.0, testing::numerical_zero_tolerance));

        std::vector<double> AX(X.size());
        sparsexx::spblas::gespmbv(1, 1., H, X.data(), H.n(), 0., AX.data(),
                                  H.n());
        REQUIRE_THAT(
            blas::dot(X.size(), X.data(), 1, AX.data(), 1),
            Catch::Matchers::WithinAbs(E0, testing::numerical_zero_tolerance));
      }
    }

    std::vector<double> X_guess(H.n());
    macis::diagonal_guess(H.n(), H, X_guess.data());
    REQUIRE_THROWS(macis::restarted_davidson(H.n(), 4, 3,
                                             macis::SparseMatrixOperator(H),
                                             D.data(), 1e-8, X_guess.data()));

    // Without throwing, too few iterations return the current Ritz pair and
    // its residual norm
    macis::diagonal_guess(H.n(), H, X_guess.data());
    double res = 0.0;
    auto [niter, E3] = macis::restarted_davidson(
        H.n(), 4, 3, macis::SparseMatrixOperator(H), D.data(), 1e-8,
        X_guess.data(), 1e-12, false, &res);
    REQUIRE(niter == 3);
    REQUIRE(res > 1e-8);
    REQUIRE_THAT(
        blas::nrm2(X_guess.size(), X_guess.data(), 1),
        Catch::Matchers::WithinAbs(1.0, testing::numerical_zero_tolerance));
    std::vector<double> R(H.n());
    sparsexx::spblas::gespmbv(1, 1., H, X_guess.data(), H.n(), 0., R.data(),
                              H.n());
    REQUIRE_THAT(blas::dot(H.n(), X_guess.data(), 1, R.data(), 1),
                 Catch::Matchers::WithinAbs(E3, 1e-10));
    blas::axpy(H.n(), -E3, X_guess.data(), 1, R.data(), 1);
    REQUIRE_THAT(blas::nrm2(H.n(), R.data(), 1),
                 Catch::Matchers::WithinRel(res, 1e-6));

    // Inexact operator (H plus a diagonal perturbation of about 1e-6)
    // anchored by H: the Ritz pair after anchoring carries the residual of
    // H, and convergence (even to a tolerance the inexact operator alone
    // would declare met at once) is only accepted after anchoring
    struct PerturbedOperator {
      const decltype(H)& A;
      const std::vector<double>& d;
      void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                           double beta, double* AV, size_t LDAV) const {
        macis::SparseMatrixOperator(A).operator_action(m, alpha, V, LDV, beta,
                                                       AV, LDAV);
        for (size_t j = 0; j < m; ++j)
          for (size_t i = 0; i < d.size(); ++i)
            AV[i + j * LDAV] += alpha * d[i] * V[i + j * LDV];
      }
    };
    // (residual tolerance, anchor tolerance, perturbation): the last case
    // anchors only when the perturbed solve would stop, with a perturbation
    // whose own eigenpair misses the tolerance
    const std::vector<std::array<double, 3>> cases = {
        {1e-6, 1e-3, 1e-6}, {1e-2, 1e-3, 1e-6}, {1e-4, 1e-12, 1e-3}};
    macis::SparseMatrixOperator exact(H);
    for (auto [tol, anchor_tol, scale] : cases) {
      std::vector<double> d(H.n());
      for (size_t i = 0; i < d.size(); ++i) d[i] = scale * std::sin(1.0 + i);
      PerturbedOperator perturbed{H, d};
      macis::diagonal_guess(H.n(), H, X_guess.data());
      auto [nit, E_mixed] = macis::restarted_davidson(
          H.n(), 4, 500, perturbed, D.data(), tol, X_guess.data(), 1e-12, false,
          &res, 0, &exact, anchor_tol);
      REQUIRE(res < tol);
      sparsexx::spblas::gespmbv(1, 1., H, X_guess.data(), H.n(), 0., R.data(),
                                H.n());
      const double rq = blas::dot(H.n(), X_guess.data(), 1, R.data(), 1);
      blas::axpy(H.n(), -rq, X_guess.data(), 1, R.data(), 1);
      REQUIRE(blas::nrm2(H.n(), R.data(), 1) < tol);
      REQUIRE_THAT(blas::nrm2(H.n(), R.data(), 1),
                   Catch::Matchers::WithinAbs(res, 1e-2 * tol));
      REQUIRE_THAT(E_mixed, Catch::Matchers::WithinAbs(rq, 10 * tol * tol));
      if (tol < 1e-5)
        REQUIRE_THAT(
            E_mixed + E_core,
            Catch::Matchers::WithinAbs(E0_ref, testing::davidson_tolerance));
    }

    // Exact block of the preconditioner on the determinants of lowest
    // diagonal: the same eigenpair in fewer iterations. The guess is the
    // block's own lowest eigenvector, so at the first iteration lam equals
    // its lowest eigenvalue (that component is dropped).
    {
      const int64_t n = H.n(), nblock = 100;
      std::vector<int64_t> order(n);
      std::iota(order.begin(), order.end(), int64_t(0));
      std::partial_sort(order.begin(), order.begin() + nblock, order.end(),
                        [&](int64_t a, int64_t b) { return D[a] < D[b]; });
      macis::DavidsonBlockPreconditioner block;
      block.index.assign(order.begin(), order.begin() + nblock);
      std::sort(block.index.begin(), block.index.end());
      std::vector<int64_t> where(n, -1);
      for (int64_t a = 0; a < nblock; ++a) where[block.index[a]] = a;
      block.Z.assign(nblock * nblock, 0.0);
      const auto base = H.indexing();
      for (int64_t a = 0; a < nblock; ++a) {
        const int64_t row = block.index[a];
        for (auto e = H.rowptr()[row] - base; e < H.rowptr()[row + 1] - base;
             ++e) {
          const int64_t b = where[H.colind()[e] - base];
          if (b >= 0) block.Z[a + b * nblock] = H.nzval()[e];
        }
      }
      block.lambda.resize(nblock);
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, nblock,
                   block.Z.data(), nblock, block.lambda.data());

      macis::SparseMatrixOperator op(H);
      auto block_guess = [&] {
        std::vector<double> X(n, 0.0);
        for (int64_t a = 0; a < nblock; ++a) X[block.index[a]] = block.Z[a];
        return X;
      };
      std::vector<double> X_diag = block_guess(), X_block = block_guess();
      auto [nit_diag, E_diag] = macis::restarted_davidson(
          n, 3, 500, op, D.data(), 1e-8, X_diag.data());
      auto [nit_block, E_block] = macis::restarted_davidson(
          n, 3, 500, op, D.data(), 1e-8, X_block.data(), 1e-12, true, nullptr,
          0, static_cast<const decltype(op)*>(nullptr), 0.0, &block);
      REQUIRE(nit_block < nit_diag);
      REQUIRE_THAT(E_block, Catch::Matchers::WithinAbs(E_diag, 1e-12));
      REQUIRE_THAT(E_block + E_core, Catch::Matchers::WithinAbs(
                                         E0_ref, testing::davidson_tolerance));
      sparsexx::spblas::gespmbv(1, 1., H, X_block.data(), n, 0., R.data(), n);
      blas::axpy(n, -E_block, X_block.data(), 1, R.data(), 1);
      REQUIRE(blas::nrm2(n, R.data(), 1) < 1e-8);
    }
  }
  testing::drop_test_loggers();
}

TEST_CASE("Restarted Davidson Long Tail") {
  ROOT_ONLY(MPI_COMM_WORLD);
  if (!spdlog::get("davidson")) spdlog::null_logger_mt("davidson");

  // A vector with one dominant component and a long tail of components whose
  // squares lie below half an ulp of 1, as in large CI vectors: one running
  // sum per thread rounds the whole tail of the first thread away, which put
  // the eigenvalue 2e-12 (32 threads) to 6e-11 (one thread) too high here.
  // The guess is converged at the loose tolerance, so the returned eigenvalue
  // is its Rayleigh quotient, from the start vector and, when anchored, from
  // the anchor's image.
  struct Diagonal {
    const std::vector<double>& d;
    void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                         double beta, double* AV, size_t LDAV) const {
      for (size_t j = 0; j < m; ++j)
        for (size_t i = 0; i < d.size(); ++i)
          AV[i + j * LDAV] = alpha * d[i] * V[i + j * LDV] +
                             (beta == 0.0 ? 0.0 : beta * AV[i + j * LDAV]);
    }
  };
  const int64_t n = int64_t(1) << 20;
  std::vector<double> d(n, 1.0), X0(n);
  d[0] = -1.0;
  X0[0] = 1.0;
  long double xx = 1.0L, xdx = -1.0L;
  for (int64_t i = 1; i < n; ++i) {
    X0[i] = 5e-9 * (1.0 + 0.5 * std::sin(double(i)));
    xx += (long double)X0[i] * X0[i];
    xdx += (long double)X0[i] * X0[i];
  }
  const double rho = double(xdx / xx);
  REQUIRE(rho + 1.0 > 5e-11);  // the tail matters at this tolerance

  const Diagonal op{d};
  for (bool anchored : {false, true}) {
    std::vector<double> X = X0;
    auto [niter, E] = macis::restarted_davidson(
        n, 3, 10, op, d.data(), 1e-4, X.data(), 1e-12, true, nullptr, 0,
        anchored ? &op : nullptr, 1e-4);
    REQUIRE(niter == (anchored ? 2 : 1));
    REQUIRE_THAT(E, Catch::Matchers::WithinAbs(rho, 1e-13));
  }
  testing::drop_test_loggers();
}

#ifdef MACIS_ENABLE_MPI
TEST_CASE("Parallel Davidson") {
  if (!spdlog::get("davidson")) {
    auto l = spdlog::null_logger_mt("davidson");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  size_t num_occupied_orbitals = 5;

  std::vector<double> T(norb * norb);
  std::vector<double> V(norb * norb * norb * norb);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  using wfn_type = macis::wfn_t<64>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_type = macis::DoubleLoopHamiltonianGenerator<wfn_type>;

  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  // Generate configuration space
  const auto hf_det = wfn_traits::canonical_hf_determinant(
      num_occupied_orbitals, num_occupied_orbitals);
  auto dets = macis::generate_cisd_hilbert_space(norb, hf_det);
  auto E0_ref = -7.623197835987e+01;

  // Generate CSR Hamiltonian
  auto H = macis::make_dist_csr_hamiltonian<int32_t>(
      MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, 1e-16);
  auto spmv_info = sparsexx::spblas::generate_spmv_comm_info(H);

  // Obtain lowest eigenvalue
  SECTION("With Vectors") {
    std::vector<double> X_local(H.local_row_extent());
    macis::p_diagonal_guess(X_local.size(), H, X_local.data());
    auto D_local = sparsexx::extract_diagonal_elements(H.diagonal_tile());
    auto [niter, E0] =
        macis::p_davidson(X_local.size(), 15, macis::SparseMatrixOperator(H),
                          D_local.data(), 1e-8, X_local.data(), MPI_COMM_WORLD);

    REQUIRE_THAT(E0 + E_core, Catch::Matchers::WithinAbs(
                                  E0_ref, testing::davidson_tolerance));
    double nrm =
        blas::dot(X_local.size(), X_local.data(), 1, X_local.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nrm = std::sqrt(nrm);
    REQUIRE_THAT(nrm, Catch::Matchers::WithinAbs(
                          1.0, testing::numerical_zero_tolerance));

    std::vector<double> AX_local(X_local.size());
    sparsexx::spblas::pgespmv(1., H, X_local.data(), 0., AX_local.data(),
                              spmv_info);
    double inner =
        blas::dot(X_local.size(), AX_local.data(), 1, X_local.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &inner, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    REQUIRE_THAT(inner, Catch::Matchers::WithinAbs(
                            E0, testing::numerical_zero_tolerance));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  testing::drop_test_loggers();
}
#endif
