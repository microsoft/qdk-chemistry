/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <macis/csr_hamiltonian.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/solvers/davidson.hpp>
#include <macis/util/fcidump.hpp>
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
  spdlog::drop("davidson");
}

TEST_CASE("Davidson at the full subspace dimension") {
  ROOT_ONLY(MPI_COMM_WORLD);
  const double s = std::ldexp(1.0, 1023);
  sparsexx::csr_matrix<double, int32_t> H(2, 2, {0, 2, 4}, {0, 1, 0, 1},
                                          {-s, s / 4, s / 4, s});
  const auto op = macis::SparseMatrixOperator(H);
  const double D[] = {-s, s};
  double X[] = {1., 0.};

  SECTION("attainable absolute residual tolerance") {
    const double tol = s * 1e-12;
    const double e = macis::davidson(2, 10, op, D, tol, X).second / s;
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(-std::sqrt(17.) / 4, tol / s));
    REQUIRE_THAT(std::hypot(X[0], X[1]),
                 Catch::Matchers::WithinAbs(1., tol / s));
    REQUIRE(std::hypot(-X[0] + X[1] / 4 - e * X[0],
                       X[0] / 4 + X[1] - e * X[1]) < tol / s);
  }

  SECTION("unmet tolerance is a convergence failure") {
    // The strict residual test cannot succeed, even if a BLAS rounds it to 0.
    REQUIRE_THROWS_WITH(macis::davidson(2, 10, op, D, 0., X),
                        "Davidson Did Not Converge!");
  }
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
  spdlog::drop("davidson");
}
#endif
