/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <blas.hh>
#include <cmath>
#include <lapack.hh>
#include <limits>
#include <macis/csr_hamiltonian.hpp>
#include <macis/fci/fci_hamiltonian.hpp>
#include <macis/fci/fci_rdms.hpp>
#include <macis/fci/string_space.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/sd_operations.hpp>
#include <macis/solvers/davidson.hpp>
#include <macis/util/fcidump.hpp>
#include <numeric>
#include <random>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <tuple>
#include <vector>

#include "ut_common.hpp"

namespace {

using spin_string_t = std::bitset<64>;
using wfn_type = macis::wfn_t<64>;

// Random real integrals with the 8-fold permutational symmetry of (pq|rs) and
// a symmetric one-body matrix; column-major with leading dimension norb.
void random_integrals(size_t norb, std::mt19937& gen, std::vector<double>& T,
                      std::vector<double>& V) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const size_t n2 = norb * norb;
  T.assign(n2, 0.0);
  V.assign(n2 * n2, 0.0);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q <= p; ++q)
      T[p + q * norb] = T[q + p * norb] = dist(gen);

  auto v = [&](size_t p, size_t q, size_t r, size_t s) -> double& {
    return V[p + q * norb + r * n2 + s * n2 * norb];
  };
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q <= p; ++q)
      for (size_t r = 0; r < norb; ++r)
        for (size_t s = 0; s <= r; ++s) {
          if (macis::fci_pair_index(r, s) > macis::fci_pair_index(p, q))
            continue;
          const double x = dist(gen);
          v(p, q, r, s) = v(q, p, r, s) = v(p, q, s, r) = v(q, p, s, r) = x;
          v(r, s, p, q) = v(s, r, p, q) = v(r, s, q, p) = v(s, r, q, p) = x;
        }
}

// || a - b || / max(|| b ||, 1) for a single-precision result a
double relative_error(const std::vector<float>& a,
                      const std::vector<double>& b) {
  double err2 = 0.0, ref2 = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    const double d = double(a[i]) - b[i];
    err2 += d * d;
    ref2 += b[i] * b[i];
  }
  return std::sqrt(err2 / std::max(ref2, 1.0));
}

}  // namespace

TEST_CASE("FCI String Space") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // (norb, nelec) pairs covering empty, full, single-orbital and generic cases
  const std::vector<std::pair<size_t, size_t>> cases = {
      {1, 0}, {1, 1}, {4, 0}, {4, 4}, {4, 2}, {6, 3}, {7, 1}, {8, 3}, {10, 4}};

  for (auto [norb, nelec] : cases) {
    DYNAMIC_SECTION("norb = " << norb << " nelec = " << nelec) {
      macis::FCIStringSpace space(norb, nelec);
      auto ref = macis::generate_combs<spin_string_t>(norb, nelec);

      REQUIRE(space.norb() == norb);
      REQUIRE(space.nelec() == nelec);
      REQUIRE(space.size() == ref.size());
      REQUIRE(space.num_excitations() == nelec * (norb - nelec));

      for (size_t i = 0; i < space.size(); ++i) {
        // Same order as generate_combs (and hence generate_hilbert_space)
        std::vector<uint32_t> ref_occ;
        for (uint32_t p = 0; p < 64; ++p)
          if (ref[i][p]) ref_occ.push_back(p);
        std::vector<uint32_t> occ(space.occupied(i), space.occupied(i) + nelec);
        REQUIRE(occ == ref_occ);
        REQUIRE(space.address(occ.data()) == i);

        // Single excitations E_{create,destroy}|i> = sign |target>
        std::vector<std::pair<uint32_t, uint32_t>> seen;
        const auto* ex = space.excitations(i);
        for (size_t e = 0; e < space.num_excitations(); ++e) {
          const auto c = ex[e].create;
          const auto d = ex[e].destroy;
          REQUIRE(ref[i][d]);
          REQUIRE_FALSE(ref[i][c]);
          auto target = ref[i];
          target.flip(d).flip(c);
          REQUIRE(ref[ex[e].target] == target);
          REQUIRE(ex[e].sign == macis::single_excitation_sign(ref[i], c, d));
          REQUIRE(ex[e].pair == macis::fci_pair_index(c, d));
          seen.emplace_back(c, d);
        }
        std::sort(seen.begin(), seen.end());
        REQUIRE(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
      }
    }
  }

  SECTION("Packed pair index") {
    REQUIRE(macis::fci_pair_index(0, 0) == 0);
    REQUIRE(macis::fci_pair_index(1, 0) == 1);
    REQUIRE(macis::fci_pair_index(0, 1) == 1);
    REQUIRE(macis::fci_pair_index(1, 1) == 2);
    REQUIRE(macis::fci_pair_index(3, 2) == 8);
    REQUIRE(macis::fci_pair_index(2, 3) == 8);
  }
}

TEST_CASE("FCI Hamiltonian") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // (norb, nalpha, nbeta): empty, single orbital, closed/open shell, one spin
  // empty or full, nalpha < nbeta and generic cases
  const std::vector<std::array<size_t, 3>> cases = {
      {1, 0, 0},
      {1, 1, 0},
      {1, 1, 1},
      {2, 1, 1},
      {3, 1, 2},
      {4, 2, 2},
      {4, 0, 3},
      {4, 3, 0},
      {5, 2, 3},
      {6, 3, 3},
      {6, 4, 2},
      {7, 2, 5},
      {8, 4, 4},
      {9, 3, 4},
      {3, 3, 1},
      {4, 2, 4},
      // many cliques but few same-spin tiles (tiles split over tasks)
      {13, 4, 1},
      {13, 1, 4}};

  std::mt19937 gen(1234);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto [norb, na, nb] : cases) {
    DYNAMIC_SECTION("norb = " << norb << " nalpha = " << na
                              << " nbeta = " << nb) {
      std::vector<double> T, V;
      random_integrals(norb, gen, T, V);

      macis::FCIHamiltonian H(norb, na, nb, T.data(), norb, V.data(), norb);
      macis::DoubleLoopHamiltonianGenerator<wfn_type> ham_gen(
          macis::matrix_span<double>(T.data(), norb, norb),
          macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
      auto dets = macis::generate_hilbert_space<wfn_type>(norb, na, nb);
      const size_t ndet = dets.size();
      REQUIRE(H.size() == ndet);

      auto H_ref = macis::make_csr_hamiltonian<int64_t>(
          dets.begin(), dets.end(), ham_gen, 0.0);

      std::vector<double> x(ndet), y(ndet);
      for (auto& xi : x) xi = dist(gen);
      for (auto& yi : y) yi = dist(gen);

      std::vector<double> sigma(ndet), sigma_ref(ndet);
      H.sigma(x.data(), sigma.data());
      sparsexx::spblas::gespmbv(1, 1.0, H_ref, x.data(), ndet, 0.0,
                                sigma_ref.data(), ndet);
      for (size_t i = 0; i < ndet; ++i)
        REQUIRE_THAT(sigma[i], Catch::Matchers::WithinAbs(sigma_ref[i], 1e-10));

      // Every kernel level this CPU supports, in both precisions
      REQUIRE(H.simd() == macis::fci_simd_supported());
      for (int level = 0; level <= int(macis::fci_simd_supported()); ++level) {
        H.set_simd(macis::FCISimd(level));
        H.sigma(x.data(), sigma.data());
        for (size_t i = 0; i < ndet; ++i)
          REQUIRE_THAT(sigma[i],
                       Catch::Matchers::WithinAbs(sigma_ref[i], 1e-10));
        std::vector<float> xf(x.begin(), x.end()), sf(ndet);
        H.sigma(xf.data(), sf.data());
        REQUIRE(relative_error(sf, sigma_ref) < 1e-5);
      }
      H.set_simd(macis::fci_simd_supported());
      if (macis::fci_simd_supported() != macis::FCISimd::avx512)
        REQUIRE_THROWS_AS(H.set_simd(macis::FCISimd::avx512),
                          std::invalid_argument);

      std::vector<double> D(ndet);
      H.diagonal(D.data());
      for (size_t i = 0; i < ndet; ++i)
        REQUIRE_THAT(D[i],
                     Catch::Matchers::WithinAbs(
                         ham_gen.matrix_element(dets[i], dets[i]), 1e-11));

      // Davidson operator contract: AV = alpha * H * V + beta * AV
      std::vector<double> AV = y;
      H.operator_action(1, 2.0, x.data(), ndet, -0.5, AV.data(), ndet);
      for (size_t i = 0; i < ndet; ++i)
        REQUIRE_THAT(AV[i], Catch::Matchers::WithinAbs(
                                2.0 * sigma_ref[i] - 0.5 * y[i], 1e-10));

      // Single precision: the same contraction to float accuracy, for
      // S = H C and S = alpha H C + beta S
      std::vector<float> xf(x.begin(), x.end()), sf(ndet);
      H.sigma(xf.data(), sf.data());
      REQUIRE(relative_error(sf, sigma_ref) < 1e-5);
      sf.assign(y.begin(), y.end());
      H.sigma(xf.data(), sf.data(), 2.0f, -0.5f);
      std::vector<double> scaled_ref(ndet);
      for (size_t i = 0; i < ndet; ++i)
        scaled_ref[i] = 2.0 * sigma_ref[i] - 0.5 * y[i];
      REQUIRE(relative_error(sf, scaled_ref) < 1e-5);

      // Explicit matrix elements on an arbitrary determinant subset
      std::vector<size_t> subset;
      for (size_t i = 0; i < ndet; i += 1 + i % 3) subset.push_back(i);
      std::reverse(subset.begin(), subset.end());
      const size_t nsub = subset.size();
      std::vector<double> H_sub(nsub * nsub);
      H.submatrix(subset, H_sub.data());
      for (size_t j = 0; j < nsub; ++j)
        for (size_t i = 0; i < nsub; ++i)
          REQUIRE_THAT(
              H_sub[i + j * nsub],
              Catch::Matchers::WithinAbs(
                  ham_gen.matrix_element(dets[subset[i]], dets[subset[j]]),
                  1e-11));
    }
  }

#ifdef _OPENMP
  // Thread counts are set here rather than taken from the environment, so
  // that both same-spin schedules (threads owning tiles, threads sharing a
  // tile's cliques when tiles are fewer than threads) are exercised
  SECTION("Thread counts") {
    struct RestoreThreads {
      int n = omp_get_max_threads();
      ~RestoreThreads() { omp_set_num_threads(n); }
    } restore;
    for (auto [norb, na, nb] :
         std::vector<std::array<size_t, 3>>{{13, 4, 1}, {13, 1, 4}}) {
      std::vector<double> T, V;
      random_integrals(norb, gen, T, V);
      macis::FCIHamiltonian H(norb, na, nb, T.data(), norb, V.data(), norb);
      macis::DoubleLoopHamiltonianGenerator<wfn_type> ham_gen(
          macis::matrix_span<double>(T.data(), norb, norb),
          macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
      auto dets = macis::generate_hilbert_space<wfn_type>(norb, na, nb);
      const size_t ndet = dets.size();
      auto H_ref = macis::make_csr_hamiltonian<int64_t>(
          dets.begin(), dets.end(), ham_gen, 0.0);

      std::vector<double> x(ndet), sigma(ndet), sigma_ref(ndet);
      for (auto& xi : x) xi = dist(gen);
      sparsexx::spblas::gespmbv(1, 1.0, H_ref, x.data(), ndet, 0.0,
                                sigma_ref.data(), ndet);
      for (int nthreads : {1, 2, 5}) {
        omp_set_num_threads(nthreads);
        for (int level = 0; level <= int(macis::fci_simd_supported());
             ++level) {
          H.set_simd(macis::FCISimd(level));
          H.sigma(x.data(), sigma.data());
          for (size_t i = 0; i < ndet; ++i)
            REQUIRE_THAT(sigma[i],
                         Catch::Matchers::WithinAbs(sigma_ref[i], 1e-10));
          std::vector<float> xf(x.begin(), x.end()), sf(ndet);
          H.sigma(xf.data(), sf.data());
          REQUIRE(relative_error(sf, sigma_ref) < 1e-5);
        }
      }
    }
  }
#endif

  // Concurrent calls on one Hamiltonian: only one can use its cached work
  // arrays, the others allocate their own
  SECTION("Concurrent calls") {
    const size_t norb = 8, na = 4, nb = 3;
    std::vector<double> T, V;
    random_integrals(norb, gen, T, V);
    const macis::FCIHamiltonian H(norb, na, nb, T.data(), norb, V.data(), norb);
    const size_t ndet = H.size();
    constexpr int ncallers = 4;
    std::vector<std::vector<double>> x(ncallers, std::vector<double>(ndet));
    std::vector<std::vector<double>> ref(ncallers, std::vector<double>(ndet));
    for (int c = 0; c < ncallers; ++c) {
      for (auto& xi : x[c]) xi = dist(gen);
      H.sigma(x[c].data(), ref[c].data());
    }
    std::vector<double> error(ncallers, 0.0), error_f(ncallers, 0.0);
    std::vector<std::thread> callers;
    for (int c = 0; c < ncallers; ++c)
      callers.emplace_back([&, c] {
        std::vector<double> s(ndet);
        std::vector<float> xf(x[c].begin(), x[c].end()), sf(ndet);
        for (int rep = 0; rep < 20; ++rep) {
          H.sigma(x[c].data(), s.data());
          H.sigma(xf.data(), sf.data());
          for (size_t i = 0; i < ndet; ++i)
            error[c] = std::max(error[c], std::abs(s[i] - ref[c][i]));
          error_f[c] = std::max(error_f[c], relative_error(sf, ref[c]));
        }
      });
    for (auto& t : callers) t.join();
    for (int c = 0; c < ncallers; ++c) {
      REQUIRE(error[c] < 1e-10);
      REQUIRE(error_f[c] < 1e-5);
    }
  }
}

TEST_CASE("FCI Spin Flip") {
  ROOT_ONLY(MPI_COMM_WORLD);

  std::mt19937 gen(97);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  // X + eta X^T is exactly symmetric (eta = 1) or antisymmetric (eta = -1)
  auto sector_vector = [&](size_t ns, int eta) {
    std::vector<double> X(ns * ns), C(ns * ns);
    for (auto& x : X) x = dist(gen);
    for (size_t i = 0; i < ns; ++i)
      for (size_t j = 0; j < ns; ++j)
        C[i * ns + j] = X[i * ns + j] + eta * X[j * ns + i];
    return C;
  };

  // (norb, nalpha = nbeta): no or all orbitals occupied, a single beta block
  // and several beta blocks of the alpha-beta pass
  const std::vector<std::pair<size_t, size_t>> cases = {
      {1, 0}, {1, 1}, {2, 1}, {3, 1},  {4, 2},
      {5, 2}, {6, 3}, {8, 4}, {10, 5}, {13, 4}};
  for (auto [norb, k] : cases) {
    for (int eta : {1, -1}) {
      DYNAMIC_SECTION("norb = " << norb << " nelec = " << k
                                << " eta = " << eta) {
        std::vector<double> T, V;
        random_integrals(norb, gen, T, V);
        macis::FCIHamiltonian H(norb, k, k, T.data(), norb, V.data(), norb);
        const size_t ns = H.alpha_strings().size();
        const size_t ndet = H.size();

        // Same result as the general contraction, exactly in the sector
        auto C = sector_vector(ns, eta);
        std::vector<double> ref(ndet), S(ndet);
        H.sigma(C.data(), ref.data());
        H.sigma(C.data(), S.data(), 1.0, 0.0, eta);
        for (size_t i = 0; i < ndet; ++i)
          REQUIRE_THAT(S[i], Catch::Matchers::WithinAbs(ref[i], 1e-10));
        for (size_t i = 0; i < ns; ++i)
          for (size_t j = 0; j < ns; ++j)
            REQUIRE(S[i * ns + j] == eta * S[j * ns + i]);

        // S = alpha H C + beta S for S in the sector
        const auto S0 = sector_vector(ns, eta);
        S = S0;
        H.sigma(C.data(), S.data(), 2.0, -0.5, eta);
        for (size_t i = 0; i < ndet; ++i)
          REQUIRE_THAT(S[i], Catch::Matchers::WithinAbs(
                                 2.0 * ref[i] - 0.5 * S0[i], 1e-10));

        // Single precision: float accuracy, still exactly in the sector
        std::vector<float> Cf(C.begin(), C.end()), Sf(ndet);
        H.sigma(Cf.data(), Sf.data(), 1.0f, 0.0f, eta);
        REQUIRE(relative_error(Sf, ref) < 1e-5);
        for (size_t i = 0; i < ns; ++i)
          for (size_t j = 0; j < ns; ++j)
            REQUIRE(Sf[i * ns + j] == float(eta) * Sf[j * ns + i]);
      }
    }
  }

  SECTION("Invalid arguments") {
    std::vector<double> T, V;
    random_integrals(4, gen, T, V);
    macis::FCIHamiltonian H(4, 2, 1, T.data(), 4, V.data(), 4);
    std::vector<double> C(H.size(), 0.0), S(H.size());
    REQUIRE_THROWS_AS(H.sigma(C.data(), S.data(), 1.0, 0.0, 1),
                      std::invalid_argument);
    macis::FCIHamiltonian H2(4, 2, 2, T.data(), 4, V.data(), 4);
    C.assign(H2.size(), 0.0);
    S.resize(H2.size());
    REQUIRE_THROWS_AS(H2.sigma(C.data(), S.data(), 1.0, 0.0, 2),
                      std::invalid_argument);
  }
}

TEST_CASE("FCI Ground State") {
  ROOT_ONLY(MPI_COMM_WORLD);
  for (auto name : {"ci_solver", "davidson"})
    if (!spdlog::get(name)) spdlog::null_logger_mt(name);

  // Leading 8x8 orbital block of the water cc-pVDZ integrals
  const size_t nfull = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  std::vector<double> T_full(nfull * nfull),
      V_full(nfull * nfull * nfull * nfull);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T_full.data(), nfull);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V_full.data(), nfull);

  const size_t norb = 8;
  std::vector<double> T(norb * norb), V(norb * norb * norb * norb);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q) {
      T[p + q * norb] = T_full[p + q * nfull];
      for (size_t r = 0; r < norb; ++r)
        for (size_t s = 0; s < norb; ++s)
          V[p + norb * (q + norb * (r + norb * s))] =
              V_full[p + nfull * (q + nfull * (r + nfull * s))];
    }

  const std::vector<std::pair<size_t, size_t>> cases = {{5, 5}, {5, 4}, {3, 2}};
  for (auto [na, nb] : cases) {
    DYNAMIC_SECTION("nalpha = " << na << " nbeta = " << nb) {
      macis::FCIHamiltonian H(norb, na, nb, T.data(), norb, V.data(), norb);
      const size_t ndet = H.size();

      macis::DoubleLoopHamiltonianGenerator<wfn_type> ham_gen(
          macis::matrix_span<double>(T.data(), norb, norb),
          macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
      auto dets = macis::generate_hilbert_space<wfn_type>(norb, na, nb);
      auto H_csr = macis::make_csr_hamiltonian<int64_t>(
          dets.begin(), dets.end(), ham_gen, 0.0);
      std::vector<double> H_dense(ndet * ndet), evals(ndet);
      sparsexx::convert_to_dense(H_csr, H_dense.data(), ndet);
      lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower, ndet,
                   H_dense.data(), ndet, evals.data());

      std::vector<double> C;
      const double E = macis::fci_ground_state(H, 200, 1e-8, C);
      REQUIRE_THAT(E, Catch::Matchers::WithinAbs(evals[0], 1e-10));
      REQUIRE(C.size() == ndet);
      REQUIRE_THAT(blas::nrm2(ndet, C.data(), 1),
                   Catch::Matchers::WithinAbs(1.0, 1e-12));

      // || H C - E C || is the converged residual
      std::vector<double> R(ndet);
      H.sigma(C.data(), R.data());
      blas::axpy(ndet, -E, C.data(), 1, R.data(), 1);
      REQUIRE(blas::nrm2(ndet, R.data(), 1) < 1e-8);

      // A converged vector is accepted as a warm start: one iteration suffices
      std::vector<double> C_warm = C;
      const double E_warm = macis::fci_ground_state(H, 1, 1e-8, C_warm);
      REQUIRE_THAT(E_warm, Catch::Matchers::WithinAbs(E, 1e-12));

      // Caller storage: without a guess its contents (here NaN, as
      // uninitialized memory may hold) are ignored; with one it is used
      std::vector<double> C_raw(ndet, std::numeric_limits<double>::quiet_NaN());
      REQUIRE_THAT(macis::fci_ground_state(H, 200, 1e-8, C_raw.data(), false),
                   Catch::Matchers::WithinAbs(E, 1e-10));
      REQUIRE_THAT(std::abs(blas::dot(ndet, C_raw.data(), 1, C.data(), 1)),
                   Catch::Matchers::WithinAbs(1.0, 1e-10));
      std::copy(C.begin(), C.end(), C_raw.begin());
      REQUIRE_THAT(macis::fci_ground_state(H, 1, 1e-8, C_raw.data(), true),
                   Catch::Matchers::WithinAbs(E, 1e-12));

      if (na != nb) {
        std::vector<double> C_flip;
        REQUIRE_THROWS_AS(macis::fci_ground_state(H, 200, 1e-8, C_flip, 1),
                          std::invalid_argument);
      } else {
        // Spin-flip sectors: C^T = C holds the states of even total spin S,
        // C^T = -C those of odd S. Reference: the MACIS matrix in an
        // orthonormal basis of each sector, (|I,J> + eta |J,I>)/sqrt(2) and
        // |I,I> (even sector only)
        const size_t ns = H.alpha_strings().size();
        std::fill(H_dense.begin(), H_dense.end(), 0.0);
        sparsexx::convert_to_dense(H_csr, H_dense.data(), ndet);
        std::vector<std::vector<double>> C_sector(2);
        double E_lowest = std::numeric_limits<double>::infinity();
        for (int eta : {1, -1}) {
          std::vector<std::array<size_t, 2>> reps;
          for (size_t i = 0; i < ns; ++i)
            for (size_t j = 0; j <= i; ++j)
              if (i != j or eta == 1) reps.push_back({i, j});
          const size_t m = reps.size();
          auto members = [&](size_t r) {
            const auto [i, j] = reps[r];
            std::vector<std::pair<size_t, double>> mem;
            if (i == j) {
              mem.emplace_back(i * ns + i, 1.0);
            } else {
              mem.emplace_back(i * ns + j, std::sqrt(0.5));
              mem.emplace_back(j * ns + i, eta * std::sqrt(0.5));
            }
            return mem;
          };
          std::vector<double> H_eta(m * m), w(m);
          for (size_t s = 0; s < m; ++s)
            for (size_t r = 0; r < m; ++r) {
              double h = 0.0;
              for (auto [a, ca] : members(r))
                for (auto [b, cb] : members(s))
                  h += ca * cb * H_dense[a + b * ndet];
              H_eta[r + s * m] = h;
            }
          lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower, m, H_eta.data(),
                       m, w.data());

          auto& C_eta = C_sector[eta == 1 ? 0 : 1];
          const double E_eta =
              macis::fci_ground_state(H, 200, 1e-8, C_eta, eta);
          REQUIRE_THAT(E_eta, Catch::Matchers::WithinAbs(w[0], 1e-10));
          REQUIRE_THAT(blas::nrm2(ndet, C_eta.data(), 1),
                       Catch::Matchers::WithinAbs(1.0, 1e-12));
          for (size_t i = 0; i < ns; ++i)
            for (size_t j = 0; j < ns; ++j)
              REQUIRE(C_eta[i * ns + j] == eta * C_eta[j * ns + i]);
          E_lowest = std::min(E_lowest, E_eta);

          // A converged sector vector is accepted as a warm start
          std::vector<double> C_warm_eta = C_eta;
          REQUIRE_THAT(macis::fci_ground_state(H, 1, 1e-8, C_warm_eta, eta),
                       Catch::Matchers::WithinAbs(E_eta, 1e-12));

          // Total spin from the opposite-spin 2-RDM (Ms = 0):
          // <S^2> = N_beta - sum_pq <E^b_qp E^a_pq> = N_beta - 2 sum_pq
          // aabb(q,p,p,q)
          std::vector<double> aabb(norb * norb * norb * norb);
          macis::fci_rdms_spin_dep(
              H.alpha_strings(), H.beta_strings(), C_eta.data(),
              macis::matrix_span<double>(nullptr, norb, norb),
              macis::matrix_span<double>(nullptr, norb, norb),
              macis::rank4_span<double>(nullptr, norb, norb, norb, norb),
              macis::rank4_span<double>(nullptr, norb, norb, norb, norb),
              macis::rank4_span<double>(aabb.data(), norb, norb, norb, norb));
          double s2 = nb;
          for (size_t p = 0; p < norb; ++p)
            for (size_t q = 0; q < norb; ++q)
              s2 -= 2.0 * aabb[q + norb * (p + norb * (p + norb * q))];
          const double S = 0.5 * (std::sqrt(1.0 + 4.0 * s2) - 1.0);
          REQUIRE_THAT(S, Catch::Matchers::WithinAbs(std::round(S), 1e-6));
          REQUIRE(std::lround(S) % 2 == (eta == 1 ? 0 : 1));

          // Dense diagonalization of the whole sector
          std::vector<size_t> all(ndet);
          std::iota(all.begin(), all.end(), size_t{0});
          std::vector<double> C_dense;
          REQUIRE_THAT(macis::fci_subspace_ground_state(H, all, C_dense, eta),
                       Catch::Matchers::WithinAbs(w[0], 1e-10));

          // A determinant subset is closed under I <-> J; the result is a
          // normalized sector vector whose Rayleigh quotient is the returned
          // (variational) energy
          std::vector<size_t> subset;
          for (size_t d = 0; d < ndet; d += 7) subset.push_back(d);
          std::vector<double> C_sub;
          const double E_sub =
              macis::fci_subspace_ground_state(H, subset, C_sub, eta);
          REQUIRE(E_sub >= w[0] - 1e-10);
          REQUIRE_THAT(blas::nrm2(ndet, C_sub.data(), 1),
                       Catch::Matchers::WithinAbs(1.0, 1e-12));
          for (size_t i = 0; i < ns; ++i)
            for (size_t j = 0; j < ns; ++j)
              REQUIRE(C_sub[i * ns + j] == eta * C_sub[j * ns + i]);
          std::vector<double> HC(ndet);
          H.sigma(C_sub.data(), HC.data());
          REQUIRE_THAT(blas::dot(ndet, C_sub.data(), 1, HC.data(), 1),
                       Catch::Matchers::WithinAbs(E_sub, 1e-10));
        }
        REQUIRE_THAT(E_lowest, Catch::Matchers::WithinAbs(E, 1e-10));

        // A guess from the other sector projects to zero and is replaced
        std::vector<double> C_swap = C_sector[1];
        std::vector<double> C_even = C_sector[0];
        REQUIRE_THAT(
            macis::fci_ground_state(H, 200, 1e-8, C_swap, 1),
            Catch::Matchers::WithinAbs(
                macis::fci_ground_state(H, 200, 1e-8, C_even, 1), 1e-10));
      }
    }
  }

  SECTION("Empty odd sector") {
    // A single string per spin admits no antisymmetric CI matrix
    macis::FCIHamiltonian H(2, 2, 2, T.data(), norb, V.data(), norb);
    std::vector<double> C;
    REQUIRE_THROWS_AS(macis::fci_ground_state(H, 10, 1e-8, C, -1),
                      std::invalid_argument);
  }

#ifdef _OPENMP
  SECTION("Caller's thread setting") {
    // The solve sizes its OpenMP team (one thread for this small space) and
    // restores the caller's setting, also when it throws
    struct RestoreThreads {
      int n = omp_get_max_threads();
      ~RestoreThreads() { omp_set_num_threads(n); }
    } restore;
    omp_set_num_threads(3);
    macis::FCIHamiltonian H(7, 3, 3, T.data(), norb, V.data(), norb);
    std::vector<double> C;
    macis::fci_ground_state(H, 200, 1e-8, C);
    REQUIRE(omp_get_max_threads() == 3);
    C.clear();
    REQUIRE_THROWS(macis::fci_ground_state(H, 1, 1e-12, C));
    REQUIRE(omp_get_max_threads() == 3);
  }
#endif

  SECTION("P-space guess") {
    // With at most 400 determinants the P-space of the guess is the whole
    // space: the guess is the ground state and one iteration converges
    for (int spin_flip : {0, 1, -1}) {
      macis::FCIHamiltonian H(6, 3, 3, T.data(), norb, V.data(), norb);
      REQUIRE(H.size() == 400);
      std::vector<size_t> all(H.size());
      std::iota(all.begin(), all.end(), size_t{0});
      std::vector<double> C, C_dense;
      REQUIRE_THAT(macis::fci_ground_state(H, 1, 1e-6, C, spin_flip),
                   Catch::Matchers::WithinAbs(macis::fci_subspace_ground_state(
                                                  H, all, C_dense, spin_flip),
                                              1e-10));
    }
  }

  SECTION("Spin-flip parity of the guess") {
    // Orbital windows (first orbital, orbitals, electrons per spin) of the
    // water integrals whose lowest P-space vector over 400 determinants had
    // the other spin-flip parity than the ground state; as the Davidson
    // iteration keeps that parity, the solve converged to an excited state
    // (0.006 to 0.038 Eh too high)
    const std::vector<std::array<size_t, 3>> windows = {
        {4, 7, 4}, {5, 9, 2}, {5, 10, 2}, {12, 9, 2}, {4, 10, 2}};
    for (auto [first, n, k] : windows) {
      std::vector<double> Tw(n * n), Vw(n * n * n * n);
      for (size_t p = 0; p < n; ++p)
        for (size_t q = 0; q < n; ++q) {
          Tw[p + q * n] = T_full[first + p + (first + q) * nfull];
          for (size_t r = 0; r < n; ++r)
            for (size_t s = 0; s < n; ++s)
              Vw[p + n * (q + n * (r + n * s))] =
                  V_full[first + p +
                         nfull * (first + q +
                                  nfull * (first + r + nfull * (first + s)))];
        }
      macis::FCIHamiltonian H(n, k, k, Tw.data(), n, Vw.data(), n);
      std::vector<size_t> all(H.size());
      std::iota(all.begin(), all.end(), size_t{0});
      std::vector<double> C, C_dense;
      const double E_dense = macis::fci_subspace_ground_state(H, all, C_dense);
      CHECK_THAT(macis::fci_ground_state(H, 200, 1e-8, C),
                 Catch::Matchers::WithinAbs(E_dense, 1e-10));
    }
  }
}

TEST_CASE("FCI Mixed Precision") {
  ROOT_ONLY(MPI_COMM_WORLD);
  for (auto name : {"ci_solver", "davidson"})
    if (!spdlog::get(name)) spdlog::null_logger_mt(name);

  // Leading 11x11 orbital block of the water cc-pVDZ integrals (mixed
  // precision forced: the automatic choice keeps spaces this small in double
  // precision)
  const size_t nfull = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  std::vector<double> T_full(nfull * nfull),
      V_full(nfull * nfull * nfull * nfull);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T_full.data(), nfull);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V_full.data(), nfull);
  const size_t norb = 11;
  std::vector<double> T(norb * norb), V(norb * norb * norb * norb);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q) {
      T[p + q * norb] = T_full[p + q * nfull];
      for (size_t r = 0; r < norb; ++r)
        for (size_t s = 0; s < norb; ++s)
          V[p + norb * (q + norb * (r + norb * s))] =
              V_full[p + nfull * (q + nfull * (r + nfull * s))];
    }

  // (nalpha, nbeta, spin_flip, residual tolerance); the tight tolerance is
  // below what one single-precision restart resolves
  const std::vector<std::tuple<size_t, size_t, int, double>> cases = {
      {4, 4, 0, 1e-6},
      {4, 4, 0, 1e-10},
      {5, 4, 0, 1e-6},
      {4, 4, 1, 1e-6},
      {4, 4, -1, 1e-6}};
  for (auto [na, nb, spin_flip, tol] : cases) {
    DYNAMIC_SECTION("nalpha = " << na << " nbeta = " << nb << " spin_flip = "
                                << spin_flip << " tol = " << tol) {
      macis::FCIHamiltonian H(norb, na, nb, T.data(), norb, V.data(), norb);
      const size_t ndet = H.size();

      std::vector<double> C_ref, C;
      const double E_ref =
          macis::fci_ground_state(H, 500, tol, C_ref, spin_flip, 3,
                                  macis::FCIPrecision::double_precision);
      // Caller storage holding NaN (ignored without a guess)
      C.assign(ndet, std::numeric_limits<double>::quiet_NaN());
      const double E =
          macis::fci_ground_state(H, 500, tol, C.data(), false, spin_flip, 3,
                                  macis::FCIPrecision::mixed);

      // Both residuals are below tol, so the eigenvalues agree to ~tol^2
      REQUIRE_THAT(E, Catch::Matchers::WithinAbs(
                          E_ref, std::max(1e-11, 100.0 * tol * tol)));
      REQUIRE_THAT(blas::nrm2(ndet, C.data(), 1),
                   Catch::Matchers::WithinAbs(1.0, 1e-12));
      std::vector<double> R(ndet);
      H.sigma(C.data(), R.data());
      REQUIRE_THAT(blas::dot(ndet, C.data(), 1, R.data(), 1),
                   Catch::Matchers::WithinAbs(E, 1e-10));
      blas::axpy(ndet, -E, C.data(), 1, R.data(), 1);
      REQUIRE(blas::nrm2(ndet, R.data(), 1) < tol);

      if (spin_flip != 0) {
        const size_t ns = H.alpha_strings().size();
        for (size_t i = 0; i < ns; ++i)
          for (size_t j = 0; j < ns; ++j)
            REQUIRE(C[i * ns + j] == spin_flip * C[j * ns + i]);
      }
    }
  }
}

TEST_CASE("FCI Single-Precision Davidson Storage") {
  ROOT_ONLY(MPI_COMM_WORLD);
  for (auto name : {"ci_solver", "davidson"})
    if (!spdlog::get(name)) spdlog::null_logger_mt(name);

  // restarted_davidson<float> rounds every stored vector to float, so its
  // basis is orthonormal only to about 1e-8. With images exact up to that
  // same rounding (double-precision sigma stored as float), the residual
  // must still converge far below |E| 1e-8: a diagonal shift of -1000 Eh,
  // which changes nothing in exact arithmetic, put the floor of a
  // Rayleigh-Ritz step that took the stored basis as orthonormal above 1e-6.
  const size_t nfull = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  std::vector<double> T_full(nfull * nfull),
      V_full(nfull * nfull * nfull * nfull);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T_full.data(), nfull);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V_full.data(), nfull);
  const size_t norb = 11;
  std::vector<double> T(norb * norb), V(norb * norb * norb * norb);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q) {
      T[p + q * norb] = T_full[p + q * nfull];
      for (size_t r = 0; r < norb; ++r)
        for (size_t s = 0; s < norb; ++s)
          V[p + norb * (q + norb * (r + norb * s))] =
              V_full[p + nfull * (q + nfull * (r + nfull * s))];
    }
  macis::FCIHamiltonian H(norb, 4, 4, T.data(), norb, V.data(), norb);
  const size_t ndet = H.size();
  const double shift = -1000.0;

  struct Shifted {
    const macis::FCIHamiltonian& H;
    double shift;
    void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                         double beta, double* AV, size_t LDAV) const {
      for (size_t j = 0; j < m; ++j) {
        H.sigma(V + j * LDV, AV + j * LDAV, alpha, beta);
        for (size_t i = 0; i < H.size(); ++i)
          AV[i + j * LDAV] += alpha * shift * V[i + j * LDV];
      }
    }
  } exact{H, shift};
  struct Rounded {
    const Shifted& exact;
    void operator_action(size_t m, double alpha, const float* V, size_t LDV,
                         double beta, float* AV, size_t LDAV) const {
      const size_t n = exact.H.size();
      std::vector<double> v(n), av(n);
      for (size_t j = 0; j < m; ++j) {
        std::copy_n(V + j * LDV, n, v.begin());
        exact.operator_action(1, alpha, v.data(), n, 0.0, av.data(), n);
        for (size_t i = 0; i < n; ++i)
          AV[i + j * LDAV] =
              float(beta == 0.0 ? av[i] : av[i] + beta * AV[i + j * LDAV]);
      }
    }
  } rounded{exact};

  std::vector<double> D(ndet), X(ndet, 0.0);
  H.diagonal(D.data());
  for (auto& d : D) d += shift;
  X[std::min_element(D.begin(), D.end()) - D.begin()] = 1.0;

  const double tol = 1e-8;
  double res = 0.0;
  auto [niter, E] = macis::restarted_davidson<float>(
      ndet, 3, 100, rounded, D.data(), tol, X.data(), 1e-12, false, &res, 0,
      &exact, 1e-3);
  REQUIRE(res < tol);

  std::vector<double> C_ref;
  const double E_ref = macis::fci_ground_state(
      H, 500, 1e-9, C_ref, 0, 3, macis::FCIPrecision::double_precision);
  REQUIRE_THAT(E, Catch::Matchers::WithinAbs(E_ref + shift, 1e-8));
  // The returned vector in double precision: its Rayleigh quotient is
  // accurate to second order in its residual
  std::vector<double> R(ndet);
  exact.operator_action(1, 1.0, X.data(), ndet, 0.0, R.data(), ndet);
  const double rq = blas::dot(ndet, X.data(), 1, R.data(), 1);
  REQUIRE_THAT(rq, Catch::Matchers::WithinAbs(E_ref + shift, 1e-10));
  blas::axpy(ndet, -rq, X.data(), 1, R.data(), 1);
  REQUIRE(blas::nrm2(ndet, R.data(), 1) < 1e-7);
}

TEST_CASE("FCI RDMs") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const std::vector<std::array<size_t, 3>> cases = {
      {1, 1, 1}, {2, 1, 1}, {3, 2, 1}, {4, 2, 2}, {4, 0, 2},
      {4, 3, 0}, {5, 2, 3}, {6, 3, 3}, {6, 4, 2}, {7, 3, 4}};

  std::mt19937 gen(4321);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto [norb, na, nb] : cases) {
    DYNAMIC_SECTION("norb = " << norb << " nalpha = " << na
                              << " nbeta = " << nb) {
      macis::FCIStringSpace alpha(norb, na), beta(norb, nb);
      const size_t ndet = alpha.size() * beta.size();
      std::vector<double> C(ndet);
      for (auto& c : C) c = dist(gen);
      blas::scal(ndet, 1.0 / blas::nrm2(ndet, C.data(), 1), C.data(), 1);

      // The integrals do not enter the RDMs
      const size_t n2 = norb * norb, n4 = n2 * n2;
      std::vector<double> T(n2, 0.0), V(n4, 0.0);
      macis::DoubleLoopHamiltonianGenerator<wfn_type> ham_gen(
          macis::matrix_span<double>(T.data(), norb, norb),
          macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
      auto dets = macis::generate_hilbert_space<wfn_type>(norb, na, nb);

      auto m = [&](std::vector<double>& x) {
        return macis::matrix_span<double>(x.data(), norb, norb);
      };
      auto r4 = [&](std::vector<double>& x) {
        return macis::rank4_span<double>(x.data(), norb, norb, norb, norb);
      };
      std::vector<double> ref_aa(n2), ref_bb(n2), ref_aaaa(n4), ref_bbbb(n4),
          ref_aabb(n4);
      ham_gen.form_rdms_spin_dep(dets.begin(), dets.end(), dets.begin(),
                                 dets.end(), C.data(), m(ref_aa), m(ref_bb),
                                 r4(ref_aaaa), r4(ref_bbbb), r4(ref_aabb));

      std::vector<double> aa(n2), bb(n2), aaaa(n4), bbbb(n4), aabb(n4);
      macis::fci_rdms_spin_dep(alpha, beta, C.data(), m(aa), m(bb), r4(aaaa),
                               r4(bbbb), r4(aabb));
      for (size_t i = 0; i < n2; ++i) {
        REQUIRE_THAT(aa[i], Catch::Matchers::WithinAbs(ref_aa[i], 1e-12));
        REQUIRE_THAT(bb[i], Catch::Matchers::WithinAbs(ref_bb[i], 1e-12));
      }
      for (size_t i = 0; i < n4; ++i) {
        REQUIRE_THAT(aaaa[i], Catch::Matchers::WithinAbs(ref_aaaa[i], 1e-12));
        REQUIRE_THAT(bbbb[i], Catch::Matchers::WithinAbs(ref_bbbb[i], 1e-12));
        REQUIRE_THAT(aabb[i], Catch::Matchers::WithinAbs(ref_aabb[i], 1e-12));
      }

      // Null spans are skipped
      std::vector<double> aa_only(n2), aabb_only(n4);
      macis::fci_rdms_spin_dep(
          alpha, beta, C.data(), m(aa_only),
          macis::matrix_span<double>(nullptr, norb, norb),
          macis::rank4_span<double>(nullptr, norb, norb, norb, norb),
          macis::rank4_span<double>(nullptr, norb, norb, norb, norb),
          r4(aabb_only));
      for (size_t i = 0; i < n2; ++i)
        REQUIRE_THAT(aa_only[i], Catch::Matchers::WithinAbs(aa[i], 1e-14));
      for (size_t i = 0; i < n4; ++i)
        REQUIRE_THAT(aabb_only[i], Catch::Matchers::WithinAbs(aabb[i], 1e-14));

      // Vectors in a spin-flip sector have equal alpha and beta RDMs, which
      // are then copied rather than computed twice
      if (na == nb) {
        const size_t ns = alpha.size();
        for (int eta : {1, -1}) {
          std::vector<double> X(ndet), Cf(ndet);
          for (auto& x : X) x = dist(gen);
          for (size_t i = 0; i < ns; ++i)
            for (size_t j = 0; j < ns; ++j)
              Cf[i * ns + j] = X[i * ns + j] + eta * X[j * ns + i];
          const double nrm = blas::nrm2(ndet, Cf.data(), 1);
          if (nrm > 0.0) blas::scal(ndet, 1.0 / nrm, Cf.data(), 1);

          std::vector<double> g_aa(n2), g_bb(n2), g_aaaa(n4), g_bbbb(n4),
              g_aabb(n4);
          macis::fci_rdms_spin_dep(alpha, beta, Cf.data(), m(g_aa), m(g_bb),
                                   r4(g_aaaa), r4(g_bbbb), r4(g_aabb));
          std::vector<double> f_aa(n2), f_bb(n2), f_aaaa(n4), f_bbbb(n4),
              f_aabb(n4);
          macis::fci_rdms_spin_dep(alpha, beta, Cf.data(), m(f_aa), m(f_bb),
                                   r4(f_aaaa), r4(f_bbbb), r4(f_aabb), eta);
          for (size_t i = 0; i < n2; ++i) {
            REQUIRE_THAT(f_aa[i], Catch::Matchers::WithinAbs(g_aa[i], 1e-12));
            REQUIRE_THAT(f_bb[i], Catch::Matchers::WithinAbs(g_bb[i], 1e-12));
          }
          for (size_t i = 0; i < n4; ++i) {
            REQUIRE_THAT(f_aaaa[i],
                         Catch::Matchers::WithinAbs(g_aaaa[i], 1e-12));
            REQUIRE_THAT(f_bbbb[i],
                         Catch::Matchers::WithinAbs(g_bbbb[i], 1e-12));
            REQUIRE_THAT(f_aabb[i],
                         Catch::Matchers::WithinAbs(g_aabb[i], 1e-12));
          }

          std::vector<double> bb_only(n2);
          macis::fci_rdms_spin_dep(
              alpha, beta, Cf.data(),
              macis::matrix_span<double>(nullptr, norb, norb), m(bb_only),
              macis::rank4_span<double>(nullptr, norb, norb, norb, norb),
              macis::rank4_span<double>(nullptr, norb, norb, norb, norb),
              macis::rank4_span<double>(nullptr, norb, norb, norb, norb), eta);
          for (size_t i = 0; i < n2; ++i)
            REQUIRE_THAT(bb_only[i],
                         Catch::Matchers::WithinAbs(g_bb[i], 1e-12));
        }
      }
    }
  }
}
