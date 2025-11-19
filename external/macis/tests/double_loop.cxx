/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <blas.hh>
#include <catch2/catch_template_test_macros.hpp>
#include <iomanip>
#include <iostream>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/wavefunction_io.hpp>

#include "ut_common.hpp"

using wfn_type = macis::wfn_t<64>;
TEMPLATE_TEST_CASE("Restricted Double Loop", "[ham-gen]",
                   macis::DoubleLoopHamiltonianGenerator<wfn_type>,
                   macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>) {
  ROOT_ONLY(MPI_COMM_WORLD);

  auto norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  const auto norb2 = norb * norb;
  const auto norb3 = norb2 * norb;
  const size_t num_occupied_orbitals = 5;

  std::vector<double> T(norb * norb);
  std::vector<double> V(norb * norb * norb * norb);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_type = TestType;

  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  const auto hf_det = wfn_traits::canonical_hf_determinant(
      num_occupied_orbitals, num_occupied_orbitals);

  std::vector<double> eps(norb);
  for (auto p = 0ul; p < norb; ++p) {
    double tmp = 0.;
    for (auto i = 0ul; i < num_occupied_orbitals; ++i) {
      tmp += 2. * V[p * (norb + 1) + i * (norb2 + norb3)] -
             V[p * (1 + norb3) + i * (norb + norb2)];
    }
    eps[p] = T[p * (norb + 1)] + tmp;
  }
  const auto EHF = ham_gen.matrix_element(hf_det, hf_det);

  SECTION("HF Energy") {
    REQUIRE_THAT(EHF + E_core,
                 Catch::Matchers::WithinAbs(-76.0267803489191,
                                            testing::ascii_text_tolerance));
  }

  SECTION("Excited Diagonals") {
    auto state = hf_det;
    std::vector<uint32_t> occ = {0, 1, 2, 3, 4};

    SECTION("Singles") {
      state.flip(0).flip(num_occupied_orbitals);
      const auto ES = ham_gen.matrix_element(state, state);
      REQUIRE_THAT(ES, Catch::Matchers::WithinAbs(
                           -6.488097259228e+01, testing::ascii_text_tolerance));

      auto fast_ES =
          ham_gen.fast_diag_single(occ, occ, 0, num_occupied_orbitals, EHF);
      REQUIRE_THAT(ES, Catch::Matchers::WithinAbs(
                           fast_ES, testing::numerical_zero_tolerance));
    }

    SECTION("Doubles - Same Spin") {
      state.flip(0)
          .flip(num_occupied_orbitals)
          .flip(1)
          .flip(num_occupied_orbitals + 1);
      const auto ED = ham_gen.matrix_element(state, state);
      REQUIRE_THAT(ED, Catch::Matchers::WithinAbs(
                           -6.314093508151e+01, testing::ascii_text_tolerance));

      auto fast_ED =
          ham_gen.fast_diag_ss_double(occ, occ, 0, 1, num_occupied_orbitals,
                                      num_occupied_orbitals + 1, EHF);
      REQUIRE_THAT(ED, Catch::Matchers::WithinAbs(
                           fast_ED, testing::numerical_zero_tolerance));
    }

    SECTION("Doubles - Opposite Spin") {
      state.flip(0)
          .flip(num_occupied_orbitals)
          .flip(1 + 32)
          .flip(num_occupied_orbitals + 1 + 32);
      const auto ED = ham_gen.matrix_element(state, state);
      REQUIRE_THAT(ED, Catch::Matchers::WithinAbs(
                           -6.304547887231e+01, testing::ascii_text_tolerance));

      auto fast_ED =
          ham_gen.fast_diag_os_double(occ, occ, 0, 1, num_occupied_orbitals,
                                      num_occupied_orbitals + 1, EHF);
      REQUIRE_THAT(ED, Catch::Matchers::WithinAbs(
                           fast_ED, testing::numerical_zero_tolerance));
    }
  }

  SECTION("Brilloin") {
    // Alpha -> Alpha
    for (size_t i = 0; i < num_occupied_orbitals; ++i)
      for (size_t a = num_occupied_orbitals; a < norb; ++a) {
        // Generate excited determinant
        wfn_type state = hf_det;
        state.flip(i).flip(a);
        auto el_1 = ham_gen.matrix_element(hf_det, state);
        auto el_2 = ham_gen.matrix_element(state, hf_det);
        REQUIRE(std::abs(el_1) < testing::ascii_text_tolerance);
        REQUIRE_THAT(el_1, Catch::Matchers::WithinAbs(
                               el_2, testing::numerical_zero_tolerance));
      }

    // Beta -> Beta
    for (size_t i = 0; i < num_occupied_orbitals; ++i)
      for (size_t a = num_occupied_orbitals; a < norb; ++a) {
        // Generate excited determinant
        wfn_type state = hf_det;
        state.flip(i + 32).flip(a + 32);
        auto el_1 = ham_gen.matrix_element(hf_det, state);
        auto el_2 = ham_gen.matrix_element(state, hf_det);
        REQUIRE(std::abs(el_1) < testing::ascii_text_tolerance);
        REQUIRE_THAT(el_1, Catch::Matchers::WithinAbs(
                               el_2, testing::numerical_zero_tolerance));
      }
  }

  SECTION("MP2") {
    double EMP2 = 0.;
    for (size_t a = num_occupied_orbitals; a < norb; ++a)
      for (size_t b = a + 1; b < norb; ++b)
        for (size_t i = 0; i < num_occupied_orbitals; ++i)
          for (size_t j = i + 1; j < num_occupied_orbitals; ++j) {
            auto state = hf_det;
            state.flip(i).flip(j).flip(a).flip(b);
            auto h_el = ham_gen.matrix_element(hf_det, state);
            double diag = eps[a] + eps[b] - eps[i] - eps[j];

            EMP2 += (h_el * h_el) / diag;

            REQUIRE_THAT(ham_gen.matrix_element(state, hf_det),
                         Catch::Matchers::WithinAbs(
                             h_el, testing::ascii_text_tolerance));
          }

    for (size_t a = num_occupied_orbitals; a < norb; ++a)
      for (size_t b = a + 1; b < norb; ++b)
        for (size_t i = 0; i < num_occupied_orbitals; ++i)
          for (size_t j = i + 1; j < num_occupied_orbitals; ++j) {
            auto state = hf_det;
            state.flip(i + 32).flip(j + 32).flip(a + 32).flip(b + 32);
            auto h_el = ham_gen.matrix_element(hf_det, state);
            double diag = eps[a] + eps[b] - eps[i] - eps[j];

            EMP2 += (h_el * h_el) / diag;

            REQUIRE_THAT(ham_gen.matrix_element(state, hf_det),
                         Catch::Matchers::WithinAbs(
                             h_el, testing::ascii_text_tolerance));
          }

    for (size_t a = num_occupied_orbitals; a < norb; ++a)
      for (size_t b = num_occupied_orbitals; b < norb; ++b)
        for (size_t i = 0; i < num_occupied_orbitals; ++i)
          for (size_t j = 0; j < num_occupied_orbitals; ++j) {
            auto state = hf_det;
            state.flip(i).flip(j + 32).flip(a).flip(b + 32);
            auto h_el = ham_gen.matrix_element(hf_det, state);
            double diag = eps[a] + eps[b] - eps[i] - eps[j];

            EMP2 += (h_el * h_el) / diag;

            REQUIRE_THAT(ham_gen.matrix_element(state, hf_det),
                         Catch::Matchers::WithinAbs(
                             h_el, testing::ascii_text_tolerance));
          }

    REQUIRE_THAT((-EMP2),
                 Catch::Matchers::WithinAbs(-0.203989305096243,
                                            testing::ascii_text_tolerance));
  }

  SECTION("RDM - SPIN TRACED") {
    std::vector<double> ordm(norb * norb, 0.0), trdm(norb3 * norb, 0.0);
    std::vector<wfn_type> dets = {wfn_traits::canonical_hf_determinant(
        num_occupied_orbitals, num_occupied_orbitals)};

    std::vector<double> C = {1.};

    ham_gen.form_rdms(
        dets.begin(), dets.end(), dets.begin(), dets.end(), C.data(),
        macis::matrix_span<double>(ordm.data(), norb, norb),
        macis::rank4_span<double>(trdm.data(), norb, norb, norb, norb));

    auto E_tmp = blas::dot(norb2, ordm.data(), 1, T.data(), 1) +
                 blas::dot(norb3 * norb, trdm.data(), 1, V.data(), 1);
    REQUIRE_THAT(
        E_tmp, Catch::Matchers::WithinAbs(EHF, testing::ascii_text_tolerance));
  }

  SECTION("RDM - SPIN SEPARATED") {
    std::vector<double> ordm_aa(norb * norb, 0.0), ordm_bb(norb * norb, 0.0),
        trdm_aaaa(norb3 * norb, 0.0), trdm_bbbb(norb3 * norb, 0.0),
        trdm_aabb(norb3 * norb, 0.0);
    std::vector<wfn_type> dets = {wfn_traits::canonical_hf_determinant(
        num_occupied_orbitals, num_occupied_orbitals)};

    std::vector<double> C = {1.};

    ham_gen.form_rdms_spin_dep(
        dets.begin(), dets.end(), dets.begin(), dets.end(), C.data(),
        macis::matrix_span<double>(ordm_aa.data(), norb, norb),
        macis::matrix_span<double>(ordm_bb.data(), norb, norb),
        macis::rank4_span<double>(trdm_aaaa.data(), norb, norb, norb, norb),
        macis::rank4_span<double>(trdm_bbbb.data(), norb, norb, norb, norb),
        macis::rank4_span<double>(trdm_aabb.data(), norb, norb, norb, norb));

    auto E_tmp = blas::dot(norb2, ordm_aa.data(), 1, T.data(), 1) +
                 blas::dot(norb2, ordm_bb.data(), 1, T.data(), 1) +
                 blas::dot(norb3 * norb, trdm_aaaa.data(), 1, V.data(), 1) +
                 blas::dot(norb3 * norb, trdm_bbbb.data(), 1, V.data(), 1) +
                 2 * blas::dot(norb3 * norb, trdm_aabb.data(), 1, V.data(), 1);
    REQUIRE_THAT(
        E_tmp, Catch::Matchers::WithinAbs(EHF, testing::ascii_text_tolerance));
  }
}

using wfn_128_type = macis::wfn_t<128>;
TEMPLATE_TEST_CASE("Restricted RDMS", "[ham-gen]",
                   macis::DoubleLoopHamiltonianGenerator<wfn_128_type>,
                   macis::SortedDoubleLoopHamiltonianGenerator<wfn_128_type>) {
  ROOT_ONLY(MPI_COMM_WORLD);

  auto norb = 34;
  const auto norb2 = norb * norb;
  const auto norb3 = norb2 * norb;
  const size_t num_occupied_orbitals = 5;

  std::vector<double> T(norb * norb, 0.0);
  std::vector<double> V(norb3 * norb, 0.0);
  std::vector<double> ordm(norb * norb, 0.0), trdm(norb3 * norb, 0.0);

  macis::matrix_span<double> T_span(T.data(), norb, norb);
  macis::matrix_span<double> ordm_span(ordm.data(), norb, norb);
  macis::rank4_span<double> V_span(V.data(), norb, norb, norb, norb);
  macis::rank4_span<double> trdm_span(trdm.data(), norb, norb, norb, norb);

  using wfn_traits = macis::wavefunction_traits<wfn_128_type>;
  using generator_type = TestType;
  generator_type ham_gen(T_span, V_span);

  auto abs_sum = [](auto a, auto b) { return a + std::abs(b); };

  SECTION("HF") {
    std::vector<wfn_128_type> dets = {wfn_traits::canonical_hf_determinant(
        num_occupied_orbitals, num_occupied_orbitals)};

    std::vector<double> C = {1.};

    ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                      C.data(), ordm_span, trdm_span);

    for (auto i = 0ul; i < num_occupied_orbitals; ++i)
      for (auto j = 0ul; j < num_occupied_orbitals; ++j)
        for (auto k = 0ul; k < num_occupied_orbitals; ++k)
          for (auto l = 0ul; l < num_occupied_orbitals; ++l) {
            // ii jj
            if (i == j && k == l && i != k) {
              REQUIRE(trdm_span(i, j, k, l) - 2.0 <
                      testing::numerical_zero_tolerance);
            }
            // ijji
            else if (i == l && j == k && i != j) {
              REQUIRE(trdm_span(i, j, k, l) + 1.0 <
                      testing::numerical_zero_tolerance);
            }
            // iiii
            else if (i == j && k == l && i == k) {
              REQUIRE(trdm_span(i, j, k, l) - 1.0 <
                      testing::numerical_zero_tolerance);
            }
          }

    for (auto i = 0ul; i < num_occupied_orbitals; ++i) {
      REQUIRE(ordm_span(i, i) - 2.0 < testing::numerical_zero_tolerance);
    }
  }

  SECTION("CI") {
    std::vector<wfn_128_type> states;
    std::vector<double> coeffs;
    macis::read_wavefunction<128>(ch4_wfn_fname, states, coeffs);

    coeffs.resize(5000);
    states.resize(5000);

    // Renormalize C for trace computation
    auto c_nrm = blas::nrm2(coeffs.size(), coeffs.data(), 1);
    blas::scal(coeffs.size(), 1. / c_nrm, coeffs.data(), 1);

    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(), ordm_span, trdm_span);
    auto sum_ordm = std::accumulate(ordm.begin(), ordm.end(), 0.0, abs_sum);
    auto sum_trdm = std::accumulate(trdm.begin(), trdm.end(), 0.0, abs_sum);
    REQUIRE_THAT(sum_ordm,
                 Catch::Matchers::WithinAbs(1.038559618650e+01,
                                            testing::ascii_text_tolerance));
    REQUIRE_THAT(sum_trdm, Catch::Matchers::WithinAbs(
                               99.2388204965, testing::ascii_text_tolerance));

    double trace_ordm = 0.;
    for (auto p = 0; p < norb; ++p) trace_ordm += ordm_span(p, p);
    REQUIRE_THAT(trace_ordm,
                 Catch::Matchers::WithinAbs(2.0 * num_occupied_orbitals,
                                            testing::ascii_text_tolerance));

    // Check symmetries
    for (auto p = 0; p < norb; ++p)
      for (auto q = p; q < norb; ++q) {
        REQUIRE_THAT(ordm_span(p, q),
                     Catch::Matchers::WithinAbs(ordm_span(q, p),
                                                testing::ascii_text_tolerance));
      }

    // check spin dependent RDMs
    std::vector<double> ordm_aa(norb * norb, 0.0);
    std::vector<double> ordm_bb(norb * norb, 0.0);
    std::vector<double> trdm_aaaa(norb3 * norb, 0.0);
    std::vector<double> trdm_bbbb(norb3 * norb, 0.0);
    std::vector<double> trdm_aabb(norb3 * norb, 0.0);

    macis::matrix_span<double> ordm_aa_span(ordm_aa.data(), norb, norb);
    macis::matrix_span<double> ordm_bb_span(ordm_bb.data(), norb, norb);
    macis::rank4_span<double> trdm_aaaa_span(trdm_aaaa.data(), norb, norb, norb,
                                             norb);
    macis::rank4_span<double> trdm_bbbb_span(trdm_bbbb.data(), norb, norb, norb,
                                             norb);
    macis::rank4_span<double> trdm_aabb_span(trdm_aabb.data(), norb, norb, norb,
                                             norb);

    ham_gen.form_rdms_spin_dep(states.begin(), states.end(), states.begin(),
                               states.end(), coeffs.data(), ordm_aa_span,
                               ordm_bb_span, trdm_aaaa_span, trdm_bbbb_span,
                               trdm_aabb_span);
    // check 1rdm
    for (auto p = 0; p < norb; ++p)
      for (auto q = 0; q < norb; ++q) {
        REQUIRE_THAT(
            ordm_span(p, q),
            Catch::Matchers::WithinAbs(ordm_aa_span(p, q) + ordm_bb_span(p, q),
                                       testing::numerical_zero_tolerance));
      }

    // check 2rdm
    for (auto p = 0; p < norb; ++p)
      for (auto q = 0; q < norb; ++q)
        for (auto r = 0; r < norb; ++r)
          for (auto s = 0; s < norb; ++s) {
            REQUIRE_THAT(
                trdm_span(p, q, r, s),
                Catch::Matchers::WithinAbs(
                    trdm_aaaa_span(p, q, r, s) + trdm_bbbb_span(p, q, r, s) +
                        trdm_aabb_span(p, q, r, s) + trdm_aabb_span(r, s, p, q),
                    testing::numerical_zero_tolerance));
          }
  }

  SECTION("Selective Evaluation") {
    std::vector<wfn_128_type> states;
    std::vector<double> coeffs;
    macis::read_wavefunction<128>(ch4_wfn_fname, states, coeffs);

    coeffs.resize(5000);
    states.resize(5000);

    // Renormalize C for trace computation
    auto c_nrm = blas::nrm2(coeffs.size(), coeffs.data(), 1);
    blas::scal(coeffs.size(), 1. / c_nrm, coeffs.data(), 1);

    std::vector<double> ordm_test(norb * norb, 0.0);
    macis::matrix_span<double> ordm_test_span(ordm_test.data(), norb, norb);
    std::vector<double> trdm_test(norb * norb * norb * norb, 0.0);
    macis::rank4_span<double> trdm_test_span(trdm_test.data(), norb, norb, norb,
                                             norb);

    // Compute reference RDMs
    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(), ordm_span, trdm_span);

    // Selective evaluation of the 1RDM
    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(), ordm_test_span,
                      macis::rank4_span<double>(nullptr, 0, 0, 0, 0));

    for (auto i = 0; i < norb * norb; ++i) {
      REQUIRE_THAT(ordm_test[i],
                   Catch::Matchers::WithinAbs(
                       ordm[i], testing::numerical_zero_tolerance));
    }

    // Selective evaluation of the 2RDM
    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(),
                      macis::matrix_span<double>(nullptr, 0, 0),
                      trdm_test_span);

    for (auto i = 0; i < norb * norb * norb * norb; ++i) {
      REQUIRE_THAT(trdm_test[i],
                   Catch::Matchers::WithinAbs(
                       trdm[i], testing::numerical_zero_tolerance));
    }
  }
}

using wfn_128_type = macis::wfn_t<128>;
TEMPLATE_TEST_CASE("Unrestricted RDMS", "[ham-gen]",
                   macis::DoubleLoopHamiltonianGenerator<wfn_128_type>,
                   macis::SortedDoubleLoopHamiltonianGenerator<wfn_128_type>) {
  ROOT_ONLY(MPI_COMM_WORLD);

  auto norb = 34;
  const auto norb2 = norb * norb;
  const auto norb3 = norb2 * norb;
  const size_t num_occupied_orbitals_alpha = 5;
  const size_t num_occupied_orbitals_beta = 3;

  std::vector<double> T(norb * norb, 0.0);
  std::vector<double> V(norb3 * norb, 0.0);
  std::vector<double> ordm(norb * norb, 0.0), trdm(norb3 * norb, 0.0);

  macis::matrix_span<double> T_span(T.data(), norb, norb);
  macis::matrix_span<double> ordm_span(ordm.data(), norb, norb);
  macis::rank4_span<double> V_span(V.data(), norb, norb, norb, norb);
  macis::rank4_span<double> trdm_span(trdm.data(), norb, norb, norb, norb);

  using wfn_traits = macis::wavefunction_traits<wfn_128_type>;
  using generator_type = TestType;
  generator_type ham_gen(T_span, V_span);

  auto abs_sum = [](auto a, auto b) { return a + std::abs(b); };

  SECTION("HF") {
    std::vector<wfn_128_type> dets = {wfn_traits::canonical_hf_determinant(
        num_occupied_orbitals_alpha, num_occupied_orbitals_beta)};

    std::vector<double> C = {1.};

    ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                      C.data(), ordm_span, trdm_span);
    for (auto i = 0ul; i < num_occupied_orbitals_alpha; ++i) {
      for (auto j = 0ul; j < num_occupied_orbitals_alpha; ++j)
        for (auto k = 0ul; k < num_occupied_orbitals_alpha; ++k)
          for (auto l = 0ul; l < num_occupied_orbitals_alpha; ++l) {
            // ii jj
            if (i == j && k == l && i != k) {
              if (k >= num_occupied_orbitals_beta ||
                  i >= num_occupied_orbitals_beta) {
                REQUIRE(trdm_span(i, j, k, l) - 1.0 <
                        testing::numerical_zero_tolerance);
              } else {
                REQUIRE(trdm_span(i, j, k, l) - 2.0 <
                        testing::numerical_zero_tolerance);
              }
            }
            // ijji
            else if (i == l && j == k && i != j) {
              if (k >= num_occupied_orbitals_beta ||
                  i >= num_occupied_orbitals_beta) {
                REQUIRE(trdm_span(i, j, k, l) + 0.5 <
                        testing::numerical_zero_tolerance);
              } else {
                REQUIRE(trdm_span(i, j, k, l) + 1.0 <
                        testing::numerical_zero_tolerance);
              }
            }
            // iiii
            else if (i == j && k == l && i == k) {
              REQUIRE(trdm_span(i, j, k, l) - 1.0 <
                      testing::numerical_zero_tolerance);
            }
          }
    }
    for (auto i = 0ul; i < num_occupied_orbitals_beta; ++i) {
      REQUIRE(ordm_span(i, i) - 2.0 < testing::numerical_zero_tolerance);
    }
    for (auto i = num_occupied_orbitals_beta; i < num_occupied_orbitals_alpha;
         ++i) {
      REQUIRE(ordm_span(i, i) - 1.0 < testing::numerical_zero_tolerance);
    }
  }

  SECTION("CI") {
    std::vector<wfn_128_type> states;
    std::vector<double> coeffs;
    macis::read_wavefunction<128>(o2_wfn_fname, states, coeffs);

    coeffs.resize(5000);
    states.resize(5000);

    // Renormalize C for trace computation
    auto c_nrm = blas::nrm2(coeffs.size(), coeffs.data(), 1);
    blas::scal(coeffs.size(), 1. / c_nrm, coeffs.data(), 1);

    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(), ordm_span, trdm_span);
    auto sum_ordm = std::accumulate(ordm.begin(), ordm.end(), 0.0, abs_sum);
    auto sum_trdm = std::accumulate(trdm.begin(), trdm.end(), 0.0, abs_sum);
    REQUIRE_THAT(sum_ordm, Catch::Matchers::WithinAbs(
                               8.0000005763, testing::ascii_text_tolerance));
    REQUIRE_THAT(sum_trdm, Catch::Matchers::WithinAbs(
                               44.1092472977, testing::ascii_text_tolerance));

    double trace_ordm = 0.;
    for (auto p = 0; p < norb; ++p) trace_ordm += ordm_span(p, p);
    REQUIRE_THAT(trace_ordm,
                 Catch::Matchers::WithinAbs(
                     num_occupied_orbitals_alpha + num_occupied_orbitals_beta,
                     testing::ascii_text_tolerance));

    // Check symmetries
    for (auto p = 0; p < norb; ++p)
      for (auto q = p; q < norb; ++q) {
        REQUIRE_THAT(ordm_span(p, q),
                     Catch::Matchers::WithinAbs(ordm_span(q, p),
                                                testing::ascii_text_tolerance));
      }

    // check spin dependent RDMs
    std::vector<double> ordm_aa(norb * norb, 0.0);
    std::vector<double> ordm_bb(norb * norb, 0.0);
    std::vector<double> trdm_aaaa(norb3 * norb, 0.0);
    std::vector<double> trdm_bbbb(norb3 * norb, 0.0);
    std::vector<double> trdm_aabb(norb3 * norb, 0.0);

    macis::matrix_span<double> ordm_aa_span(ordm_aa.data(), norb, norb);
    macis::matrix_span<double> ordm_bb_span(ordm_bb.data(), norb, norb);
    macis::rank4_span<double> trdm_aaaa_span(trdm_aaaa.data(), norb, norb, norb,
                                             norb);
    macis::rank4_span<double> trdm_bbbb_span(trdm_bbbb.data(), norb, norb, norb,
                                             norb);
    macis::rank4_span<double> trdm_aabb_span(trdm_aabb.data(), norb, norb, norb,
                                             norb);

    ham_gen.form_rdms_spin_dep(states.begin(), states.end(), states.begin(),
                               states.end(), coeffs.data(), ordm_aa_span,
                               ordm_bb_span, trdm_aaaa_span, trdm_bbbb_span,
                               trdm_aabb_span);
    // check 1rdm
    for (auto p = 0; p < norb; ++p)
      for (auto q = 0; q < norb; ++q) {
        REQUIRE_THAT(
            ordm_span(p, q),
            Catch::Matchers::WithinAbs(ordm_aa_span(p, q) + ordm_bb_span(p, q),
                                       testing::numerical_zero_tolerance));
      }

    // check 2rdm
    for (auto p = 0; p < norb; ++p)
      for (auto q = 0; q < norb; ++q)
        for (auto r = 0; r < norb; ++r)
          for (auto s = 0; s < norb; ++s) {
            REQUIRE_THAT(
                trdm_span(p, q, r, s),
                Catch::Matchers::WithinAbs(
                    trdm_aaaa_span(p, q, r, s) + trdm_bbbb_span(p, q, r, s) +
                        trdm_aabb_span(p, q, r, s) + trdm_aabb_span(r, s, p, q),
                    testing::numerical_zero_tolerance));
          }
  }

  SECTION("Selective Evaluation") {
    std::vector<wfn_128_type> states;
    std::vector<double> coeffs;
    macis::read_wavefunction<128>(o2_wfn_fname, states, coeffs);

    coeffs.resize(5000);
    states.resize(5000);

    // Renormalize C for trace computation
    auto c_nrm = blas::nrm2(coeffs.size(), coeffs.data(), 1);
    blas::scal(coeffs.size(), 1. / c_nrm, coeffs.data(), 1);

    std::vector<double> ordm_test(norb * norb, 0.0);
    macis::matrix_span<double> ordm_test_span(ordm_test.data(), norb, norb);
    std::vector<double> trdm_test(norb * norb * norb * norb, 0.0);
    macis::rank4_span<double> trdm_test_span(trdm_test.data(), norb, norb, norb,
                                             norb);

    // Compute reference RDMs
    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(), ordm_span, trdm_span);

    // Selective evaluation of the 1RDM
    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(), ordm_test_span,
                      macis::rank4_span<double>(nullptr, 0, 0, 0, 0));

    for (auto i = 0; i < norb * norb; ++i) {
      REQUIRE_THAT(ordm_test[i],
                   Catch::Matchers::WithinAbs(
                       ordm[i], testing::numerical_zero_tolerance));
    }

    // Selective evaluation of the 2RDM
    ham_gen.form_rdms(states.begin(), states.end(), states.begin(),
                      states.end(), coeffs.data(),
                      macis::matrix_span<double>(nullptr, 0, 0),
                      trdm_test_span);

    for (auto i = 0; i < norb * norb * norb * norb; ++i) {
      REQUIRE_THAT(trdm_test[i],
                   Catch::Matchers::WithinAbs(
                       trdm[i], testing::numerical_zero_tolerance));
    }
  }
}
