/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <blas.hh>
#include <iomanip>
#include <iostream>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/wavefunction_io.hpp>

#include "ut_common.hpp"

TEST_CASE("Double Loop") {
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

  using wfn_type = macis::wfn_t<64>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_type = macis::DoubleLoopHamiltonianGenerator<wfn_type>;

#if 0
  generator_type ham_gen(norb, V.data(), T.data());
#else
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
#endif
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

  SECTION("RDM") {
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
}

TEST_CASE("RDMS") {
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

  using wfn_type = macis::wfn_t<128>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_type = macis::DoubleLoopHamiltonianGenerator<wfn_type>;
  generator_type ham_gen(T_span, V_span);

  auto abs_sum = [](auto a, auto b) { return a + std::abs(b); };

  SECTION("HF") {
    std::vector<wfn_type> dets = {wfn_traits::canonical_hf_determinant(
        num_occupied_orbitals, num_occupied_orbitals)};

    std::vector<double> C = {1.};

    ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                      C.data(), ordm_span, trdm_span);

    for (auto i = 0ul; i < num_occupied_orbitals; ++i)
      for (auto j = 0ul; j < num_occupied_orbitals; ++j)
        for (auto k = 0ul; k < num_occupied_orbitals; ++k)
          for (auto l = 0ul; l < num_occupied_orbitals; ++l) {
            trdm_span(i, j, l, k) -= 0.5 * ordm_span(i, j) * ordm_span(k, l);
            trdm_span(i, j, l, k) += 0.25 * ordm_span(i, l) * ordm_span(k, j);
          }
    auto sum = std::accumulate(trdm.begin(), trdm.end(), 0.0, abs_sum);
    REQUIRE(sum < testing::numerical_zero_tolerance);

    for (auto i = 0ul; i < num_occupied_orbitals; ++i) ordm_span(i, i) -= 2.0;
    sum = std::accumulate(ordm.begin(), ordm.end(), 0.0, abs_sum);
    REQUIRE(sum < testing::numerical_zero_tolerance);
  }

  SECTION("CI") {
    std::vector<wfn_type> states;
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
    REQUIRE_THAT(sum_trdm,
                 Catch::Matchers::WithinAbs(9.928269867561e+01,
                                            testing::ascii_text_tolerance));

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
  }
}
