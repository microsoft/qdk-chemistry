/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <iomanip>
#include <iostream>
#include <macis/mcscf/orbital_energies.hpp>
#include <macis/util/fcidump.hpp>
#include <numeric>

#include "ut_common.hpp"

TEST_CASE("FCIDUMP") {
  ROOT_ONLY(MPI_COMM_WORLD);

  SECTION("READ") {
    SECTION("Header") {
      // Test parsing the header directly
      auto header = macis::fcidump_read_header(water_ccpvdz_fcidump);

      REQUIRE(header.norb == 24);
      REQUIRE(header.nelec == 10);
      REQUIRE(header.ms2 == 0);
      REQUIRE(header.isym == 1);
      REQUIRE(header.orbsym.size() == 24);
      for (auto s : header.orbsym) {
        REQUIRE(s == 1);
      }
    }

    size_t norb_ref = 24;
    SECTION("NORB") {
      auto norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
      REQUIRE(norb == norb_ref);
    }

    SECTION("Core") {
      auto coreE = macis::read_fcidump_core(water_ccpvdz_fcidump);
      REQUIRE_THAT(coreE,
                   Catch::Matchers::WithinAbs(9.191200742618042,
                                              testing::ascii_text_tolerance));
    }

    SECTION("OneBody") {
      std::vector<double> T(norb_ref * norb_ref);
      macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb_ref);
      double sum = std::accumulate(T.begin(), T.end(), 0.0);
      REQUIRE_THAT(sum,
                   Catch::Matchers::WithinAbs(-1.095432762653e+02,
                                              testing::ascii_text_tolerance));
    }

    SECTION("TwoBody") {
      std::vector<double> V(norb_ref * norb_ref * norb_ref * norb_ref);
      macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb_ref);
      double sum = std::accumulate(V.begin(), V.end(), 0.0);
      REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(
                            2.701609068389e+02, testing::ascii_text_tolerance));
    }

    SECTION("Integer Integrals") {
      // Two-site Hubbard models (U = 4, E_core = 2) with integer-valued
      // integrals: hopping 1 in the standard column order, where every line
      // holds non-negative integers, and hopping -1 with the indices first
      const std::vector<std::pair<std::string, double>> cases = {
          {hubbard_2site_fcidump, 1.0},
          {hubbard_2site_indices_first_fcidump, -1.0}};
      for (const auto& [fname, t] : cases) {
        const std::vector<double> T_ref = {0.0, t, t, 0.0};
        std::vector<double> V_ref(16, 0.0);
        V_ref[0] = V_ref[15] = 4.0;  // (11|11) and (22|22)

        std::vector<double> T(4), V(16);
        double E_core = 0.0;
        macis::read_fcidump_all(fname, T.data(), 2, V.data(), 2, E_core);
        REQUIRE(E_core == 2.0);
        REQUIRE_THAT(T, Catch::Matchers::Equals(T_ref));
        REQUIRE_THAT(V, Catch::Matchers::Equals(V_ref));

        std::vector<double> T1(4), V1(16);
        macis::read_fcidump_1body(fname, T1.data(), 2);
        macis::read_fcidump_2body(fname, V1.data(), 2);
        REQUIRE(macis::read_fcidump_core(fname) == 2.0);
        REQUIRE_THAT(T1, Catch::Matchers::Equals(T_ref));
        REQUIRE_THAT(V1, Catch::Matchers::Equals(V_ref));
      }
    }

    SECTION("Orbital Out of Range") {
      std::vector<double> T(4), V(16);
      double E_core = 0.0;
      REQUIRE_THROWS_AS(
          macis::read_fcidump_1body(out_of_range_fcidump, T.data(), 2),
          std::runtime_error);
      REQUIRE_THROWS_AS(
          macis::read_fcidump_2body(out_of_range_fcidump, V.data(), 2),
          std::runtime_error);
      REQUIRE_THROWS_AS(macis::read_fcidump_all(out_of_range_fcidump, T.data(),
                                                2, V.data(), 2, E_core),
                        std::runtime_error);
    }

    SECTION("Validity Checks") {
      auto norb = norb_ref;
      size_t num_occupied_orbitals = 5;
      const auto norb2 = norb * norb;
      const auto norb3 = norb2 * norb;
      std::vector<double> T(norb * norb);
      std::vector<double> V(norb * norb * norb * norb);
      macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
      macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

      std::vector<double> eps(norb);
      macis::canonical_orbital_energies(
          macis::NumOrbital(norb), macis::NumInactive(num_occupied_orbitals),
          T.data(), norb, V.data(), norb, eps.data());

      // Check orbital energies
      std::vector<double> ref_eps = {
          -20.5504959651472,  -1.33652308180416,  -0.699084807015877,
          -0.566535827115135, -0.493126779151562, 0.185508268977809,
          0.25620321890592,   0.78902083505165,   0.854064342891826,
          1.16354210561965,   1.20037835222836,   1.25333579878846,
          1.44452900701373,   1.4762147730592,    1.67453827449385,
          1.86734472965445,   1.93460967763627,   2.4520550848955,
          2.48955724627828,   3.28543361094139,   3.33853354817945,
          3.5101570353257,    3.86543012937399,   4.14719831587618,
      };

      for (auto i = 0ul; i < norb; ++i)
        REQUIRE_THAT(eps[i], Catch::Matchers::WithinAbs(
                                 ref_eps[i], testing::ascii_text_tolerance));

      // MP2
      double EMP2 = 0.;
      for (size_t i = 0; i < num_occupied_orbitals; ++i)
        for (size_t a = num_occupied_orbitals; a < norb; ++a)
          for (size_t j = 0; j < num_occupied_orbitals; ++j)
            for (size_t b = num_occupied_orbitals; b < norb; ++b) {
              double den = eps[a] + eps[b] - eps[i] - eps[j];
              double dir = V[a + i * norb + b * norb2 + j * norb3];
              double exh = V[b + i * norb + a * norb2 + j * norb3];

              EMP2 -= (dir * (2 * dir - exh)) / den;
            }
      REQUIRE_THAT(EMP2,
                   Catch::Matchers::WithinAbs(-0.203989305096243,
                                              testing::ascii_text_tolerance));
    }
  }
}
