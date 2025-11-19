/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <iomanip>
#include <macis/mcscf/fock_matrices.hpp>
#include <macis/util/binary_rdms.hpp>
#include <macis/util/fcidump.hpp>
#include <numeric>

#include "ut_common.hpp"

TEST_CASE("Fock Matrices") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  const size_t norb2 = norb * norb;
  const size_t norb4 = norb2 * norb2;

  using macis::NumActive;
  using macis::NumInactive;
  using macis::NumOrbital;
  using macis::NumVirtual;

  std::vector<double> T(norb2), V(norb4);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  SECTION("Inactive Fock Matrix + Energy") {
    std::vector<double> ref_sums = {
        -1.0954327626528054e+02, -7.9049334675325397e+01,
        -5.5421794916425746e+01, -3.2896413433510098e+01,
        -9.6790625410977995e+00, 1.3541890926738670e+01,
        2.8118988584004747e+01,  4.2282474623774370e+01,
        6.1330877241905775e+01,  8.0462280904553381e+01,
        1.0115739942244966e+02,  1.2234906887017347e+02,
        1.4193737383338043e+02,  1.6158815460036422e+02,
        1.8055391956530423e+02,  1.9946376174456546e+02,
        2.2023047024882104e+02,  2.3908456464316561e+02,
        2.6013779050396846e+02,  2.8151944194600628e+02,
        3.0499216988942629e+02,  3.2832856341497029e+02,
        3.5190406182710450e+02,  3.7494850617879558e+02};

    std::vector<double> ref_ene = {
        0.0000000000000000e+00,  -6.1315494224913593e+01,
        -7.2212778636226929e+01, -7.9040381402513631e+01,
        -8.3471297093533877e+01, -8.5217981091537155e+01,
        -8.4546001263510988e+01, -8.2730095439120404e+01,
        -7.8127776664404351e+01, -7.1592548154010231e+01,
        -6.2930084392284229e+01, -5.2096342927325999e+01,
        -3.9855391765969998e+01, -2.5412026151032496e+01,
        -8.6447385900261242e+00, 9.9027251048627019e+00,
        3.0946082901947307e+01,  5.3450370843339400e+01,
        8.0911983168177073e+01,  1.1078946373217127e+02,
        1.4537422212238749e+02,  1.8291223010717863e+02,
        2.2328918072756994e+02,  2.6673925924545023e+02};

    std::vector<double> Fi(norb2, 0.0);
    for (size_t i = 0; i < norb; ++i) {
      NumInactive ninact(i);
      macis::inactive_fock_matrix(NumOrbital(norb), ninact, T.data(), norb,
                                  V.data(), norb, Fi.data(), norb);
      double sum = std::accumulate(Fi.begin(), Fi.end(), 0.0);
      REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(
                            ref_sums[i], testing::ascii_text_tolerance));

      double E =
          macis::inactive_energy(ninact, T.data(), norb, Fi.data(), norb);
      REQUIRE_THAT(E, Catch::Matchers::WithinAbs(
                          ref_ene[i], testing::ascii_text_tolerance));
    }
  }

  SECTION("Active Space Hamiltonian") {
    NumInactive ninact(1);
    NumActive nact(8);

    // Compute the inactive Fock matrix
    std::vector<double> Fi_ref(norb2);
    macis::inactive_fock_matrix(NumOrbital(norb), ninact, T.data(), norb,
                                V.data(), norb, Fi_ref.data(), norb);

    std::vector<double> T_active(nact.get() * nact.get()),
        V_active(T_active.size() * T_active.size()), Fi(norb2);
    macis::active_hamiltonian(NumOrbital(norb), nact, ninact, T.data(), norb,
                              V.data(), norb, Fi.data(), norb, T_active.data(),
                              nact.get(), V_active.data(), nact.get());

    for (auto i = 0; i < norb2; ++i)
      REQUIRE_THAT(Fi[i], Catch::Matchers::WithinAbs(
                              Fi_ref[i], testing::ascii_text_tolerance));

    macis::matrix_span<double> Ta(T_active.data(), nact.get(), nact.get());
    macis::matrix_span<double> Fi_span(Fi.data(), norb, norb);

    auto act_range = std::make_pair(ninact.get(), ninact.get() + nact.get());
    auto Fi_active = macis::KokkosEx::submdspan(Fi_span, act_range, act_range);
    for (auto i = 0; i < nact.get(); ++i)
      for (auto j = 0; j < nact.get(); ++j) {
        REQUIRE(Ta(i, j) == Fi_active(i, j));
      }

    macis::rank4_span<double> V_span(V.data(), norb, norb, norb, norb);
    auto V_act_span = macis::KokkosEx::submdspan(V_span, act_range, act_range,
                                                 act_range, act_range);
    macis::rank4_span<double> Va(V_active.data(), nact.get(), nact.get(),
                                 nact.get(), nact.get());
    for (auto i = 0; i < nact.get(); ++i)
      for (auto j = 0; j < nact.get(); ++j)
        for (auto k = 0; k < nact.get(); ++k)
          for (auto l = 0; l < nact.get(); ++l) {
            REQUIRE(Va(i, j, k, l) == V_act_span(i, j, k, l));
          }
  }

  SECTION("Active Fock") {
    NumInactive ninact(1);
    NumActive nact(8);

    // Read RDMs
    size_t na2 = nact.get() * nact.get();
    size_t na4 = na2 * na2;
    std::vector<double> active_1rdm(na2), active_2rdm(na4);
    macis::read_rdms_binary(water_ccpvdz_rdms_fname, nact.get(),
                            active_1rdm.data(), nact.get(), active_2rdm.data(),
                            nact.get());

    REQUIRE(active_1rdm.size() == nact.get() * nact.get());
    std::vector<double> Fa(norb2);
    macis::active_fock_matrix(NumOrbital(norb), ninact, nact, V.data(), norb,
                              active_1rdm.data(), nact.get(), Fa.data(), norb);
    auto sum = std::accumulate(Fa.begin(), Fa.end(), 0.0);
    REQUIRE_THAT(sum,
                 Catch::Matchers::WithinAbs(9.2434393637673907e+01,
                                            testing::ascii_text_tolerance));
  }

  SECTION("Auxillary Q") {
    NumInactive ninact(1);
    NumActive nact(8);

    // Read RDMs
    size_t na2 = nact.get() * nact.get();
    size_t na4 = na2 * na2;
    std::vector<double> active_1rdm(na2), active_2rdm(na4);
    macis::read_rdms_binary(water_ccpvdz_rdms_fname, nact.get(),
                            active_1rdm.data(), nact.get(), active_2rdm.data(),
                            nact.get());

    std::vector<double> Q(nact.get() * norb);
    macis::aux_q_matrix(nact, NumOrbital(norb), ninact, V.data(), norb,
                        active_2rdm.data(), nact.get(), Q.data(), nact.get());
    auto sum = std::accumulate(Q.begin(), Q.end(), 0.0);
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(
                          2.609524939005e+01, testing::ascii_text_tolerance));
  }

  SECTION("Generalized Fock Matrix") {
    NumInactive ninact(1);
    NumActive nact(8);

    // Read RDMs
    size_t na2 = nact.get() * nact.get();
    size_t na4 = na2 * na2;
    std::vector<double> active_1rdm(na2), active_2rdm(na4);
    macis::read_rdms_binary(water_ccpvdz_rdms_fname, nact.get(),
                            active_1rdm.data(), nact.get(), active_2rdm.data(),
                            nact.get());

    // Compute Intermediates
    std::vector<double> Fi(norb2, 0.0), Fa(norb2, 0.0), Q(nact.get() * norb);
    macis::inactive_fock_matrix(NumOrbital(norb), ninact, T.data(), norb,
                                V.data(), norb, Fi.data(), norb);
    macis::active_fock_matrix(NumOrbital(norb), ninact, nact, V.data(), norb,
                              active_1rdm.data(), nact.get(), Fa.data(), norb);
    macis::aux_q_matrix(nact, NumOrbital(norb), ninact, V.data(), norb,
                        active_2rdm.data(), nact.get(), Q.data(), nact.get());

    std::vector<double> F(norb2, 0.0);
    const double ref_sum = -4.7465630072124384e+01;
    SECTION("From All Intermediates") {
      macis::generalized_fock_matrix(
          NumOrbital(norb), ninact, nact, Fi.data(), norb, Fa.data(), norb,
          active_1rdm.data(), nact.get(), Q.data(), nact.get(), F.data(), norb);
      auto sum = std::accumulate(F.begin(), F.end(), 0.0);
      REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(
                            ref_sum, testing::ascii_text_tolerance));
    }

    SECTION("From Inactive Intermediates") {
      macis::generalized_fock_matrix_comp_mat1(
          NumOrbital(norb), ninact, nact, Fi.data(), norb, V.data(), norb,
          active_1rdm.data(), nact.get(), active_2rdm.data(), nact.get(),
          F.data(), norb);
      auto sum = std::accumulate(F.begin(), F.end(), 0.0);
      REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(
                            ref_sum, testing::ascii_text_tolerance));
    }

    SECTION("From No Intermediates") {
      macis::generalized_fock_matrix_comp_mat2(
          NumOrbital(norb), ninact, nact, T.data(), norb, V.data(), norb,
          active_1rdm.data(), nact.get(), active_2rdm.data(), nact.get(),
          F.data(), norb);
      auto sum = std::accumulate(F.begin(), F.end(), 0.0);
      REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(
                            ref_sum, testing::ascii_text_tolerance));
    }
  }

  SECTION("Energy from Generalized Fock") {
    NumInactive ninact(1);
    NumActive nact(8);

    // Read RDMs
    size_t na2 = nact.get() * nact.get();
    size_t na4 = na2 * na2;
    std::vector<double> active_1rdm(na2), active_2rdm(na4);
    macis::read_rdms_binary(water_ccpvdz_rdms_fname, nact.get(),
                            active_1rdm.data(), nact.get(), active_2rdm.data(),
                            nact.get());

    // Compute Fock
    std::vector<double> F(norb2, 0.0);
    macis::generalized_fock_matrix_comp_mat2(
        NumOrbital(norb), ninact, nact, T.data(), norb, V.data(), norb,
        active_1rdm.data(), nact.get(), active_2rdm.data(), nact.get(),
        F.data(), norb);

    // Compute energy
    auto E = macis::energy_from_generalized_fock(ninact, nact, T.data(), norb,
                                                 active_1rdm.data(), nact.get(),
                                                 F.data(), norb);
    REQUIRE_THAT(E, Catch::Matchers::WithinAbs(-8.5250440649419417e+01,
                                               testing::ascii_text_tolerance));
  }
}
