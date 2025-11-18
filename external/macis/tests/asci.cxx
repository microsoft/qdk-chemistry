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

#include <iostream>
#include <macis/asci/determinant_contributions.hpp>
#include <macis/asci/grow.hpp>
#include <macis/asci/pt2.hpp>
#include <macis/asci/refine.hpp>
#include <macis/bitset_operations.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>
#include <macis/util/fcidump.hpp>

#include "ut_common.hpp"

template <size_t NRadix, size_t NBits>
std::array<unsigned, NRadix> top_set_indices(std::bitset<NBits> word) {
  std::array<unsigned, NRadix> top_set_indices;
  for (size_t i = 0; i < NRadix; ++i) {
    auto r = macis::fls(word);
    top_set_indices[i] = r;
    word.flip(r);
  }
  return top_set_indices;
}

template <size_t NRadix, size_t NBits>
size_t top_set_ordinal(std::bitset<NBits> word, size_t NSet) {
  auto ind = top_set_indices<NRadix>(word);
  size_t ord = 0;
  size_t fact = 1;
  for (size_t i = 0; i < NRadix; ++i) {
    ord += ind[NRadix - i - 1] * fact;
    fact *= NSet;
  }
  return ord;
}

template <size_t N>
auto make_quad(unsigned i, unsigned j, unsigned k, unsigned l) {
  using wfn_type = macis::wfn_t<N>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using constraint_type = macis::alpha_constraint<wfn_traits>;
  using string_type = typename constraint_type::constraint_type;

  string_type C = 0;
  C.flip(i).flip(j).flip(k).flip(l);
  string_type B = 1;
  B <<= l;
  B = B.to_ullong() - 1;

  return constraint_type(C, B, l);
}

TEST_CASE("Triplets") {
  constexpr size_t num_bits = 64;
  size_t norb = 32;

  using wfn_less_comparator = macis::bitset_less_comparator<num_bits>;
  using wfn_type = macis::wfn_t<num_bits>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;

  // Generate ficticious wfns
  std::vector<macis::wfn_t<num_bits>> wfn_a = {
      15  //, 30, 15, 29
  };

  std::vector<macis::wfn_t<num_bits>> wfn_b = {
      15  //, 15, 30, 15
  };

  const size_t ndet = wfn_a.size();

  // Combine the alpha/beta strings
  std::vector<macis::wfn_t<num_bits>> wfns(ndet);
  for (int i = 0; i < ndet; ++i) {
    wfns[i] = (wfn_a[i] << num_bits / 2) | wfn_b[i];
  }

  // Total sort of combined bit-strings
  std::sort(wfns.begin(), wfns.end(), wfn_less_comparator{});

  // Extract unique alphas
  std::vector<macis::wfn_t<num_bits>> wfn_a_uniq(wfns);
  {
    // Extract alpha strings
    std::transform(wfn_a_uniq.begin(), wfn_a_uniq.end(), wfn_a_uniq.begin(),
                   [=](const auto& w) { return w >> (num_bits / 2); });

    // Determine unique alphas in place
    auto it = std::unique(wfn_a_uniq.begin(), wfn_a_uniq.end());

    // Remove excess
    wfn_a_uniq.erase(it, wfn_a_uniq.end());
  }

  // Count beta dets per unique alpha
  const size_t nuniq_alpha = wfn_a_uniq.size();
  std::vector<size_t> nbetas(nuniq_alpha, 0);
  for (size_t i = 0, i_alpha = 0; i < ndet; ++i) {
    const auto& w = wfns[i];
    const auto& w_a = wfn_a_uniq[i_alpha];
    if ((w >> num_bits / 2) != w_a) i_alpha++;
    nbetas[i_alpha]++;
  }

  // Print beta counts
  // for( size_t i = 0; i < nuniq_alpha; ++i )
  //  std::cout << wfn_a_uniq[i].to_ulong() << " " << nbetas[i] << std::endl;

  // Compute Histograms
  std::vector<size_t> triplet_hist(norb * norb * norb, 0);
  std::vector<size_t> quad_hist(norb * norb * norb * norb, 0);
  for (auto i = 0; i < nuniq_alpha; ++i) {
    // Constant dimensions
    const size_t num_occupied_orbitals = wfn_a_uniq[i].count();
    const size_t num_virtual_orbitals = norb - num_occupied_orbitals;
    const size_t n_singles = num_occupied_orbitals * num_virtual_orbitals;
    const size_t n_doubles = (n_singles * (n_singles - num_occupied_orbitals -
                                           num_virtual_orbitals + 1)) /
                             4;

    // Generate singles and doubles
    std::vector<macis::wfn_t<num_bits>> s_a, d_a;
    macis::generate_singles_doubles(norb, wfn_a_uniq[i], s_a, d_a);

    // Histogram contribution from root determinant
    {
      const auto label = top_set_ordinal<3>(wfn_a_uniq[i], norb);
      triplet_hist[label] += n_singles + n_doubles + 1;
    }
    {
      const auto label = top_set_ordinal<4>(wfn_a_uniq[i], norb);
      quad_hist[label] += n_singles + n_doubles + 1;
    }

    // Histogram contribtutions from single excitations
    for (const auto& w : s_a) {
      const auto label = top_set_ordinal<3>(w, norb);
      triplet_hist[label] += n_singles + 1;
    }
    for (const auto& w : s_a) {
      const auto label = top_set_ordinal<4>(w, norb);
      quad_hist[label] += n_singles + 1;
    }

    // Histogram contribtuions from double excitations
    for (const auto& w : d_a) {
      const auto label = top_set_ordinal<3>(w, norb);
      triplet_hist[label]++;
    }
    for (const auto& w : d_a) {
      const auto label = top_set_ordinal<4>(w, norb);
      quad_hist[label]++;
    }
  }

  REQUIRE(std::accumulate(triplet_hist.begin(), triplet_hist.end(), 0ul) ==
          std::accumulate(quad_hist.begin(), quad_hist.end(), 0ul));

  // Print Histogram
  // for( auto i = 0; i < triplet_hist.size() ; ++ i ) {
  //  if(triplet_hist[i]) std::cout << i << " " << triplet_hist[i] << std::endl;
  //}

  std::vector<std::tuple<unsigned, unsigned, unsigned>> triplets;
  for (int i = 0; i < norb; ++i)
    for (int j = 0; j < i; ++j)
      for (int k = 0; k < j; ++k) {
        triplets.emplace_back(i, j, k);
      }

  std::vector<size_t> new_triplet_hist(triplet_hist.size(), 0);
  for (auto [i, j, k] : triplets) {
    const auto label = i * 32 * 32 + j * 32 + k;
    auto constraint = macis::make_triplet<num_bits>(i, j, k);

    for (auto det : wfn_a_uniq) {
      const size_t num_occupied_orbitals = det.count();
      const size_t num_virtual_orbitals = norb - num_occupied_orbitals;
      const size_t n_singles = num_occupied_orbitals * num_virtual_orbitals;
      const size_t n_doubles = (n_singles * (n_singles - num_occupied_orbitals -
                                             num_virtual_orbitals + 1)) /
                               4;

      new_triplet_hist[label] += macis::constraint_histogram(
          wfn_traits::alpha_string(det), n_singles, n_doubles, constraint);
    }
  }

  REQUIRE(triplet_hist == new_triplet_hist);

  std::vector<size_t> new_quad_hist(quad_hist.size(), 0);
  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>> quads;
  for (int i = 0; i < norb; ++i)
    for (int j = 0; j < i; ++j)
      for (int k = 0; k < j; ++k)
        for (int l = 0; l < k; ++l) {
          quads.emplace_back(i, j, k, l);
        }

  for (auto [i, j, k, l] : quads) {
    const size_t label =
        i * 32ul * 32ul * 32ul + j * 32ul * 32ul + k * 32ul + l;
    auto constraint = make_quad<num_bits>(i, j, k, l);

    for (auto det : wfn_a_uniq) {
      const size_t num_occupied_orbitals = det.count();
      const size_t num_virtual_orbitals = norb - num_occupied_orbitals;
      const size_t n_singles = num_occupied_orbitals * num_virtual_orbitals;
      const size_t n_doubles = (n_singles * (n_singles - num_occupied_orbitals -
                                             num_virtual_orbitals + 1)) /
                               4;

      new_quad_hist[label] += macis::constraint_histogram(
          wfn_traits::alpha_string(det), n_singles, n_doubles, constraint);
    }
  }

  REQUIRE(quad_hist == new_quad_hist);
}

TEST_CASE("Constraints") {
  using wfn_type = macis::wfn_t<64>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using spin_wfn_type = typename wfn_traits::spin_wfn_type;
  using spin_wfn_traits = macis::wavefunction_traits<spin_wfn_type>;
  using wfn_comparator = typename wfn_traits::spin_comparator;
  using spin_wfn_comparator = typename spin_wfn_traits::wfn_comparator;

  using constraint_type = macis::alpha_constraint<wfn_traits>;

  const size_t norb = 10;
  size_t nalpha, nbeta;

  SECTION("Closed Shell") {
    nalpha = 6;
    nbeta = 6;
  }

  SECTION("Open Shell") {
    nalpha = 6;
    nbeta = 3;
  }

  // Generate Hilbert Space
  auto dets = macis::generate_hilbert_space<wfn_type>(norb, nalpha, nbeta);
  std::sort(dets.begin(), dets.end(), wfn_comparator{});  // lex sort

  // Get Alpha Compression Data
  std::vector<std::pair<spin_wfn_type, size_t>> unique_alpha;
  unique_alpha.push_back({wfn_traits::alpha_string(dets[0]), 1});
  for (size_t i = 1; i < dets.size(); ++i) {
    auto& [cur_alpha, cur_count] = unique_alpha.back();
    auto alpha_i = wfn_traits::alpha_string(dets[i]);
    if (alpha_i == cur_alpha) {
      cur_count++;
    } else {
      unique_alpha.push_back({alpha_i, 1});
    }
  }

  // Get offsets
  std::vector<size_t> unique_alpha_offsets(unique_alpha.size());
  std::transform_exclusive_scan(
      unique_alpha.begin(), unique_alpha.end(), unique_alpha_offsets.begin(),
      0ul, std::plus<size_t>(), [](auto p) { return p.second; });

  // size_t ntot = std::accumulate( unique_alpha.begin(), unique_alpha.end(),
  //   0ul, [](auto s, auto a){ return s + a.second; } );
  // REQUIRE(ntot == dets.size());

  // Generate constraints
  std::vector<constraint_type> triplets, quads;
  for (int i = 0; i < norb; ++i)
    for (int j = 0; j < i; ++j)
      for (int k = 0; k < j; ++k) {
        triplets.emplace_back(macis::make_triplet<64>(i, j, k));
        for (int l = 0; l < k; ++l) {
          quads.emplace_back(make_quad<64>(i, j, k, l));
        }
      }

  // Check doubles
  auto check_doubles = [](auto det, auto C, auto& doubles) {
    // Sanity check
    auto [_O, _V] = macis::generate_constraint_double_excitations(det, C);
    REQUIRE(std::all_of(_O.begin(), _O.end(),
                        [](auto e) { return spin_wfn_traits::count(e) == 2; }));
    REQUIRE(std::all_of(_V.begin(), _V.end(),
                        [](auto e) { return spin_wfn_traits::count(e) == 2; }));

    // Extract all double excitations that satisfy the constraint
    std::vector<spin_wfn_type> doubles_c;
    std::copy_if(doubles.begin(), doubles.end(), std::back_inserter(doubles_c),
                 [=](auto d) { return C.satisfies_constraint(d); });

    // Generate all double excitations that satisfy the constraint
    std::vector<spin_wfn_type> doubles_g;
    macis::generate_constraint_doubles(det, C, doubles_g);

    // Compare and check counting
    std::sort(doubles_c.begin(), doubles_c.end(), spin_wfn_comparator{});
    std::sort(doubles_g.begin(), doubles_g.end(), spin_wfn_comparator{});
    REQUIRE_THAT(doubles_c, Catch::Matchers::Equals(doubles_g));
    REQUIRE(doubles_g.size() == macis::count_constraint_doubles(det, C));
  };

  // Check singles
  auto check_singles = [](auto det, auto C, auto& singles) {
    // Extract all double excitations that satisfy the constraint
    std::vector<spin_wfn_type> singles_c;
    std::copy_if(singles.begin(), singles.end(), std::back_inserter(singles_c),
                 [=](auto d) { return C.satisfies_constraint(d); });

    // Generate all double excitations that satisfy the constraint
    std::vector<spin_wfn_type> singles_g;
    macis::generate_constraint_singles(det, C, singles_g);

    // Compare and check counting
    std::sort(singles_c.begin(), singles_c.end(), spin_wfn_comparator{});
    std::sort(singles_g.begin(), singles_g.end(), spin_wfn_comparator{});
    REQUIRE_THAT(singles_c, Catch::Matchers::Equals(singles_g));
    REQUIRE(singles_g.size() == macis::count_constraint_singles(det, C));
  };

  for (auto i = 0; i < unique_alpha.size(); ++i) {
    auto det = unique_alpha[i].first;

    auto det_triplet = top_set_indices<3>(det);
    auto det_quad = top_set_indices<4>(det);

    std::vector<spin_wfn_type> singles, doubles;
    macis::generate_singles(norb, det, singles);
    macis::generate_doubles(norb, det, doubles);

    for (auto C : triplets) {
      // Check validity of constraint check
      auto inds = top_set_indices<3>(C.C());
      REQUIRE(C.C_min() == inds.back());
      REQUIRE(C.satisfies_constraint(det) == (inds == det_triplet));

      check_singles(det, C, singles);
      check_doubles(det, C, doubles);
    }

    for (auto C : quads) {
      // Check validity of constraint check
      auto inds = top_set_indices<4>(C.C());
      REQUIRE(C.C_min() == inds.back());
      REQUIRE(C.satisfies_constraint(det) == (inds == det_quad));

      check_singles(det, C, singles);
      check_doubles(det, C, doubles);
    }
  }
}

TEST_CASE("ASCI") {
  MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);)
  using macis::NumActive;
  using macis::NumElectron;
  using macis::NumInactive;
  using macis::NumOrbital;
  using macis::NumVirtual;

  // spdlog::null_logger_mt("davidson");
  // spdlog::null_logger_mt("ci_solver");
  // spdlog::null_logger_mt("asci_search");
  // spdlog::null_logger_mt("asci_grow");
  // spdlog::null_logger_mt("asci_refine");

  // Read Water FCIDUMP
  const size_t norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
  const size_t norb2 = norb * norb;
  const size_t norb4 = norb2 * norb2;

  std::vector<double> T(norb2), V(norb4);
  auto E_core = macis::read_fcidump_core(water_ccpvdz_fcidump);
  macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
  macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

  // Hamiltonian Genereator
  using wfn_type = macis::wfn_t<64>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;
  using generator_t = macis::DoubleLoopHamiltonianGenerator<wfn_type>;
  generator_t ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  uint32_t nalpha(5), nbeta(5);

  macis::ASCISettings asci_settings;
  macis::MCSCFSettings mcscf_settings;

  // HF guess
  std::vector<wfn_type> dets = {
      wfn_traits::canonical_hf_determinant(nalpha, nbeta)};
  ;
  std::vector<double> C = {1.0};
  double E0 = ham_gen.matrix_element(dets[0], dets[0]);

  // ASCI Grow
  asci_settings.ntdets_max = 10000;
  std::tie(E0, dets, C) = macis::asci_grow(
      asci_settings, mcscf_settings, E0, std::move(dets), std::move(C), ham_gen,
      norb MACIS_MPI_CODE(, MPI_COMM_WORLD));

  // std::cout << E0 - -8.542926243842e+01 << std::endl;
  REQUIRE_THAT(E0, Catch::Matchers::WithinAbs(-8.542926580489e+01,
                                              testing::davidson_tolerance));
  REQUIRE(dets.size() == 10000);
  REQUIRE(C.size() == 10000);
  REQUIRE_THAT(std::inner_product(C.begin(), C.end(), C.begin(), 0.0),
               Catch::Matchers::WithinAbs(1.0, 1e-12));

  // ASCI Refine
  std::tie(E0, dets, C) = macis::asci_refine(
      asci_settings, mcscf_settings, E0, std::move(dets), std::move(C), ham_gen,
      norb MACIS_MPI_CODE(, MPI_COMM_WORLD));

  REQUIRE_THAT(E0, Catch::Matchers::WithinAbs(-8.542926585527e+01,
                                              testing::davidson_tolerance));
  REQUIRE(dets.size() == 10000);
  REQUIRE(C.size() == 10000);
  REQUIRE_THAT(std::inner_product(C.begin(), C.end(), C.begin(), 0.0),
               Catch::Matchers::WithinAbs(1.0, 1e-12));

#ifdef MACIS_ENABLE_MPI
  // ASCI-PT2
  auto EPT2 = macis::asci_pt2_constraint(
      asci_settings, dets.begin(), dets.end(), E0, C, norb, ham_gen.T(),
      ham_gen.G_red(), ham_gen.V_red(), ham_gen.G(), ham_gen.V(),
      ham_gen MACIS_MPI_CODE(, MPI_COMM_WORLD));

  // std::cout << std::scientific << std::setprecision(12);
  // std::cout << EPT2 << std::endl;
  REQUIRE_THAT(EPT2, Catch::Matchers::WithinAbs(-5.701535028967e-03,
                                                testing::davidson_tolerance));
#endif

  MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);)
  spdlog::drop_all();
}
