// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <filesystem>
#include <qdk/chemistry/data/wavefunction_containers/cas_state_vector.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

namespace {

// Occupied-orbital lists of every k-electron string of n orbitals, sorted
// lexicographically
std::vector<std::vector<size_t>> lexicographic_strings(size_t n, size_t k) {
  std::vector<std::vector<size_t>> strings;
  for (size_t mask = 0; mask < (size_t(1) << n); ++mask) {
    if (static_cast<size_t>(std::popcount(mask)) != k) continue;
    std::vector<size_t> occupied;
    for (size_t p = 0; p < n; ++p)
      if ((mask >> p) & 1) occupied.push_back(p);
    strings.push_back(occupied);
  }
  std::sort(strings.begin(), strings.end());
  return strings;
}

// Every determinant of the complete active space, alpha-major
std::vector<Configuration> complete_space(size_t n, size_t n_alpha,
                                          size_t n_beta) {
  std::vector<Configuration> dets;
  for (const auto& alpha : lexicographic_strings(n, n_alpha)) {
    for (const auto& beta : lexicographic_strings(n, n_beta)) {
      std::string occupations(n, '0');
      for (size_t p : alpha) occupations[p] = 'u';
      for (size_t p : beta) occupations[p] = occupations[p] == 'u' ? '2' : 'd';
      dets.push_back(Configuration::from_spin_half_string(occupations));
    }
  }
  return dets;
}

Eigen::VectorXd normalized_random(size_t size) {
  Eigen::VectorXd coeffs = Eigen::VectorXd::Random(size);
  return coeffs / coeffs.norm();
}

}  // namespace

TEST(CasStateVectorContainerTest, DeterminantsFollowAlphaMajorLexicographic) {
  for (auto [n, n_alpha, n_beta] : std::vector<std::array<size_t, 3>>{
           {4, 2, 2}, {5, 2, 1}, {3, 0, 2}, {3, 3, 1}, {1, 1, 0}, {6, 3, 3}}) {
    const auto expected = complete_space(n, n_alpha, n_beta);
    auto orbitals = testing::create_test_orbitals(n, n, true);
    CasStateVectorContainer cas(normalized_random(expected.size()), n_alpha,
                                n_beta, orbitals);

    EXPECT_EQ(cas.size(), expected.size());
    const auto& dets = cas.get_active_determinants();
    ASSERT_EQ(dets.size(), expected.size());
    for (size_t i = 0; i < dets.size(); ++i) {
      EXPECT_EQ(dets[i].to_string(), expected[i].to_string());
    }
    EXPECT_EQ(cas.get_configuration_set().size(), expected.size());
  }
}

TEST(CasStateVectorContainerTest, CoefficientLookupMatchesSearch) {
  const size_t n = 5, n_alpha = 2, n_beta = 3;
  const auto dets = complete_space(n, n_alpha, n_beta);
  auto orbitals = testing::create_test_orbitals(n, n, true);
  const Eigen::VectorXd coeffs = normalized_random(dets.size());
  CasStateVectorContainer cas(coeffs, n_alpha, n_beta, orbitals);
  StateVectorContainer sv(coeffs, dets, orbitals);

  for (size_t i = 0; i < dets.size(); ++i) {
    EXPECT_EQ(std::get<double>(cas.get_coefficient(dets[i])), coeffs[i]);
    EXPECT_TRUE(cas.contains_determinant(dets[i]));
  }

  // Wrong electron counts, electrons past the active orbitals, and different
  // capacities are absent from both expansions; unoccupied padding within the
  // capacity is not
  for (const char* query : {"22200", "22du0", "2dd00u", "22000d", "22d",
                            "22d000000", "22d00000", "22d00"}) {
    const auto det = Configuration::from_spin_half_string(query);
    EXPECT_EQ(std::get<double>(cas.get_coefficient(det)),
              std::get<double>(sv.get_coefficient(det)))
        << query;
    EXPECT_EQ(cas.contains_determinant(det), sv.contains_determinant(det))
        << query;
  }
  EXPECT_TRUE(
      cas.contains_determinant(Configuration::from_spin_half_string("22d00")));

  const Eigen::VectorXcd complex_coeffs =
      coeffs.cast<std::complex<double>>() * std::complex<double>(0.6, 0.8);
  CasStateVectorContainer complex_cas(complex_coeffs, n_alpha, n_beta,
                                      orbitals);
  EXPECT_TRUE(complex_cas.is_complex());
  EXPECT_EQ(
      std::get<std::complex<double>>(complex_cas.get_coefficient(dets[7])),
      complex_coeffs[7]);
}

TEST(CasStateVectorContainerTest, MatchesStateVectorContainer) {
  auto base_orbitals = testing::create_test_orbitals(6, 6, true);
  auto orbitals = testing::with_active_space(base_orbitals, {1, 2, 3, 4},
                                             std::vector<size_t>{0});
  const size_t n = 4, n_alpha = 2, n_beta = 1;
  const auto dets = complete_space(n, n_alpha, n_beta);
  const Eigen::VectorXd coeffs = normalized_random(dets.size());

  Eigen::MatrixXd one_aa = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd one_bb = Eigen::MatrixXd::Zero(n, n);
  one_aa.diagonal() << 0.9, 0.7, 0.3, 0.1;
  one_bb.diagonal() << 0.6, 0.2, 0.15, 0.05;

  StateVectorContainer sv(coeffs, dets, orbitals, std::nullopt, one_aa, one_bb,
                          std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt);
  CasStateVectorContainer cas(coeffs, n_alpha, n_beta, orbitals, std::nullopt,
                              one_aa, one_bb, std::nullopt, std::nullopt,
                              std::nullopt, std::nullopt);

  EXPECT_EQ(cas.get_container_type(), "state_vector");
  EXPECT_EQ(cas.size(), sv.size());
  EXPECT_DOUBLE_EQ(cas.norm(), sv.norm());
  EXPECT_EQ(cas.get_active_num_electrons(), sv.get_active_num_electrons());
  EXPECT_EQ(cas.get_total_num_electrons(), sv.get_total_num_electrons());
  EXPECT_EQ(cas.get_total_num_electrons(),
            std::make_pair(size_t{3}, size_t{2}));
  EXPECT_TRUE(cas.get_active_orbital_occupations().first.isApprox(
      sv.get_active_orbital_occupations().first));
  EXPECT_TRUE(cas.get_total_orbital_occupations().second.isApprox(
      sv.get_total_orbital_occupations().second));
  EXPECT_EQ(cas.to_json(), sv.to_json());

  const double norm2 = coeffs.squaredNorm();
  EXPECT_NEAR(std::get<double>(cas.overlap(cas)), norm2, 1e-14);
  EXPECT_NEAR(std::get<double>(cas.overlap(sv)), norm2, 1e-14);
  EXPECT_NEAR(std::get<double>(sv.overlap(cas)), norm2, 1e-14);

  // Complex coefficients: the overlap conjugates the bra
  const Eigen::VectorXcd complex_coeffs =
      coeffs.cast<std::complex<double>>() * std::complex<double>(0.6, 0.8) +
      Eigen::VectorXcd::Constant(coeffs.size(), std::complex<double>(0, 0.1));
  StateVectorContainer complex_sv(complex_coeffs, dets, orbitals);
  CasStateVectorContainer complex_cas(complex_coeffs, n_alpha, n_beta,
                                      orbitals);
  auto complex_overlap = [](const StateVectorContainer& bra,
                            const StateVectorContainer& ket) {
    return std::get<std::complex<double>>(bra.overlap(ket));
  };
  EXPECT_LT(std::abs(complex_overlap(complex_cas, complex_cas) -
                     complex_overlap(complex_sv, complex_sv)),
            1e-14);
  EXPECT_LT(std::abs(complex_overlap(cas, complex_cas) -
                     complex_overlap(sv, complex_sv)),
            1e-14);
  EXPECT_LT(std::abs(complex_overlap(complex_cas, cas) -
                     complex_overlap(complex_sv, sv)),
            1e-14);

  Wavefunction wfn_cas(std::make_unique<CasStateVectorContainer>(
      coeffs, n_alpha, n_beta, orbitals));
  Wavefunction wfn_sv(
      std::make_unique<StateVectorContainer>(coeffs, dets, orbitals));
  EXPECT_TRUE(wfn_cas.has_container_type<StateVectorContainer>());
  EXPECT_EQ(wfn_cas.content_hash(), wfn_sv.content_hash());
  const auto [top_cas, top_cas_coeffs] = wfn_cas.get_top_determinants(3);
  const auto [top_sv, top_sv_coeffs] = wfn_sv.get_top_determinants(3);
  EXPECT_EQ(top_cas, top_sv);
  EXPECT_TRUE(std::get<Eigen::VectorXd>(top_cas_coeffs)
                  .isApprox(std::get<Eigen::VectorXd>(top_sv_coeffs)));
}

TEST(CasStateVectorContainerTest, SingleDeterminantSpace) {
  // Empty and filled shells leave a single determinant, whose RDMs are
  // generated from its occupations
  auto orbitals = testing::create_test_orbitals(3, 3, true);
  for (auto [n_alpha, n_beta] :
       std::vector<std::array<size_t, 2>>{{0, 0}, {3, 3}, {3, 0}}) {
    const auto dets = complete_space(3, n_alpha, n_beta);
    ASSERT_EQ(dets.size(), 1u);
    Eigen::VectorXd one = Eigen::VectorXd::Ones(1);
    StateVectorContainer sv(one, dets, orbitals);
    CasStateVectorContainer cas(one, n_alpha, n_beta, orbitals);

    EXPECT_EQ(cas.size(), 1u);
    EXPECT_EQ(cas.has_one_rdm_spin_traced(), sv.has_one_rdm_spin_traced());
    EXPECT_TRUE(std::get<Eigen::MatrixXd>(cas.get_active_one_rdm_spin_traced())
                    .isApprox(std::get<Eigen::MatrixXd>(
                        sv.get_active_one_rdm_spin_traced())));
    EXPECT_TRUE(cas.get_active_orbital_occupations().first.isApprox(
        sv.get_active_orbital_occupations().first));
  }
}

TEST(CasStateVectorContainerTest, CloneClearCachesAndSerialization) {
  const size_t n = 4, n_alpha = 2, n_beta = 2;
  auto orbitals = testing::create_test_orbitals(n, n, true);
  const auto dets = complete_space(n, n_alpha, n_beta);
  const Eigen::VectorXd coeffs = normalized_random(dets.size());
  CasStateVectorContainer cas(coeffs, n_alpha, n_beta, orbitals);

  const auto clone = cas.clone();
  EXPECT_NE(dynamic_cast<const CasStateVectorContainer*>(clone.get()), nullptr);
  EXPECT_EQ(clone->get_active_determinants(), dets);

  EXPECT_EQ(cas.get_active_determinants(), dets);
  cas.clear_caches();
  EXPECT_EQ(cas.get_active_determinants(), dets);

  // Files hold the explicit state vector and load as a StateVectorContainer
  Wavefunction wfn(std::make_unique<CasStateVectorContainer>(coeffs, n_alpha,
                                                             n_beta, orbitals));
  const auto from_json = Wavefunction::from_json(wfn.to_json());
  EXPECT_EQ(from_json->get_active_determinants(), dets);
  EXPECT_TRUE(std::get<Eigen::VectorXd>(from_json->get_coefficients())
                  .isApprox(coeffs));

  const std::string filename = "test_cas_state_vector.wavefunction.h5";
  wfn.to_hdf5_file(filename);
  const auto from_hdf5 = Wavefunction::from_hdf5_file(filename);
  std::filesystem::remove(filename);
  EXPECT_EQ(from_hdf5->get_container_type(), "state_vector");
  EXPECT_EQ(from_hdf5->get_active_determinants(), dets);
  EXPECT_TRUE(std::get<Eigen::VectorXd>(from_hdf5->get_coefficients())
                  .isApprox(coeffs));
}

TEST(CasStateVectorContainerTest, InvalidArgumentsThrow) {
  auto orbitals = testing::create_test_orbitals(3, 3, true);
  const Eigen::VectorXd eight = Eigen::VectorXd::Ones(8);
  const Eigen::VectorXd one = Eigen::VectorXd::Ones(1);
  EXPECT_THROW(CasStateVectorContainer(eight, 1, 2, orbitals),
               std::invalid_argument);
  EXPECT_THROW(CasStateVectorContainer(one, 4, 0, orbitals),
               std::invalid_argument);

  // Alpha and beta active spaces of different sizes
  auto spin_mismatched = std::make_shared<ModelOrbitals>(
      testing::unrestricted_index_set(4, {0, 1, 2}, {0, 1}));
  const Eigen::VectorXd nine = Eigen::VectorXd::Ones(9);
  EXPECT_THROW(CasStateVectorContainer(nine, 1, 1, spin_mismatched),
               std::invalid_argument);
}
