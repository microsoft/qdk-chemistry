// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <cstddef>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_scalar.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

// Open-shell single determinant: distinct alpha and beta electron counts.
class WavefunctionBlockedCountsTest : public ::testing::Test {};

TEST_F(WavefunctionBlockedCountsTest, ActiveElectronCountIsSpinBlocked) {
  auto orbitals = testing::create_test_orbitals(4, 4, /*restricted=*/false);
  auto det = Configuration::from_spin_half_string("2u00");  // 2 alpha, 1 beta
  StateVectorContainer sd(det, orbitals);

  auto count = sd.active_num_particles();
  ASSERT_NE(count, nullptr);
  EXPECT_TRUE(count->symmetries()[0]->has_axis(AxisName::Spin));
  EXPECT_EQ(count->value(axes::alpha()), 2u);
  EXPECT_EQ(count->value(axes::beta()), 1u);

  // Independent channels: open-shell counts are not aliased.
  EXPECT_NE(count->block_ptr({axes::alpha()}).get(),
            count->block_ptr({axes::beta()}).get());

  // The v1 spin-resolved view agrees with the blocked counts.
  auto [n_alpha, n_beta] = sd.get_active_num_electrons();
  EXPECT_EQ(n_alpha, 2u);
  EXPECT_EQ(n_beta, 1u);
}

TEST_F(WavefunctionBlockedCountsTest, ActiveOccupationsAreSpinBlocked) {
  auto orbitals = testing::create_test_orbitals(4, 4, /*restricted=*/true);
  auto det = Configuration::from_spin_half_string("2u00");
  StateVectorContainer sd(det, orbitals);

  auto occ = sd.active_orbital_occupations();
  ASSERT_NE(occ, nullptr);
  EXPECT_TRUE(occ->symmetries()[0]->has_axis(AxisName::Spin));

  const Eigen::VectorXd& alpha = occ->block({axes::alpha()});
  const Eigen::VectorXd& beta = occ->block({axes::beta()});
  EXPECT_EQ(alpha.size(), 4);
  EXPECT_EQ(beta.size(), 4);

  // alpha occupies orbitals 0 and 1; beta occupies only orbital 0.
  EXPECT_DOUBLE_EQ(alpha(0), 1.0);
  EXPECT_DOUBLE_EQ(alpha(1), 1.0);
  EXPECT_DOUBLE_EQ(beta(0), 1.0);
  EXPECT_DOUBLE_EQ(beta(1), 0.0);

  // The v1 spin-resolved view agrees with the blocked occupations.
  auto [v1_alpha, v1_beta] = sd.get_active_orbital_occupations();
  EXPECT_TRUE(v1_alpha.isApprox(alpha));
  EXPECT_TRUE(v1_beta.isApprox(beta));
}

TEST_F(WavefunctionBlockedCountsTest, TotalCountMatchesV1) {
  auto orbitals = testing::create_test_orbitals(4, 4, /*restricted=*/true);
  auto det = Configuration::from_spin_half_string("2200");  // 2 alpha, 2 beta
  StateVectorContainer sd(det, orbitals);

  auto total = sd.total_num_particles();
  auto [n_alpha, n_beta] = sd.get_total_num_electrons();
  EXPECT_EQ(total->value(axes::alpha()), n_alpha);
  EXPECT_EQ(total->value(axes::beta()), n_beta);
}

// --------------------------------------------------------------------------
// Generic (spinless) wavefunction — trivial symmetry, no spin axis.
// --------------------------------------------------------------------------

TEST_F(WavefunctionBlockedCountsTest, TrivialSymmetryWfnAggregatesCounts) {
  // ModelOrbitals with no symmetry arg → trivial (spinless) single-particle
  // basis. This demonstrates the "generic wavefunction" construction path:
  // a linear combination of bitstrings with no spin structure.
  auto orbitals = std::make_shared<ModelOrbitals>(4);

  // Two-determinant expansion over 4 modes.
  // Occupation characters use the fermionic alphabet, but with trivial
  // symmetry the container aggregates into a single block.
  auto det1 = Configuration::from_spin_half_string("ud00");  // 2 occupied modes
  auto det2 = Configuration::from_spin_half_string("u0d0");  // 2 occupied modes
  Eigen::VectorXd coeffs(2);
  coeffs << 0.9, 0.1;
  StateVectorContainer sv(
      ContainerTypes::VectorVariant{coeffs},
      ContainerTypes::DeterminantVector{det1, det2}, orbitals);

  // Active electron count is a single trivial block (no spin axis).
  auto count = sv.active_num_particles();
  ASSERT_NE(count, nullptr);
  EXPECT_FALSE(count->symmetries()[0]->has_axis(AxisName::Spin));

  // Sz-dependent v1 accessors must throw — there is no spin axis.
  EXPECT_THROW(sv.get_active_num_electrons(), std::runtime_error);
  EXPECT_THROW(sv.get_total_num_electrons(), std::runtime_error);
  EXPECT_THROW(sv.get_active_orbital_occupations(), std::runtime_error);
  EXPECT_THROW(sv.get_total_orbital_occupations(), std::runtime_error);
}

// --------------------------------------------------------------------------
// Generic (bitstring) wavefunction construction — from_bitstring + inactive.
// --------------------------------------------------------------------------

TEST_F(WavefunctionBlockedCountsTest, BitstringWithInactiveSpaceCountsCorrectly) {
  // 5 modes: inactive {0}, active {1,2,3}, virtual {4}
  auto active_idx = std::make_shared<const SymmetryBlockedIndexSet>(
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial()),
      std::unordered_map<SymmetryLabel, std::size_t>{{SymmetryLabel{}, 5}},
      std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>{
          {SymmetryLabel{}, {1, 2, 3}}});
  auto inactive_idx = std::make_shared<const SymmetryBlockedIndexSet>(
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial()),
      std::unordered_map<SymmetryLabel, std::size_t>{{SymmetryLabel{}, 5}},
      std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>{
          {SymmetryLabel{}, {0}}});
  auto orbitals = std::make_shared<ModelOrbitals>(active_idx, inactive_idx);

  // Active-space determinant: 2 of 3 active modes occupied.
  auto det = Configuration::from_bitstring("110");
  EXPECT_EQ(det.bits_per_mode(), 1);

  StateVectorContainer sv(det, orbitals);

  // Active particles: 2 (bits set in the bitstring).
  auto active_count = sv.active_num_particles();
  ASSERT_NE(active_count, nullptr);
  EXPECT_FALSE(active_count->symmetries()[0]->has_axis(AxisName::Spin));
  EXPECT_EQ(active_count->value(SymmetryLabel{}), 2u);

  // Total particles: 2 active + 1 inactive = 3.
  auto total_count = sv.total_num_particles();
  ASSERT_NE(total_count, nullptr);
  EXPECT_EQ(total_count->value(SymmetryLabel{}), 3u);

  // Total occupations: inactive mode 0 = 1.0, active modes 1,2 = 1.0, rest 0.
  auto total_occ = sv.total_orbital_occupations();
  ASSERT_NE(total_occ, nullptr);
  const auto& occ_vec = total_occ->block({SymmetryLabel{}});
  EXPECT_EQ(occ_vec.size(), 5);
  EXPECT_DOUBLE_EQ(occ_vec(0), 1.0);  // inactive
  EXPECT_DOUBLE_EQ(occ_vec(1), 1.0);  // active, occupied
  EXPECT_DOUBLE_EQ(occ_vec(2), 1.0);  // active, occupied
  EXPECT_DOUBLE_EQ(occ_vec(3), 0.0);  // active, unoccupied
  EXPECT_DOUBLE_EQ(occ_vec(4), 0.0);  // virtual

  // Spin-resolved v1 accessors must throw.
  EXPECT_THROW(sv.get_total_num_electrons(), std::runtime_error);
  EXPECT_THROW(sv.get_total_orbital_occupations(), std::runtime_error);
}

TEST_F(WavefunctionBlockedCountsTest, BitstringFullActiveCountsCorrectly) {
  // Full active space (no inactive), pure bitstring construction.
  auto orbitals = std::make_shared<ModelOrbitals>(4);

  auto det1 = Configuration::from_bitstring("1010");
  auto det2 = Configuration::from_bitstring("0110");
  Eigen::VectorXd coeffs(2);
  coeffs << 0.8, 0.6;
  StateVectorContainer sv(
      ContainerTypes::VectorVariant{coeffs},
      ContainerTypes::DeterminantVector{det1, det2}, orbitals);

  auto active_count = sv.active_num_particles();
  ASSERT_NE(active_count, nullptr);
  EXPECT_EQ(active_count->value(SymmetryLabel{}), 2u);

  auto total_count = sv.total_num_particles();
  EXPECT_EQ(total_count->value(SymmetryLabel{}), 2u);

  // Spin-resolved v1 accessors must throw — no spin axis.
  EXPECT_THROW(sv.get_active_num_electrons(), std::runtime_error);

  // Multi-determinant occupations require a 1-RDM; without one the
  // container correctly reports unavailability.
  EXPECT_FALSE(sv.has_one_rdm_spin_dependent());
}
