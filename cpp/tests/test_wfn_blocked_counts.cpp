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
  Configuration det("2u00");  // 2 alpha, 1 beta
  StateVectorContainer sd(det, orbitals);

  auto count = sd.active_num_electrons();
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
  Configuration det("2u00");
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
  Configuration det("2200");  // 2 alpha, 2 beta
  StateVectorContainer sd(det, orbitals);

  auto total = sd.total_num_electrons();
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
  Configuration det1("ud00");  // 2 occupied modes
  Configuration det2("u0d0");  // 2 occupied modes
  Eigen::VectorXd coeffs(2);
  coeffs << 0.9, 0.1;
  StateVectorContainer sv(
      ContainerTypes::VectorVariant{coeffs},
      ContainerTypes::DeterminantVector{det1, det2}, orbitals);

  // Active electron count is a single trivial block (no spin axis).
  auto count = sv.active_num_electrons();
  ASSERT_NE(count, nullptr);
  EXPECT_FALSE(count->symmetries()[0]->has_axis(AxisName::Spin));

  // Sz-dependent v1 accessors must throw — there is no spin axis.
  EXPECT_THROW(sv.get_active_num_electrons(), std::runtime_error);
  EXPECT_THROW(sv.get_total_num_electrons(), std::runtime_error);
  EXPECT_THROW(sv.get_active_orbital_occupations(), std::runtime_error);
  EXPECT_THROW(sv.get_total_orbital_occupations(), std::runtime_error);
}
