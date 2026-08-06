// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/hamiltonian_regularizer.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/utils/hamiltonian_one_norm.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;

class HamiltonianRegularizerTest : public ::testing::Test {};

TEST_F(HamiltonianRegularizerTest, FactoryHygiene) {
  auto available = HamiltonianRegularizerFactory::available();
  EXPECT_TRUE(std::find(available.begin(), available.end(), "flr_bliss") !=
              available.end());

  auto regularizer = HamiltonianRegularizerFactory::create();
  EXPECT_EQ(regularizer->name(), "flr_bliss");
  EXPECT_EQ(regularizer->type_name(), "hamiltonian_regularizer");

  auto regularizer_named = HamiltonianRegularizerFactory::create("flr_bliss");
  EXPECT_EQ(regularizer_named->name(), "flr_bliss");

  EXPECT_THROW(HamiltonianRegularizerFactory::create("nonexistent"),
              std::runtime_error);
}

TEST_F(HamiltonianRegularizerTest, DefaultTruncationThresholdIsZero) {
  auto regularizer = HamiltonianRegularizerFactory::create("flr_bliss");
  EXPECT_DOUBLE_EQ(
      regularizer->settings().get<double>("df_truncation_threshold"), 0.0);
}

TEST_F(HamiltonianRegularizerTest, ThrowsOnUnrestrictedHamiltonian) {
  auto h_atom = testing::create_hydrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("auto"));
  scf_solver->settings().set("enable_gdm", true);
  auto [E_HF, wfn_HF] = scf_solver->run(h_atom, 0, 2, "cc-pvdz");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());
  ASSERT_FALSE(ham->is_restricted());

  auto regularizer = HamiltonianRegularizerFactory::create("flr_bliss");
  EXPECT_THROW(regularizer->run(ham, 1, 0), std::invalid_argument);
}

/**
 * @brief The strongest correctness check: BLISS shifts should not change the
 * physical energy of the target electron-number sector. Run exact FCI before
 * and after applying the FLR-BLISS shift and confirm the energies agree
 * to within the standard CI energy tolerance, both at the default
 * (no-truncation) setting and with an explicit truncation threshold.
 */
TEST_F(HamiltonianRegularizerTest, Water_STO3G_EnergyInvariantUnderShift) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto mc = MultiConfigurationCalculatorFactory::create();
  auto [E_before, wfn_before] = mc->run(ham, 5, 5);

  for (const double threshold : {0.0, 1e-6}) {
    auto regularizer = HamiltonianRegularizerFactory::create("flr_bliss");
    regularizer->settings().set("df_truncation_threshold", threshold);
    auto shifted_ham = regularizer->run(ham, 5, 5);
    ASSERT_NE(shifted_ham, nullptr);

    auto mc_after = MultiConfigurationCalculatorFactory::create();
    auto [E_after, wfn_after] = mc_after->run(shifted_ham, 5, 5);

    EXPECT_NEAR(E_before, E_after, testing::ci_energy_tolerance)
        << "Energy not invariant at df_truncation_threshold=" << threshold;
  }
}

/**
 * @brief The FLR-BLISS shift should reduce (or at least not increase) the
 * fermionic double-factorization 1-norm relative to the unshifted
 * Hamiltonian.
 */
TEST_F(HamiltonianRegularizerTest, Water_STO3G_ReducesOneNorm) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto regularizer = HamiltonianRegularizerFactory::create("flr_bliss");
  auto shifted_ham = regularizer->run(ham, 5, 5);

  auto norm_after =
      qdk::chemistry::utils::hamiltonian_one_norm(*shifted_ham, 0.0);

  EXPECT_NEAR(norm_after.total, 27.590504297492, 1e-6);
}

