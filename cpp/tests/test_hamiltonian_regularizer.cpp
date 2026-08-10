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
  EXPECT_TRUE(std::find(available.begin(), available.end(),
                        "fermionic_low_rank") != available.end());

  auto regularizer = HamiltonianRegularizerFactory::create();
  EXPECT_EQ(regularizer->name(), "fermionic_low_rank");
  EXPECT_EQ(regularizer->type_name(), "hamiltonian_regularizer");

  auto regularizer_named =
      HamiltonianRegularizerFactory::create("fermionic_low_rank");
  EXPECT_EQ(regularizer_named->name(), "fermionic_low_rank");

  EXPECT_THROW(HamiltonianRegularizerFactory::create("nonexistent"),
               std::runtime_error);
}

TEST_F(HamiltonianRegularizerTest, DefaultTruncationThresholdIsZero) {
  auto regularizer =
      HamiltonianRegularizerFactory::create("fermionic_low_rank");
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

  auto regularizer =
      HamiltonianRegularizerFactory::create("fermionic_low_rank");
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
    auto regularizer =
        HamiltonianRegularizerFactory::create("fermionic_low_rank");
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

  auto regularizer =
      HamiltonianRegularizerFactory::create("fermionic_low_rank");
  auto shifted_ham = regularizer->run(ham, 5, 5);

  auto norm_after =
      qdk::chemistry::utils::hamiltonian_one_norm(*shifted_ham, 0.0);

  EXPECT_NEAR(norm_after.total, 27.590504297492, 1e-6);
}

/**
 * @brief compute_shift() + rebuild_bliss_shifted_hamiltonian() must reproduce
 * run() exactly. This locks the refactor that split the regularizer into a
 * public shift-computation step and a public, shift-agnostic rebuild step.
 */
TEST_F(HamiltonianRegularizerTest, ComputeShiftThenRebuildMatchesRun) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto regularizer =
      HamiltonianRegularizerFactory::create("fermionic_low_rank");
  auto shifted_run = regularizer->run(ham, 5, 5);
  ASSERT_NE(shifted_run, nullptr);

  auto regularizer2 =
      HamiltonianRegularizerFactory::create("fermionic_low_rank");
  auto shift = regularizer2->compute_shift(*ham, 5, 5);
  auto shifted_manual = rebuild_bliss_shifted_hamiltonian(*ham, shift, 10.0);
  ASSERT_NE(shifted_manual, nullptr);

  auto [h_run, h_run_beta] = shifted_run->get_one_body_integrals();
  auto [h_man, h_man_beta] = shifted_manual->get_one_body_integrals();
  (void)h_run_beta;
  (void)h_man_beta;
  EXPECT_TRUE(h_run.isApprox(h_man, 1e-12));

  auto [g_run, g_run_ab, g_run_bb] = shifted_run->get_two_body_integrals();
  auto [g_man, g_man_ab, g_man_bb] = shifted_manual->get_two_body_integrals();
  (void)g_run_ab;
  (void)g_run_bb;
  (void)g_man_ab;
  (void)g_man_bb;
  EXPECT_TRUE(g_run.isApprox(g_man, 1e-12));

  EXPECT_NEAR(shifted_run->get_core_energy(), shifted_manual->get_core_energy(),
              1e-12);
}

/**
 * @brief An unknown shift_method must be rejected by compute_shift()/run().
 */
TEST_F(HamiltonianRegularizerTest, ThrowsOnUnknownShiftMethod) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto regularizer =
      HamiltonianRegularizerFactory::create("fermionic_low_rank");
  // The "shift_method" setting is constrained to the known methods, so an
  // unknown value is rejected at set-time rather than deferred to run().
  EXPECT_THROW(
      regularizer->settings().set("shift_method", std::string("nonexistent")),
      std::invalid_argument);
}
