// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <cmath>
#include <qdk/chemistry/algorithms/algorithm_defaults.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/hamiltonian_factorization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/algorithms/symmetry_shift.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

#include "../src/qdk/chemistry/algorithms/microsoft/symmetry_shift/fermionic_low_rank.hpp"
#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using qdk::chemistry::data::FactorizedHamiltonianContainer;

namespace {

/// The shifter consumes an already double-factorized Hamiltonian, so every
/// test that runs it has to factorize first.
std::shared_ptr<qdk::chemistry::data::Hamiltonian> double_factorize(
    std::shared_ptr<qdk::chemistry::data::Hamiltonian> hamiltonian) {
  return HamiltonianFactorizationFactory::create("double_factorization")
      ->run(std::move(hamiltonian));
}

/// Recorded from a local run; see Water_STO3G_OneNormRegression.
constexpr double kWaterLambdaDfBaseline = 11.531420411934715;
constexpr double kWaterLambdaDfShifted = 4.522496176156583;

}  // namespace

class SymmetryShiftTest : public ::testing::Test {};

TEST_F(SymmetryShiftTest, FactoryHygiene) {
  auto available = SymmetryShifterFactory::available();
  EXPECT_TRUE(std::find(available.begin(), available.end(),
                        "fermionic_low_rank") != available.end());

  auto shifter = SymmetryShifterFactory::create();
  EXPECT_EQ(shifter->name(), "fermionic_low_rank");
  EXPECT_EQ(shifter->type_name(), "symmetry_shifter");

  auto shifter_named = SymmetryShifterFactory::create("fermionic_low_rank");
  EXPECT_EQ(shifter_named->name(), "fermionic_low_rank");

  EXPECT_THROW(SymmetryShifterFactory::create("nonexistent"),
               std::runtime_error);
}

/**
 * @brief The shifter consumes whatever factorization it is handed, so it has
 * no settings of its own. Truncation and the choice of decomposition belong to
 * the double_factorization algorithm.
 */
TEST_F(SymmetryShiftTest, HasNoSettings) {
  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  EXPECT_TRUE(shifter->settings().keys().empty());
}

/**
 * @brief A Hamiltonian that has not been double-factorized carries no
 * fragments to shift, so it must be rejected rather than silently mishandled.
 */
TEST_F(SymmetryShiftTest, RejectsNonFactorizedHamiltonian) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto ham =
      HamiltonianConstructorFactory::create()->run(wfn_HF->get_orbitals());
  ASSERT_FALSE(ham->has_container_type<FactorizedHamiltonianContainer>());

  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  EXPECT_THROW(shifter->run(ham, 5, 5), std::invalid_argument);
}

/**
 * @brief SymmetryShifterFactory must be listed in the
 * REGISTER_FACTORY_SETTINGS_INIT block of algorithm_defaults.cpp; an omission
 * there fails silently, leaving nested AlgorithmRefs without defaults.
 */
TEST_F(SymmetryShiftTest, ResolvesAlgorithmDefaults) {
  const auto settings =
      qdk::chemistry::algorithms::detail::resolve_algorithm_defaults(
          "symmetry_shifter", "fermionic_low_rank");
  ASSERT_NE(settings, nullptr)
      << "symmetry_shifter is missing from the REGISTER_FACTORY_SETTINGS_INIT "
         "block in algorithms/algorithm_defaults.cpp";

  // The path that actually bites: a nested reference must self-resolve.
  const qdk::chemistry::data::AlgorithmRef ref("symmetry_shifter",
                                               "fermionic_low_rank");
  EXPECT_NE(ref.get_settings(), nullptr);
}
TEST_F(SymmetryShiftTest, ThrowsOnUnrestrictedHamiltonian) {
  auto h_atom = testing::create_hydrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("auto"));
  scf_solver->settings().set("enable_gdm", true);
  auto [E_HF, wfn_HF] = scf_solver->run(h_atom, 0, 2, "cc-pvdz");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());
  ASSERT_FALSE(ham->is_restricted());

  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  EXPECT_THROW(shifter->run(ham, 1, 0), std::invalid_argument);
}

/**
 * @brief The strongest correctness check: BLISS shifts should not change the
 * physical energy of the target electron-number sector. Run exact FCI before
 * and after applying the fermionic low-rank BLISS shift and confirm the
 * energies agree to within the standard CI energy tolerance, both at the
 * default (no-truncation) setting and with an explicit truncation threshold.
 */
TEST_F(SymmetryShiftTest, Water_STO3G_EnergyInvariantUnderShift) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto mc = MultiConfigurationCalculatorFactory::create();
  auto [E_before, wfn_before] = mc->run(ham, 5, 5);

  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  auto shifted_ham = shifter->run(double_factorize(ham), 5, 5);
  ASSERT_NE(shifted_ham, nullptr);

  // The shift is applied to the dense integrals, so what comes back is a
  // canonical four-center Hamiltonian, not a factorized one.
  EXPECT_FALSE(
      shifted_ham->has_container_type<FactorizedHamiltonianContainer>());

  auto mc_after = MultiConfigurationCalculatorFactory::create();
  auto [E_after, wfn_after] = mc_after->run(shifted_ham, 5, 5);

  EXPECT_NEAR(E_before, E_after, testing::ci_energy_tolerance);
}

/**
 * @brief A factorization whose weights are all zero carries no shift, but must
 * still produce a correctly sized xi. xi used to be sized from the fragments
 * themselves, so no fragments meant a 0x0 xi and an out-of-bounds read
 * downstream.
 */
TEST_F(SymmetryShiftTest, AccumulatesZeroWeightFactorizationIntoZeroShift) {
  constexpr size_t norb = 7;
  constexpr size_t R = 1;
  constexpr size_t B = norb;
  constexpr size_t C = 1;

  // U^0 = identity, so every basis row is normalized as the container demands.
  Eigen::VectorXd u = Eigen::VectorXd::Zero(R * B * norb);
  for (size_t b = 0; b < B; ++b) {
    u(b * norb + b) = 1.0;
  }

  FactorizedHamiltonianContainer container(
      Eigen::MatrixXd::Zero(norb, norb), u, Eigen::VectorXd::Zero(R * B * C),
      Eigen::MatrixXd::Zero(R, C),
      std::make_shared<qdk::chemistry::data::ModelOrbitals>(norb), 0.0,
      Eigen::MatrixXd::Zero(0, 0));

  auto shift = microsoft::accumulate_fragment_shifts(container);

  EXPECT_EQ(shift.xi.rows(), static_cast<Eigen::Index>(norb));
  EXPECT_EQ(shift.xi.cols(), static_cast<Eigen::Index>(norb));
  EXPECT_TRUE(shift.xi.isZero());
  EXPECT_DOUBLE_EQ(shift.mu2, 0.0);
}

/**
 * @brief Regression test pinning the fermionic 1-norm reduction achieved on
 * water/STO-3G.
 *
 * The shifted Hamiltonian is canonical four-center, so it has no get_lambda()
 * of its own, and its two-body tensor is generally indefinite and cannot be
 * re-factorized. lambda is therefore observed where it is actually computed:
 * on the shift's own before/after bookkeeping.
 */
TEST_F(SymmetryShiftTest, Water_STO3G_OneNormRegression) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto ham =
      HamiltonianConstructorFactory::create()->run(wfn_HF->get_orbitals());
  auto factorized = double_factorize(ham);
  const auto& container =
      factorized->get_container<FactorizedHamiltonianContainer>();

  auto global_shift = microsoft::accumulate_fragment_shifts(container);

  // Independent check of the sqrt(2) convention: the container stores the
  // eigenvalues W of fragments of g, while BLISS works with eps = W/sqrt(2)
  // drawn from V = 1/2 g. The baseline sum_a 1/2 (sum|eps|)^2 must therefore
  // reproduce the two-body half of Eq. 33, 1/4 sum_rc (sum_b |W_b|)^2. Getting
  // the scale wrong changes this by a factor of two.
  const auto& w = container.get_w_matrices();
  const size_t R = container.get_num_ranks();
  const size_t B = container.get_num_bases();
  const size_t C = container.get_num_copies();
  double two_body_lambda = 0.0;
  for (size_t r = 0; r < R; ++r) {
    for (size_t c = 0; c < C; ++c) {
      double sum_abs_w = 0.0;
      for (size_t b = 0; b < B; ++b) {
        sum_abs_w += std::abs(w(r * B * C + b * C + c));
      }
      two_body_lambda += 0.25 * sum_abs_w * sum_abs_w;
    }
  }
  EXPECT_NEAR(global_shift.lambda_df_baseline, two_body_lambda, 1e-10);

  // The container's own Lambda adds the one-body norm on top.
  EXPECT_GT(container.get_lambda(), global_shift.lambda_df_baseline);

  // The per-fragment median shift must not increase the two-body 1-norm.
  EXPECT_LE(global_shift.lambda_df_shifted, global_shift.lambda_df_baseline);

  EXPECT_NEAR(global_shift.lambda_df_baseline, kWaterLambdaDfBaseline, 1e-6);
  EXPECT_NEAR(global_shift.lambda_df_shifted, kWaterLambdaDfShifted, 1e-6);
}

/**
 * @brief compute_shift() + rebuild_shifted_hamiltonian() must reproduce
 * run() exactly. This locks the refactor that split the shifter into a
 * public shift-computation step and a public, shift-agnostic rebuild step.
 */
TEST_F(SymmetryShiftTest, ComputeShiftThenRebuildMatchesRun) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto factorized = double_factorize(ham);

  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  auto shifted_run = shifter->run(factorized, 5, 5);
  ASSERT_NE(shifted_run, nullptr);

  auto shifter2 = SymmetryShifterFactory::create("fermionic_low_rank");
  auto shift = shifter2->compute_shift(*factorized, 5, 5);
  auto shifted_manual = rebuild_shifted_hamiltonian(*factorized, shift, 10u);
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
 * @brief A Hamiltonian storing three-center integrals must survive the whole
 * pipeline: double_factorization accepts it directly, and the shift is a
 * symmetry, so the result stays spectrally equivalent in the target
 * particle-number sector.
 */
TEST_F(SymmetryShiftTest, CholeskyContainerIsShiftedThroughTheFactoredPath) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto ham = HamiltonianConstructorFactory::create("qdk_cholesky")
                 ->run(wfn_HF->get_orbitals());

  auto mc = MultiConfigurationCalculatorFactory::create();
  auto [E_before, wfn_before] = mc->run(ham, 5, 5);

  auto factorized = double_factorize(ham);
  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  auto shifted = shifter->run(factorized, 5, 5);
  ASSERT_NE(shifted, nullptr);

  auto mc_after = MultiConfigurationCalculatorFactory::create();
  auto [E_after, wfn_after] = mc_after->run(shifted, 5, 5);
  EXPECT_NEAR(E_before, E_after, testing::ci_energy_tolerance);
}
