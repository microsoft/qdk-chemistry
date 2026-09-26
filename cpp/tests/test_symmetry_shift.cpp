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

/// Independent oracle for the two-body correction dg that the shift
/// (mu2, xi) adds to the raw tensor g:
///   dg_ijkl = -2*mu2*d_ij*d_kl - xi_ij*d_kl - d_ij*xi_kl
/// Written out from the closed form with no reference to the factorization,
/// so it can catch a sign, scale or index-order error in the production path
/// that folds the shift into the fragment eigenvalues. `g` is the flattened
/// tensor ((i*norb+j)*norb+k)*norb+l.
void add_two_body_correction(Eigen::VectorXd& g, Eigen::Index norb, double mu2,
                             const Eigen::MatrixXd& xi) {
  const auto index = [norb](Eigen::Index i, Eigen::Index j, Eigen::Index k,
                            Eigen::Index l) {
    return ((i * norb + j) * norb + k) * norb + l;
  };

  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index j = 0; j < norb; ++j) {
      for (Eigen::Index k = 0; k < norb; ++k) {
        for (Eigen::Index l = 0; l < norb; ++l) {
          double dg = -xi(i, j) * (k == l ? 1.0 : 0.0) -
                      (i == j ? 1.0 : 0.0) * xi(k, l);
          if (i == j && k == l) {
            dg -= 2.0 * mu2;
          }
          g[index(i, j, k, l)] += dg;
        }
      }
    }
  }
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
 * energies agree to within the standard CI energy tolerance.
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

  // The shift is absorbed into the fragment eigenvalues, so what comes back is
  // still a factorization.
  EXPECT_TRUE(
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

  const auto accumulation = microsoft::accumulate_fragment_shifts(container);

  EXPECT_TRUE(accumulation.coulomb.isZero());
  EXPECT_TRUE(accumulation.exchange.isZero());
  EXPECT_EQ(accumulation.xi.rows(), static_cast<Eigen::Index>(norb));
  EXPECT_EQ(accumulation.xi.cols(), static_cast<Eigen::Index>(norb));
  EXPECT_TRUE(accumulation.xi.isZero());
  EXPECT_DOUBLE_EQ(accumulation.mu2, 0.0);
}

/**
 * @brief Regression test pinning the fermionic 1-norm reduction achieved on
 * water/STO-3G.
 *
 * Observed on the shift's own before/after bookkeeping; see
 * Water_STO3G_ShiftedLambdaClosure for the matching check that the shifted
 * Hamiltonian itself reports these numbers.
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

  // norb is 7 here, so B is odd and the median interval is a point.
  const auto accumulation = microsoft::accumulate_fragment_shifts(container);

  // Independent check of the sqrt(2) convention: the container stores the
  // eigenvalues W of fragments of g, while BLISS works with eps = W/sqrt(2)
  // drawn from V = 1/2 g. The baseline sum_a 1/2 (sum|eps|)^2 must therefore
  // reproduce the two-body half of Eq. 17, 1/4 sum_r (sum_b |W_b|)^2. Getting
  // the scale wrong changes this by a factor of two.
  const auto& w = container.get_w_matrices();
  const size_t R = container.get_num_ranks();
  const size_t B = container.get_num_bases();
  double two_body_lambda = 0.0;
  for (size_t r = 0; r < R; ++r) {
    double sum_abs_w = 0.0;
    for (size_t b = 0; b < B; ++b) {
      sum_abs_w += std::abs(w(r * B + b));
    }
    two_body_lambda += 0.25 * sum_abs_w * sum_abs_w;
  }
  // Relative: lambda is extensive, so a fixed absolute bound would turn into
  // a size limit once the accumulation error grows with the system.
  EXPECT_NEAR(accumulation.lambda_df_baseline, two_body_lambda,
              1e-12 * two_body_lambda);

  // The container's own Lambda adds the one-body norm on top.
  EXPECT_GT(container.get_lambda(), accumulation.lambda_df_baseline);

  // The per-fragment median shift must not increase the two-body 1-norm.
  EXPECT_LE(accumulation.lambda_df_shifted, accumulation.lambda_df_baseline);

  EXPECT_NEAR(accumulation.lambda_df_baseline, kWaterLambdaDfBaseline, 1e-6);
  EXPECT_NEAR(accumulation.lambda_df_shifted, kWaterLambdaDfShifted, 1e-6);
}

/**
 * @brief The decisive test for the factorized output: absorbing the shift into
 * the fragment eigenvalues must reproduce the integrals that the closed-form
 * definition of the shift prescribes.
 *
 * add_two_body_correction() is the independent oracle, folding dg into
 * the original tensor with no reference to the factorization. If the in-place
 * shift (W~ = W - sqrt(2)*phi, U untouched) were wrong in sign, scale or index
 * order, the two would disagree.
 */
TEST_F(SymmetryShiftTest, ShiftedFactorizationReproducesDenseShiftedIntegrals) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto ham =
      HamiltonianConstructorFactory::create()->run(wfn_HF->get_orbitals());
  auto factorized = double_factorize(ham);

  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  auto shifted = shifter->run(factorized, 5, 5);
  ASSERT_NE(shifted, nullptr);
  ASSERT_TRUE(shifted->has_container_type<FactorizedHamiltonianContainer>());

  ASSERT_TRUE(shifter->last_shift().has_value());
  const SymmetryShiftCoeffs shift = *shifter->last_shift();

  auto [h0, h0_beta] = factorized->get_one_body_integrals();
  (void)h0_beta;
  const Eigen::Index norb = h0.rows();
  constexpr double ne = 10.0;

  Eigen::MatrixXd h_expected = h0 + (ne - 1.0) * shift.xi;
  h_expected.diagonal().array() -= (shift.mu1 + shift.mu2);

  auto [g0, g0_ab, g0_bb] = factorized->get_two_body_integrals();
  (void)g0_ab;
  (void)g0_bb;
  Eigen::VectorXd g_expected = g0;
  add_two_body_correction(g_expected, norb, shift.mu2, shift.xi);

  auto [h_got, h_got_beta] = shifted->get_one_body_integrals();
  (void)h_got_beta;
  EXPECT_TRUE(h_got.isApprox(h_expected, 1e-12));

  auto [g_got, g_got_ab, g_got_bb] = shifted->get_two_body_integrals();
  (void)g_got_ab;
  (void)g_got_bb;
  EXPECT_LT((g_got - g_expected).cwiseAbs().maxCoeff(), 1e-10);

  EXPECT_NEAR(
      shifted->get_core_energy(),
      factorized->get_core_energy() + shift.mu1 * ne + shift.mu2 * ne * ne,
      1e-12);
}

/**
 * @brief The shift must touch nothing but the fragment eigenvalues, which is
 * what makes the output usable by a block encoding built for the input.
 */
TEST_F(SymmetryShiftTest, ShiftPreservesFactorizationStructure) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto ham =
      HamiltonianConstructorFactory::create()->run(wfn_HF->get_orbitals());
  auto factorized = double_factorize(ham);
  const auto& before =
      factorized->get_container<FactorizedHamiltonianContainer>();

  auto shifted = SymmetryShifterFactory::create("fermionic_low_rank")
                     ->run(factorized, 5, 5);
  const auto& after = shifted->get_container<FactorizedHamiltonianContainer>();

  EXPECT_EQ(after.get_num_ranks(), before.get_num_ranks());
  EXPECT_EQ(after.get_num_bases(), before.get_num_bases());
  EXPECT_EQ(after.get_num_copies(), before.get_num_copies());
  EXPECT_TRUE(after.get_u_matrices().isApprox(before.get_u_matrices(), 1e-15));
  EXPECT_TRUE(after.get_wb_matrix().isZero(0.0));

  // The eigenvalues did move, or the test would be vacuous.
  EXPECT_FALSE(after.get_w_matrices().isApprox(before.get_w_matrices(), 1e-12));

  // Within a fragment every eigenvalue shifts by the SAME amount, because the
  // shift is -phi_r * N_hat and N_hat is the fragment's own occupation sum.
  const Eigen::VectorXd delta =
      after.get_w_matrices() - before.get_w_matrices();
  const size_t R = before.get_num_ranks();
  const size_t B = before.get_num_bases();
  for (size_t r = 0; r < R; ++r) {
    const double reference = delta(r * B);
    for (size_t b = 0; b < B; ++b) {
      EXPECT_NEAR(delta(r * B + b), reference, 1e-12);
    }
  }
}

/**
 * @brief Because the output is still a factorization, it reports its own
 * fermionic 1-norm -- and that number must be the one the shifter's internal
 * bookkeeping claimed. A dense output could not close this loop.
 */
TEST_F(SymmetryShiftTest, Water_STO3G_ShiftedLambdaClosure) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  auto ham =
      HamiltonianConstructorFactory::create()->run(wfn_HF->get_orbitals());
  auto factorized = double_factorize(ham);
  const auto& before =
      factorized->get_container<FactorizedHamiltonianContainer>();
  const auto accumulation = microsoft::accumulate_fragment_shifts(before);

  auto shifted = SymmetryShifterFactory::create("fermionic_low_rank")
                     ->run(factorized, 5, 5);
  const auto& after = shifted->get_container<FactorizedHamiltonianContainer>();

  // Two-body half of Eq. 17, recomputed from the shifted eigenvalues alone.
  const Eigen::VectorXd& w = after.get_w_matrices();
  const size_t R = after.get_num_ranks();
  const size_t B = after.get_num_bases();
  double two_body_lambda = 0.0;
  for (size_t r = 0; r < R; ++r) {
    double sum_abs_w = 0.0;
    for (size_t b = 0; b < B; ++b) {
      sum_abs_w += std::abs(w(r * B + b));
    }
    two_body_lambda += 0.25 * sum_abs_w * sum_abs_w;
  }
  EXPECT_NEAR(two_body_lambda, accumulation.lambda_df_shifted,
              1e-12 * accumulation.lambda_df_shifted);
  EXPECT_LT(after.get_lambda(), before.get_lambda());
}

/**
 * @brief Absorbing -phi*N_hat into the fragment eigenvalues is only valid when
 * sum_b n_b^r really is N_hat, i.e. when U^r is a complete orthogonal
 * rotation. Otherwise the reported 1-norms would not describe the shifted
 * operator either, so the internal "did lambda drop?" guard could not catch it.
 */
TEST_F(SymmetryShiftTest, RejectsFactorizationWithoutCompleteRotations) {
  constexpr size_t norb = 4;
  constexpr size_t C = 1;

  auto make_hamiltonian = [&](size_t B, const Eigen::VectorXd& u) {
    auto container = std::make_unique<FactorizedHamiltonianContainer>(
        Eigen::MatrixXd::Identity(norb, norb), u,
        Eigen::VectorXd::Ones(1 * B * C), Eigen::MatrixXd::Zero(1, C),
        std::make_shared<qdk::chemistry::data::ModelOrbitals>(norb), 0.0,
        Eigen::MatrixXd::Zero(0, 0));
    return std::make_shared<qdk::chemistry::data::Hamiltonian>(
        std::move(container));
  };

  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");

  // Unit-norm rows (so the container accepts them) that are not orthogonal.
  Eigen::VectorXd u_skewed = Eigen::VectorXd::Zero(norb * norb);
  for (size_t b = 0; b < norb; ++b) {
    u_skewed(b * norb + 0) = 1.0;
  }
  EXPECT_THROW(shifter->run(make_hamiltonian(norb, u_skewed), 2, 2),
               std::invalid_argument);

  // Orthonormal but incomplete: B < norb, so sum_b n_b is a projector.
  constexpr size_t B_small = 2;
  Eigen::VectorXd u_partial = Eigen::VectorXd::Zero(B_small * norb);
  for (size_t b = 0; b < B_small; ++b) {
    u_partial(b * norb + b) = 1.0;
  }
  auto shifter2 = SymmetryShifterFactory::create("fermionic_low_rank");
  EXPECT_THROW(shifter2->run(make_hamiltonian(B_small, u_partial), 2, 2),
               std::invalid_argument);
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

/**
 * @brief The two 1-norms are minimized sequentially, not jointly, so the
 * total can come out worse than the baseline. When it does, the shifter must
 * fall back to a zero shift and leave the Hamiltonian untouched rather than
 * ship a regression.
 *
 * A single fragment with eigenvalues (-3, -3, -3, 3) at half filling is one
 * such case: the total goes from 99 to at least 117.
 */
TEST_F(SymmetryShiftTest, FallsBackToZeroShiftWhenLambdaWouldIncrease) {
  constexpr Eigen::Index norb = 4;

  Eigen::VectorXd u = Eigen::VectorXd::Zero(norb * norb);
  for (Eigen::Index b = 0; b < norb; ++b) {
    u(b * norb + b) = 1.0;
  }
  Eigen::VectorXd w(norb);
  w << -3.0, -3.0, -3.0, 3.0;

  auto container = std::make_unique<FactorizedHamiltonianContainer>(
      Eigen::MatrixXd::Zero(norb, norb), u, w, Eigen::MatrixXd::Zero(1, 1),
      std::make_shared<qdk::chemistry::data::ModelOrbitals>(norb), 0.0,
      Eigen::MatrixXd::Zero(0, 0));
  auto hamiltonian =
      std::make_shared<qdk::chemistry::data::Hamiltonian>(std::move(container));

  auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
  auto shifted = shifter->run(hamiltonian, 4, 4);

  ASSERT_TRUE(shifter->last_shift().has_value());
  const SymmetryShiftCoeffs shift = *shifter->last_shift();
  EXPECT_EQ(shift.mu1, 0.0);
  EXPECT_EQ(shift.mu2, 0.0);
  EXPECT_EQ(shift.xi.cwiseAbs().maxCoeff(), 0.0);

  const auto& after = shifted->get_container<FactorizedHamiltonianContainer>();
  EXPECT_TRUE(after.get_w_matrices().isApprox(w));
  EXPECT_NEAR(shifted->get_core_energy(), hamiltonian->get_core_energy(),
              1e-14);
}

/**
 * @brief A nonzero identity weight wB contributes to the Hamiltonian but is
 * accounted for neither by the contractions nor by the rebuild, so it must be
 * rejected rather than silently dropped.
 */
TEST_F(SymmetryShiftTest, NonzeroIdentityWeightIsRejected) {
  constexpr Eigen::Index norb = 2;
  constexpr Eigen::Index R = 1;

  // A complete orthogonal rotation with one copy per rank, so wB is the only
  // precondition this container violates.
  Eigen::VectorXd u(R * norb * norb);
  u << 1.0, 0.0, 0.0, 1.0;

  auto container = std::make_unique<FactorizedHamiltonianContainer>(
      Eigen::MatrixXd::Identity(norb, norb), u, Eigen::VectorXd::Ones(R * norb),
      Eigen::MatrixXd::Ones(R, 1),
      std::make_shared<qdk::chemistry::data::ModelOrbitals>(norb), 0.0,
      Eigen::MatrixXd::Zero(0, 0));
  auto hamiltonian =
      std::make_shared<qdk::chemistry::data::Hamiltonian>(std::move(container));

  EXPECT_THROW(SymmetryShifterFactory::create("fermionic_low_rank")
                   ->run(hamiltonian, 1, 1),
               std::invalid_argument);
}

/**
 * @brief The implementation reads W with the single-copy stride and emits one
 * phi per rank, so a multi-copy container must be rejected rather than
 * silently read as if the copies were extra bases.
 */
TEST_F(SymmetryShiftTest, MultipleCopiesPerRankAreRejected) {
  constexpr Eigen::Index norb = 2;
  constexpr Eigen::Index R = 1;
  constexpr Eigen::Index C = 2;

  Eigen::VectorXd u(R * norb * norb);
  u << 1.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd w(R * norb * C);
  w << 1.0, 2.0, 3.0, 8.0;

  auto container = std::make_unique<FactorizedHamiltonianContainer>(
      Eigen::MatrixXd::Identity(norb, norb), u, w, Eigen::MatrixXd::Zero(R, C),
      std::make_shared<qdk::chemistry::data::ModelOrbitals>(norb), 0.0,
      Eigen::MatrixXd::Zero(0, 0));
  auto hamiltonian =
      std::make_shared<qdk::chemistry::data::Hamiltonian>(std::move(container));

  EXPECT_THROW(SymmetryShifterFactory::create("fermionic_low_rank")
                   ->run(hamiltonian, 1, 1),
               std::invalid_argument);
}

/**
 * @brief For an even number of bases the Eq. 27 objective is flat across the
 * whole median interval. phi takes the upper endpoint, which is an actual
 * eps_i -- that is what drops a unitary from the one-electron LCU -- and,
 * because the objective is flat, costs nothing in the two-body norm.
 */
TEST_F(SymmetryShiftTest, EvenBasisCountUsesTheMedianIntervalEndpoint) {
  // LiH/STO-3G has 6 orbitals, so the median interval is a real interval.
  auto lih = testing::create_lih_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(lih, 0, 1, "sto-3g");

  auto ham =
      HamiltonianConstructorFactory::create()->run(wfn_HF->get_orbitals());
  auto factorized = double_factorize(ham);
  const auto& container =
      factorized->get_container<FactorizedHamiltonianContainer>();
  ASSERT_EQ(container.get_num_bases() % 2, 0u);

  const auto accumulation = microsoft::accumulate_fragment_shifts(container);

  const auto& w = container.get_w_matrices();
  const size_t R = container.get_num_ranks();
  const size_t B = container.get_num_bases();
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

  double lambda_df_midpoint = 0.0;
  bool saw_real_interval = false;
  for (size_t r = 0; r < R; ++r) {
    Eigen::VectorXd eps(B);
    for (size_t b = 0; b < B; ++b) {
      eps(b) = inv_sqrt2 * w(r * B + b);
    }

    const auto [lo, hi] = microsoft::median_interval(eps);
    saw_real_interval = saw_real_interval || hi > lo;

    const double phi = accumulation.phi(static_cast<Eigen::Index>(r));
    EXPECT_NEAR(phi, hi, 1e-12);
    EXPECT_NEAR((eps.array() - phi).abs().minCoeff(), 0.0, 1e-12);

    const double shifted = (eps.array() - 0.5 * (lo + hi)).abs().sum();
    lambda_df_midpoint += 0.5 * shifted * shifted;
  }
  EXPECT_TRUE(saw_real_interval);

  EXPECT_NEAR(accumulation.lambda_df_shifted, lambda_df_midpoint,
              1e-10 * lambda_df_midpoint);
}
