// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/hamiltonian_regularizer.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/microsoft/flr_bliss/flr_bliss_regularizer.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/utils/hamiltonian_one_norm.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using qdk::chemistry::algorithms::microsoft::flr_bliss::TwoBodyBlissCorrection;

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

  auto norm_before = qdk::chemistry::utils::hamiltonian_one_norm(*ham, 0.0);

  auto regularizer = HamiltonianRegularizerFactory::create("flr_bliss");
  auto shifted_ham = regularizer->run(ham, 5, 5);

  auto norm_after =
      qdk::chemistry::utils::hamiltonian_one_norm(*shifted_ham, 0.0);

  EXPECT_LE(norm_after.total, norm_before.total + testing::numerical_zero_tolerance);
}

namespace {

// Brute-force index contraction of a flattened norb^4 tensor `t`, laid out as
// ((i*norb+j)*norb+k)*norb+l, matching one_electron_shift.cpp's/
// rebuild_hamiltonian.cpp's own two_body_index() helpers.
Eigen::MatrixXd contract_coulomb(const Eigen::VectorXd& t, Eigen::Index norb) {
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(norb, norb);
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index j = 0; j < norb; ++j) {
      double sum = 0.0;
      for (Eigen::Index k = 0; k < norb; ++k) {
        sum += t[((i * norb + j) * norb + k) * norb + k];
      }
      out(i, j) = sum;
    }
  }
  return out;
}

Eigen::MatrixXd contract_exchange(const Eigen::VectorXd& t, Eigen::Index norb) {
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(norb, norb);
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index j = 0; j < norb; ++j) {
      double sum = 0.0;
      for (Eigen::Index k = 0; k < norb; ++k) {
        sum += t[((i * norb + k) * norb + k) * norb + j];
      }
      out(i, j) = sum;
    }
  }
  return out;
}

}  // namespace

// TwoBodyBlissCorrection (utils.hpp) is the shared struct that
// one_electron_shift.cpp and rebuild_hamiltonian.cpp both derive their
// BLISS-shifted two-body contractions from, tested here since it backs the
// flr_bliss regularizer exercised above.
//
// This is the executable version of the "verified by hand" claim documented
// in utils.hpp: TwoBodyBlissCorrection::coulomb_contraction()/
// exchange_contraction() are closed-form shortcuts for contracting
// full_tensor() over one pair of indices. If anyone edits the closed forms
// (used by one_electron_shift.cpp) without correspondingly updating
// full_tensor() (used by rebuild_hamiltonian.cpp), or vice versa, this test
// fails -- preventing the two call sites from silently drifting apart.
class TwoBodyBlissCorrectionTest : public ::testing::Test {};

TEST_F(TwoBodyBlissCorrectionTest, ContractionsMatchBruteForceTensor) {
  const Eigen::Index norb = 4;

  Eigen::MatrixXd xi(norb, norb);
  xi << 0.3, -0.1, 0.05, 0.2, -0.1, 0.4, 0.15, -0.05, 0.05, 0.15, -0.2, 0.1,
      0.2, -0.05, 0.1, 0.25;
  // xi need not be symmetric in general (it multiplies E^m_n, not a
  // symmetric operator), so keep it asymmetric here to exercise the general
  // case.

  const double mu2 = 0.37;
  const TwoBodyBlissCorrection correction{mu2, xi};

  const Eigen::VectorXd full = correction.full_tensor(norb);
  const Eigen::MatrixXd coulomb_from_tensor = contract_coulomb(full, norb);
  const Eigen::MatrixXd exchange_from_tensor = contract_exchange(full, norb);

  const Eigen::MatrixXd coulomb_closed_form =
      correction.coulomb_contraction(norb);
  const Eigen::MatrixXd exchange_closed_form =
      correction.exchange_contraction(norb);

  EXPECT_TRUE(coulomb_closed_form.isApprox(coulomb_from_tensor, 1e-12))
      << "coulomb_contraction() diverged from brute-force contraction of "
         "full_tensor():\n"
      << (coulomb_closed_form - coulomb_from_tensor);
  EXPECT_TRUE(exchange_closed_form.isApprox(exchange_from_tensor, 1e-12))
      << "exchange_contraction() diverged from brute-force contraction of "
         "full_tensor():\n"
      << (exchange_closed_form - exchange_from_tensor);
}

TEST_F(TwoBodyBlissCorrectionTest, ZeroShiftGivesZeroCorrection) {
  const Eigen::Index norb = 3;
  const TwoBodyBlissCorrection correction{0.0, Eigen::MatrixXd::Zero(norb, norb)};

  EXPECT_TRUE(correction.full_tensor(norb).isZero(1e-14));
  EXPECT_TRUE(correction.coulomb_contraction(norb).isZero(1e-14));
  EXPECT_TRUE(correction.exchange_contraction(norb).isZero(1e-14));
}
