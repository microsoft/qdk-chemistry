// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/utils/double_factorization.hpp>
#include <qdk/chemistry/utils/hamiltonian_one_norm.hpp>
#include <tuple>

#include "ut_common.hpp"

using qdk::chemistry::utils::double_factorize;
using qdk::chemistry::utils::double_factorize_three_center;
using qdk::chemistry::utils::DoubleFactorizationMethod;
using qdk::chemistry::utils::hamiltonian_one_norm;
using qdk::chemistry::utils::TwoBodyFragment;

class HamiltonianOneNormTest : public ::testing::Test {};

TEST_F(HamiltonianOneNormTest, WaterSTO3GIsPositiveAndConsistent) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto norm = hamiltonian_one_norm(*ham, 0.0);
  EXPECT_GT(norm.one_body, 0.0);
  EXPECT_GT(norm.two_body, 0.0);
  EXPECT_NEAR(norm.total, norm.one_body + norm.two_body,
              testing::numerical_zero_tolerance);

  // No truncation is the default: calling without an explicit threshold
  // should give the same result as threshold=0.0.
  auto norm_default = hamiltonian_one_norm(*ham);
  EXPECT_NEAR(norm_default.total, norm.total,
              testing::numerical_zero_tolerance);
}

TEST_F(HamiltonianOneNormTest, TruncationNeverIncreasesTwoBodyNorm) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto norm_exact = hamiltonian_one_norm(*ham, 0.0);
  auto norm_truncated = hamiltonian_one_norm(*ham, 1e-2);
  // Truncating fragments removes contributions to the low-rank
  // reconstruction, so the reported two-body 1-norm (computed from the
  // retained fragments only) should not exceed the exact value.
  EXPECT_LE(norm_truncated.two_body,
            norm_exact.two_body + testing::numerical_zero_tolerance);
}

namespace {

/// Reconstruct the flattened g_ijkl tensor from a set of DF fragments:
///   g_ijkl = sum_alpha sign_alpha * sum_pq U_ip U_jp eps_p eps_q U_kq U_lq
Eigen::VectorXd reconstruct(const std::vector<TwoBodyFragment>& fragments,
                            size_t norb) {
  Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  for (const auto& fragment : fragments) {
    // M_ij = sum_p U_ip eps_p U_jp  (i.e. U * diag(eps) * U^T)
    Eigen::MatrixXd M =
        fragment.U * fragment.eps.asDiagonal() * fragment.U.transpose();
    for (size_t i = 0; i < norb; ++i) {
      for (size_t j = 0; j < norb; ++j) {
        for (size_t k = 0; k < norb; ++k) {
          for (size_t l = 0; l < norb; ++l) {
            g[i * norb * norb * norb + j * norb * norb + k * norb + l] +=
                fragment.sign * M(i, j) * M(k, l);
          }
        }
      }
    }
  }
  return g;
}

}  // namespace

// double_factorize() is the fragment decomposition that hamiltonian_one_norm()
// (tested above) uses internally to compute the two-body 1-norm; both are
// tested in this file since they share the same water/STO-3G test fixture.
class DoubleFactorizationTest : public ::testing::Test {};

TEST_F(DoubleFactorizationTest, ExactReconstructionNoTruncation) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto [g_aaaa, g_aabb, g_bbbb] = ham->get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;
  const size_t n =
      static_cast<size_t>(ham->get_orbitals()->get_num_molecular_orbitals());

  // Double-factorize with threshold=0.0 (no truncation): reconstruction
  // should reproduce the original tensor to machine precision.
  auto fragments = double_factorize(g_aaaa, n, 0.0);
  ASSERT_FALSE(fragments.empty());
  Eigen::VectorXd g_reconstructed = reconstruct(fragments, n);
  EXPECT_TRUE(
      g_reconstructed.isApprox(g_aaaa, testing::numerical_zero_tolerance * 100))
      << "Reconstruction max abs diff: "
      << (g_reconstructed - g_aaaa).cwiseAbs().maxCoeff();
}

TEST_F(DoubleFactorizationTest,
       TruncationReducesFragmentCountAndDefaultsToZero) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());
  auto [g_aaaa, g_aabb, g_bbbb] = ham->get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;
  const size_t n =
      static_cast<size_t>(ham->get_orbitals()->get_num_molecular_orbitals());

  // The default threshold argument is 0.0 (no truncation): calling without
  // specifying it should give the same fragment count as threshold=0.0.
  auto fragments_default = double_factorize(g_aaaa, n);
  auto fragments_explicit_zero = double_factorize(g_aaaa, n, 0.0);
  EXPECT_EQ(fragments_default.size(), fragments_explicit_zero.size());

  // A larger threshold should never retain more fragments than a smaller one.
  auto fragments_loose = double_factorize(g_aaaa, n, 1e-2);
  EXPECT_LE(fragments_loose.size(), fragments_explicit_zero.size());
}

// The default method is Cholesky. These tests pin the properties that make the
// Cholesky path a valid drop-in replacement for the eigendecomposition, and
// the one property that legitimately differs between them (lambda_df).
class DoubleFactorizationCholeskyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto water = testing::create_water_structure();
    auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
    auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
    auto hamiltonian_constructor =
        qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
    ham_ = hamiltonian_constructor->run(wfn_HF->get_orbitals());
    std::tie(g_aaaa_, std::ignore, std::ignore) =
        ham_->get_two_body_integrals();
    norb_ =
        static_cast<size_t>(ham_->get_orbitals()->get_num_molecular_orbitals());
  }

  std::shared_ptr<qdk::chemistry::data::Hamiltonian> ham_;
  Eigen::VectorXd g_aaaa_;
  size_t norb_ = 0;
};

TEST_F(DoubleFactorizationCholeskyTest, ReconstructsTensorExactly) {
  auto fragments = double_factorize(g_aaaa_, norb_, 0.0,
                                    DoubleFactorizationMethod::Cholesky);
  ASSERT_FALSE(fragments.empty());
  Eigen::VectorXd g_reconstructed = reconstruct(fragments, norb_);
  EXPECT_TRUE(g_reconstructed.isApprox(g_aaaa_,
                                       testing::numerical_zero_tolerance * 100))
      << "Reconstruction max abs diff: "
      << (g_reconstructed - g_aaaa_).cwiseAbs().maxCoeff();
}

TEST_F(DoubleFactorizationCholeskyTest, IsTheDefaultMethod) {
  auto fragments_default = double_factorize(g_aaaa_, norb_, 0.0);
  auto fragments_cholesky = double_factorize(
      g_aaaa_, norb_, 0.0, DoubleFactorizationMethod::Cholesky);
  ASSERT_EQ(fragments_default.size(), fragments_cholesky.size());
  for (size_t i = 0; i < fragments_default.size(); ++i) {
    EXPECT_NEAR(fragments_default[i].lambda_df, fragments_cholesky[i].lambda_df,
                testing::numerical_zero_tolerance);
  }
}

TEST_F(DoubleFactorizationCholeskyTest, RankIsBoundedBySymmetricPairDimension) {
  // The supermatrix M_(ij),(kl) satisfies M = M^T and is invariant under
  // (ij) -> (ji), so every antisymmetric pair vector lies in its null space
  // and the rank cannot exceed the symmetric-pair dimension.
  auto fragments = double_factorize(g_aaaa_, norb_, 0.0,
                                    DoubleFactorizationMethod::Cholesky);
  EXPECT_LE(fragments.size(), norb_ * (norb_ + 1) / 2);
}

TEST_F(DoubleFactorizationCholeskyTest, RankIsInvariantUnderTensorScaling) {
  // Scaling the tensor rescales the supermatrix but not its rank, so the
  // "lossless" cutoff must be relative to the supermatrix scale.
  auto reference = double_factorize(g_aaaa_, norb_, 0.0,
                                    DoubleFactorizationMethod::Cholesky);
  ASSERT_FALSE(reference.empty());

  for (const double scale : {1.0e-6, 1.0e+6}) {
    Eigen::VectorXd scaled = scale * g_aaaa_;
    auto fragments = double_factorize(scaled, norb_, 0.0,
                                      DoubleFactorizationMethod::Cholesky);
    EXPECT_EQ(fragments.size(), reference.size()) << "scale = " << scale;
    for (const auto& fragment : fragments) {
      EXPECT_EQ(fragment.sign, 1) << "scale = " << scale;
    }
  }
}

TEST_F(DoubleFactorizationCholeskyTest,
       PhysicalIntegralsGiveOnlyPositiveSigns) {
  // Raw electron-repulsion integrals form a positive semi-definite
  // supermatrix, so Cholesky never needs a negative fragment and never falls
  // back to the eigensolver.
  auto fragments = double_factorize(g_aaaa_, norb_, 0.0,
                                    DoubleFactorizationMethod::Cholesky);
  ASSERT_FALSE(fragments.empty());
  for (const auto& fragment : fragments) {
    EXPECT_EQ(fragment.sign, 1);
  }
}

TEST_F(DoubleFactorizationCholeskyTest, IndefiniteInputFallsBackToEigen) {
  // Negating the tensor makes the supermatrix negative definite. Cholesky
  // cannot represent it, so double_factorize() must warn and fall back to the
  // eigendecomposition, which still reconstructs it exactly using negative
  // fragments.
  Eigen::VectorXd g_negated = -g_aaaa_;
  auto fragments = double_factorize(g_negated, norb_, 0.0,
                                    DoubleFactorizationMethod::Cholesky);
  ASSERT_FALSE(fragments.empty());

  bool has_negative_fragment = false;
  for (const auto& fragment : fragments) {
    if (fragment.sign == -1) {
      has_negative_fragment = true;
    }
  }
  EXPECT_TRUE(has_negative_fragment)
      << "Expected the eigen fallback to produce negative fragments";

  Eigen::VectorXd g_reconstructed = reconstruct(fragments, norb_);
  EXPECT_TRUE(g_reconstructed.isApprox(g_negated,
                                       testing::numerical_zero_tolerance * 100))
      << "Reconstruction max abs diff: "
      << (g_reconstructed - g_negated).cwiseAbs().maxCoeff();
}

TEST_F(DoubleFactorizationCholeskyTest,
       BothMethodsReconstructButLambdaDiffers) {
  // Both methods reconstruct the tensor exactly, and sum_r ||A_r||_F^2 =
  // tr(M) is gauge invariant, but lambda = sum_r ||A_r||_*^2 / 2 is not.
  auto cholesky = double_factorize(g_aaaa_, norb_, 0.0,
                                   DoubleFactorizationMethod::Cholesky);
  auto eigen =
      double_factorize(g_aaaa_, norb_, 0.0, DoubleFactorizationMethod::Eigen);
  ASSERT_FALSE(cholesky.empty());
  ASSERT_FALSE(eigen.empty());

  EXPECT_TRUE(reconstruct(cholesky, norb_)
                  .isApprox(g_aaaa_, testing::numerical_zero_tolerance * 100));
  EXPECT_TRUE(reconstruct(eigen, norb_)
                  .isApprox(g_aaaa_, testing::numerical_zero_tolerance * 100));

  // Gauge-invariant quantity: the total squared Frobenius norm, i.e. the sum
  // of squared fragment eigenvalues, equals tr(M) for both methods.
  auto frobenius_squared = [](const std::vector<TwoBodyFragment>& fragments) {
    double total = 0.0;
    for (const auto& fragment : fragments) {
      total += fragment.sign * fragment.eps.squaredNorm();
    }
    return total;
  };
  EXPECT_NEAR(frobenius_squared(cholesky), frobenius_squared(eigen),
              testing::numerical_zero_tolerance * 100);

  auto one_norm = [](const std::vector<TwoBodyFragment>& fragments) {
    double total = 0.0;
    for (const auto& fragment : fragments) {
      total += fragment.lambda_df;
    }
    return total;
  };
  // Not an equality: lambda is gauge dependent, so the two methods report
  // genuinely different two-body 1-norms for the same operator.
  EXPECT_GT(one_norm(eigen), 0.0);
  EXPECT_GT(one_norm(cholesky), 0.0);
}

TEST_F(DoubleFactorizationCholeskyTest, HamiltonianOneNormAcceptsBothMethods) {
  auto norm_cholesky =
      hamiltonian_one_norm(*ham_, 0.0, DoubleFactorizationMethod::Cholesky);
  auto norm_eigen =
      hamiltonian_one_norm(*ham_, 0.0, DoubleFactorizationMethod::Eigen);
  auto norm_default = hamiltonian_one_norm(*ham_, 0.0);

  // The one-body term does not depend on the two-body factorization.
  EXPECT_NEAR(norm_cholesky.one_body, norm_eigen.one_body,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(norm_default.two_body, norm_cholesky.two_body,
              testing::numerical_zero_tolerance);
  EXPECT_GT(norm_eigen.two_body, 0.0);
}

// ---------------------------------------------------------------------------
// Three-center (Cholesky container) path: the same decomposition without ever
// materializing the norb^4 tensor.
// ---------------------------------------------------------------------------

class ThreeCenterDoubleFactorizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto water = testing::create_water_structure();
    auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
    auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
    ham_ = qdk::chemistry::algorithms::HamiltonianConstructorFactory::create(
               "qdk_cholesky")
               ->run(wfn_HF->get_orbitals());
    ASSERT_TRUE(ham_->has_container_type<
                qdk::chemistry::data::CholeskyHamiltonianContainer>());
    l_alpha_ = ham_->get_container<
                       qdk::chemistry::data::CholeskyHamiltonianContainer>()
                   .get_three_center_integrals()
                   .first;
    std::tie(g_aaaa_, std::ignore, std::ignore) =
        ham_->get_two_body_integrals();
    norb_ =
        static_cast<size_t>(ham_->get_orbitals()->get_num_molecular_orbitals());
  }

  std::shared_ptr<qdk::chemistry::data::Hamiltonian> ham_;
  Eigen::MatrixXd l_alpha_;
  Eigen::VectorXd g_aaaa_;
  size_t norb_ = 0;
};

TEST_F(ThreeCenterDoubleFactorizationTest, ReconstructsTensorExactly) {
  const Eigen::MatrixXd half = l_alpha_ / std::sqrt(2.0);
  const Eigen::VectorXd expected = 0.5 * g_aaaa_;

  for (auto method : {DoubleFactorizationMethod::Cholesky,
                      DoubleFactorizationMethod::Eigen}) {
    auto fragments = double_factorize_three_center(half, norb_, 0.0, method);
    ASSERT_FALSE(fragments.empty());
    auto reconstructed = reconstruct(fragments, norb_);
    EXPECT_LT((reconstructed - expected).cwiseAbs().maxCoeff(), 1e-9);
  }
}

TEST_F(ThreeCenterDoubleFactorizationTest, FragmentsArePositiveAndOrdered) {
  auto fragments =
      double_factorize_three_center(l_alpha_ / std::sqrt(2.0), norb_, 0.0);
  ASSERT_FALSE(fragments.empty());
  for (size_t n = 0; n < fragments.size(); ++n) {
    EXPECT_DOUBLE_EQ(fragments[n].sign, 1.0);
    if (n > 0) {
      EXPECT_LE(fragments[n].lambda_df, fragments[n - 1].lambda_df);
    }
  }
}

TEST_F(ThreeCenterDoubleFactorizationTest, EigenMethodRemovesRedundantColumns) {
  const Eigen::MatrixXd half = l_alpha_ / std::sqrt(2.0);
  auto eigen_fragments = double_factorize_three_center(
      half, norb_, 0.0, DoubleFactorizationMethod::Eigen);
  // The aux index is sized by the AO basis, so it can only ever exceed the
  // symmetric-pair dimension of the (possibly smaller) orbital space.
  EXPECT_LE(eigen_fragments.size(), norb_ * (norb_ + 1) / 2);
  EXPECT_LE(eigen_fragments.size(), static_cast<size_t>(l_alpha_.cols()));
}

TEST_F(ThreeCenterDoubleFactorizationTest,
       MeanFieldContractionsMatchDensePath) {
  auto [coulomb_dense, exchange_dense] =
      qdk::chemistry::utils::mean_field_contractions(g_aaaa_, norb_);
  auto [coulomb_factored, exchange_factored] =
      qdk::chemistry::utils::mean_field_contractions_three_center(l_alpha_,
                                                                  norb_);
  EXPECT_LT((coulomb_factored - coulomb_dense).cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_LT((exchange_factored - exchange_dense).cwiseAbs().maxCoeff(), 1e-9);
}

TEST_F(ThreeCenterDoubleFactorizationTest, OneNormUsesTheFactoredPath) {
  auto norm = hamiltonian_one_norm(*ham_, 0.0);
  EXPECT_GT(norm.one_body, 0.0);
  EXPECT_GT(norm.two_body, 0.0);
  EXPECT_NEAR(norm.total, norm.one_body + norm.two_body,
              testing::numerical_zero_tolerance);

  // The one-body term is a plain contraction of the same tensor, so it must
  // agree with the dense container. lambda_2e is gauge dependent and is not
  // expected to.
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto dense =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create()->run(
          wfn_HF->get_orbitals());
  EXPECT_NEAR(norm.one_body, hamiltonian_one_norm(*dense, 0.0).one_body, 1e-6);
}
