// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

constexpr double kReconstructionTolerance = 1e-10;

/// Build a random two-body tensor with a negative entry in @p signs.
Eigen::VectorXd make_two_body(std::size_t norb,
                              const std::vector<double>& signs, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<Eigen::MatrixXd> factors;
  factors.reserve(signs.size());
  for (std::size_t k = 0; k < signs.size(); ++k) {
    Eigen::MatrixXd factor(norb, norb);
    for (std::size_t p = 0; p < norb; ++p) {
      for (std::size_t q = 0; q <= p; ++q) {
        const double value = dist(rng);
        factor(p, q) = value;
        factor(q, p) = value;
      }
    }
    factors.push_back(std::move(factor));
  }

  Eigen::VectorXd two_body = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  for (std::size_t p = 0; p < norb; ++p) {
    for (std::size_t q = 0; q < norb; ++q) {
      for (std::size_t r = 0; r < norb; ++r) {
        for (std::size_t s = 0; s < norb; ++s) {
          double value = 0.0;
          for (std::size_t k = 0; k < factors.size(); ++k) {
            value += signs[k] * factors[k](p, q) * factors[k](r, s);
          }
          two_body[((p * norb + q) * norb + r) * norb + s] = value;
        }
      }
    }
  }
  return two_body;
}

Eigen::MatrixXd make_one_body(std::size_t norb, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Eigen::MatrixXd one_body(norb, norb);
  for (std::size_t p = 0; p < norb; ++p) {
    for (std::size_t q = 0; q <= p; ++q) {
      const double value = dist(rng);
      one_body(p, q) = value;
      one_body(q, p) = value;
    }
  }
  return one_body;
}

std::shared_ptr<Hamiltonian> make_hamiltonian(std::size_t norb,
                                              const Eigen::VectorXd& two_body,
                                              double core_energy = -1.25) {
  return std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          make_one_body(norb, 7), two_body,
          testing::create_test_orbitals(static_cast<int>(norb),
                                        static_cast<int>(norb)),
          core_energy, Eigen::MatrixXd::Zero(0, 0)));
}

std::shared_ptr<Hamiltonian> make_unrestricted_hamiltonian(std::size_t norb) {
  const auto two_body = make_two_body(norb, {1.0, 1.0}, 3);
  return std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          make_one_body(norb, 7), make_one_body(norb, 11), two_body, two_body,
          two_body,
          testing::create_test_orbitals(static_cast<int>(norb),
                                        static_cast<int>(norb)),
          0.0, Eigen::MatrixXd::Zero(0, 0), Eigen::MatrixXd::Zero(0, 0)));
}

const FactorizedHamiltonianContainer& as_factorized(
    const std::shared_ptr<Hamiltonian>& hamiltonian) {
  return hamiltonian->get_container<FactorizedHamiltonianContainer>();
}

}  // namespace

TEST(DoubleFactorizerTest, MetaDataAndFactoryRegistration) {
  auto factorizer = DoubleFactorizerFactory::create("eigen_decomposition");
  ASSERT_NE(factorizer, nullptr);
  EXPECT_EQ(factorizer->type_name(), "double_factorizer");
  EXPECT_EQ(factorizer->name(), "eigen_decomposition");
  EXPECT_TRUE(factorizer->settings().has("truncation_threshold"));
  EXPECT_TRUE(factorizer->settings().has("symmetry_tolerance"));

  const auto available = DoubleFactorizerFactory::available();
  EXPECT_NE(
      std::find(available.begin(), available.end(), "eigen_decomposition"),
      available.end());
  EXPECT_THROW(DoubleFactorizerFactory::create("nonexistent_factorizer"),
               std::runtime_error);
}

TEST(DoubleFactorizerTest, RejectsInvalidInput) {
  constexpr std::size_t norb = 4;
  auto hamiltonian = make_hamiltonian(norb, make_two_body(norb, {1.0}, 31));
  auto factorizer = DoubleFactorizerFactory::create("eigen_decomposition");

  EXPECT_THROW(factorizer->settings().set("truncation_threshold", -1.0),
               std::exception);
  EXPECT_THROW(eigen_decompose_two_body(Eigen::VectorXd::Zero(10), norb),
               std::invalid_argument);

  // The free function is public API and is reachable without the settings
  // BoundConstraint, so it has to reject these arguments itself.
  EXPECT_THROW(eigen_decompose_two_body(Eigen::VectorXd(), 0),
               std::invalid_argument);
  const Eigen::VectorXd tensor = make_two_body(norb, {1.0}, 31);
  EXPECT_THROW(eigen_decompose_two_body(tensor, norb, -1.0),
               std::invalid_argument);
  EXPECT_THROW(eigen_decompose_two_body(
                   tensor, norb, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  // A negative or NaN threshold compares false against every eigenvalue, so
  // without the guard it would silently retain the whole decomposition
  // rather than fail.
  EXPECT_FALSE(eigen_decompose_two_body(tensor, norb, 0.0).empty());

  // A non-finite entry defeats every later guard: NaN compares false against
  // the symmetry tolerance, and it makes the sort comparator |a| > |b| false
  // in both directions, which is undefined behavior in std::sort.
  Eigen::VectorXd with_nan = make_two_body(norb, {1.0}, 31);
  with_nan[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(eigen_decompose_two_body(with_nan, norb), std::invalid_argument);
  Eigen::VectorXd with_inf = make_two_body(norb, {1.0}, 31);
  with_inf[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(eigen_decompose_two_body(with_inf, norb), std::invalid_argument);

  EXPECT_THROW(factorizer->run(nullptr), std::invalid_argument);
  EXPECT_THROW(factorizer->run(make_unrestricted_hamiltonian(norb)),
               std::invalid_argument);
  auto truncating = DoubleFactorizerFactory::create("eigen_decomposition");
  truncating->settings().set("truncation_threshold", 1e6);
  EXPECT_THROW(truncating->run(hamiltonian), std::invalid_argument);
}

TEST(DoubleFactorizerTest, RejectsAsymmetricTwoBodyIntegrals) {
  constexpr std::size_t norb = 4;
  const auto flat = [](std::size_t p, std::size_t q, std::size_t r,
                       std::size_t s) {
    return ((p * norb + q) * norb + r) * norb + s;
  };

  // A tensor that breaks p<->q only. Perturbing g[0,1,2,2] and its
  // (pq)<->(rs) image g[2,2,0,1] keeps the supermatrix symmetric, so this is
  // caught only by the second generator.
  Eigen::VectorXd pq_broken = make_two_body(norb, {1.0}, 31);
  pq_broken[flat(0, 1, 2, 2)] += 1.0;
  pq_broken[flat(2, 2, 0, 1)] += 1.0;
  EXPECT_THROW(eigen_decompose_two_body(pq_broken, norb),
               std::invalid_argument);

  // A tensor that breaks (pq)<->(rs) only. Perturbing the four elements of
  // the p<->q / r<->s orbit of (0,1),(2,3) leaves both index-pair swaps
  // intact, so this is caught only by the first generator.
  Eigen::VectorXd rs_broken = make_two_body(norb, {1.0}, 31);
  rs_broken[flat(0, 1, 2, 3)] += 1.0;
  rs_broken[flat(1, 0, 2, 3)] += 1.0;
  rs_broken[flat(0, 1, 3, 2)] += 1.0;
  rs_broken[flat(1, 0, 3, 2)] += 1.0;
  EXPECT_THROW(eigen_decompose_two_body(rs_broken, norb),
               std::invalid_argument);

  // Both are accepted once the tolerance is loosened past the perturbation,
  // so the rejections above come from the symmetry check and not from some
  // other guard.
  EXPECT_NO_THROW(eigen_decompose_two_body(pq_broken, norb,
                                           DEFAULT_TRUNCATION_THRESHOLD, 1e3));
  EXPECT_NO_THROW(eigen_decompose_two_body(rs_broken, norb,
                                           DEFAULT_TRUNCATION_THRESHOLD, 1e3));

  // The tolerance is itself validated, like truncation_threshold.
  const Eigen::VectorXd symmetric = make_two_body(norb, {1.0}, 31);
  EXPECT_THROW(eigen_decompose_two_body(symmetric, norb,
                                        DEFAULT_TRUNCATION_THRESHOLD, -1.0),
               std::invalid_argument);
  EXPECT_THROW(
      eigen_decompose_two_body(symmetric, norb, DEFAULT_TRUNCATION_THRESHOLD,
                               std::numeric_limits<double>::quiet_NaN()),
      std::invalid_argument);

  // A tensor assembled from symmetric factors survives a tolerance far
  // tighter than the default, so the check has real margin on valid input.
  EXPECT_NO_THROW(eigen_decompose_two_body(
      symmetric, norb, DEFAULT_TRUNCATION_THRESHOLD, 1e-14));
}

TEST(DoubleFactorizerTest, EigenDecomposeFragmentsReconstructTensor) {
  constexpr std::size_t norb = 3;
  const auto two_body = make_two_body(norb, {1.0, -1.0}, 17);
  const auto fragments = eigen_decompose_two_body(two_body, norb);
  ASSERT_FALSE(fragments.empty());

  Eigen::VectorXd reconstructed =
      Eigen::VectorXd::Zero(norb * norb * norb * norb);
  for (const auto& fragment : fragments) {
    const Eigen::MatrixXd m =
        fragment.U * fragment.eps.asDiagonal() * fragment.U.transpose();
    for (std::size_t p = 0; p < norb; ++p) {
      for (std::size_t q = 0; q < norb; ++q) {
        for (std::size_t r = 0; r < norb; ++r) {
          for (std::size_t s = 0; s < norb; ++s) {
            reconstructed[((p * norb + q) * norb + r) * norb + s] +=
                fragment.sign * m(p, q) * m(r, s);
          }
        }
      }
    }
  }

  for (Eigen::Index i = 0; i < two_body.size(); ++i) {
    EXPECT_NEAR(reconstructed[i], two_body[i], kReconstructionTolerance);
  }
}

TEST(DoubleFactorizerTest, FragmentLambdaMatchesContainerLambda) {
  // lambda_df is a per-fragment share of the same block-encoding 1-norm that
  // get_lambda() reports, so the fragments must sum to the container's
  // two-body part. Asserting lambda_df against its own defining formula would
  // only restate the code; this pins the shared 1/4 convention (Eq. 34)
  // across both call sites.
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, -1.0}, 41);
  const auto fragments = eigen_decompose_two_body(two_body, norb);
  ASSERT_FALSE(fragments.empty());

  double fragment_lambda_sum = 0.0;
  for (const auto& fragment : fragments) {
    fragment_lambda_sum += fragment.lambda_df;
  }

  auto factorizer = DoubleFactorizerFactory::create("eigen_decomposition");
  auto factorized = factorizer->run(make_hamiltonian(norb, two_body));
  ASSERT_NE(factorized, nullptr);
  const auto& container = as_factorized(factorized);

  // get_lambda() adds the one-body term, which carries no fragment share.
  const Eigen::MatrixXd h1p = container.get_h1_prime();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h1p);
  ASSERT_EQ(solver.info(), Eigen::Success);
  const double one_body_norm = solver.eigenvalues().array().abs().sum();

  EXPECT_NEAR(fragment_lambda_sum, container.get_lambda() - one_body_norm,
              kReconstructionTolerance);
}

TEST(DoubleFactorizerTest, EigenDecomposeSortsFragmentsByDecreasingWeight) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, -1.0, 1.0}, 23);
  const auto fragments = eigen_decompose_two_body(two_body, norb);
  ASSERT_GE(fragments.size(), 2u);

  for (std::size_t r = 1; r < fragments.size(); ++r) {
    EXPECT_LE(fragments[r].eps.squaredNorm(),
              fragments[r - 1].eps.squaredNorm() + 1e-12);
  }
}

TEST(DoubleFactorizerTest, RepresentsNegativeFragments) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, -1.0, 1.0}, 5);
  auto hamiltonian = make_hamiltonian(norb, two_body);

  auto factorized =
      DoubleFactorizerFactory::create("eigen_decomposition")->run(hamiltonian);
  ASSERT_NE(factorized, nullptr);
  EXPECT_EQ(factorized->get_container_type(), "factorized");
  const auto& container = as_factorized(factorized);

  const Eigen::VectorXd& signs = container.get_signs();
  ASSERT_EQ(static_cast<std::size_t>(signs.size()), container.get_num_ranks());
  EXPECT_EQ(signs.cwiseAbs().maxCoeff(), 1.0);
  EXPECT_TRUE((signs.array() < 0.0).any())
      << "expected at least one negative fragment for an indefinite tensor";

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(two_body, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - two_body).cwiseAbs().maxCoeff();
}

TEST(DoubleFactorizerTest, PreservesOneBodyTermAndCoreEnergy) {
  constexpr std::size_t norb = 4;
  constexpr double core_energy = -3.75;
  auto hamiltonian =
      make_hamiltonian(norb, make_two_body(norb, {1.0, 1.0}, 17), core_energy);

  auto [h_alpha, h_beta] = hamiltonian->get_one_body_integrals();
  auto factorized =
      DoubleFactorizerFactory::create("eigen_decomposition")->run(hamiltonian);

  auto [factorized_h_alpha, factorized_h_beta] =
      factorized->get_one_body_integrals();

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
  EXPECT_TRUE(factorized_h_alpha.isApprox(h_alpha, kReconstructionTolerance));
  EXPECT_TRUE(factorized->is_restricted());
}

TEST(DoubleFactorizerTest, TruncationDiscardsSmallFragments) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, 1e-4, 1e-4}, 23);
  auto hamiltonian = make_hamiltonian(norb, two_body);

  auto exact = DoubleFactorizerFactory::create("eigen_decomposition");
  const auto num_ranks_exact =
      as_factorized(exact->run(hamiltonian)).get_num_ranks();

  auto truncated = DoubleFactorizerFactory::create("eigen_decomposition");
  truncated->settings().set("truncation_threshold", 1e-2);
  auto truncated_hamiltonian = truncated->run(hamiltonian);
  const auto& truncated_container = as_factorized(truncated_hamiltonian);

  EXPECT_LT(truncated_container.get_num_ranks(), num_ranks_exact);
  EXPECT_GT(truncated_container.get_num_ranks(), 0u);

  // Truncation is lossy by construction: the reconstruction must move, but
  // only by the weight that was thrown away.
  auto [g_aaaa, g_aabb, g_bbbb] =
      truncated_hamiltonian->get_two_body_integrals();
  EXPECT_FALSE(g_aaaa.isApprox(two_body, kReconstructionTolerance));
  EXPECT_LT((g_aaaa - two_body).cwiseAbs().maxCoeff(), 1e-2);
}
