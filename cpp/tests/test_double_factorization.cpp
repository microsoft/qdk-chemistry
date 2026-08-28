// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cstddef>
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
  EXPECT_THROW(factorizer->run(nullptr), std::invalid_argument);
  EXPECT_THROW(factorizer->run(make_unrestricted_hamiltonian(norb)),
               std::invalid_argument);
  auto truncating = DoubleFactorizerFactory::create("eigen_decomposition");
  truncating->settings().set("truncation_threshold", 1e6);
  EXPECT_THROW(truncating->run(hamiltonian), std::invalid_argument);
}

TEST(DoubleFactorizerTest, EigenDecomposeFragmentsReconstructTensor) {
  constexpr std::size_t norb = 3;
  const auto two_body = make_two_body(norb, {1.0, -1.0}, 17);
  const auto fragments = eigen_decompose_two_body(two_body, norb);
  ASSERT_FALSE(fragments.empty());

  Eigen::VectorXd reconstructed =
      Eigen::VectorXd::Zero(norb * norb * norb * norb);
  for (const auto& fragment : fragments) {
    const double eps_abs_sum = fragment.eps.array().abs().sum();
    EXPECT_NEAR(fragment.lambda_df, 0.5 * eps_abs_sum * eps_abs_sum,
                kReconstructionTolerance);

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
