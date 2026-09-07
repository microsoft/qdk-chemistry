// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

constexpr double kReconstructionTolerance = 1e-10;

/// Random symmetric factors F_k of the tensor g_pqrs = sum_k s_k F_k(p,q)
/// F_k(r,s). With every s_k positive these are also Cholesky vectors of the
/// two-electron supermatrix, so the same fixture drives both factorizations.
std::vector<Eigen::MatrixXd> make_factors(std::size_t norb, std::size_t count,
                                          unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<Eigen::MatrixXd> factors;
  factors.reserve(count);
  for (std::size_t k = 0; k < count; ++k) {
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
  return factors;
}

/// Build a random two-body tensor with a negative entry in @p signs.
Eigen::VectorXd make_two_body(std::size_t norb,
                              const std::vector<double>& signs, unsigned seed) {
  const std::vector<Eigen::MatrixXd> factors =
      make_factors(norb, signs.size(), seed);

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

/// Pack the same factors as norb^2 x naux Cholesky vectors, so that
/// L L^T is the supermatrix of make_two_body with all-positive signs.
Eigen::MatrixXd make_cholesky_vectors(std::size_t norb, std::size_t naux,
                                      unsigned seed) {
  const std::vector<Eigen::MatrixXd> factors = make_factors(norb, naux, seed);

  Eigen::MatrixXd vectors(static_cast<Eigen::Index>(norb * norb),
                          static_cast<Eigen::Index>(naux));
  for (std::size_t k = 0; k < naux; ++k) {
    for (std::size_t p = 0; p < norb; ++p) {
      for (std::size_t q = 0; q < norb; ++q) {
        vectors(static_cast<Eigen::Index>(p * norb + q),
                static_cast<Eigen::Index>(k)) = factors[k](p, q);
      }
    }
  }
  return vectors;
}

/// Pack `total` Cholesky vectors spanning only `independent` dimensions, by
/// making the surplus columns fixed multiples of the independent ones. Reusing
/// the stored vectors then yields `total` fragments while re-decomposing the
/// dense tensor yields only `independent`, which is what makes the fast path
/// observable rather than merely correct.
Eigen::MatrixXd make_rank_deficient_cholesky_vectors(std::size_t norb,
                                                     std::size_t independent,
                                                     std::size_t total,
                                                     unsigned seed) {
  const Eigen::MatrixXd base = make_cholesky_vectors(norb, independent, seed);

  Eigen::MatrixXd vectors(static_cast<Eigen::Index>(norb * norb),
                          static_cast<Eigen::Index>(total));
  vectors.leftCols(static_cast<Eigen::Index>(independent)) = base;
  for (std::size_t k = independent; k < total; ++k) {
    const double scale = 0.5 + 0.25 * static_cast<double>(k - independent);
    vectors.col(static_cast<Eigen::Index>(k)) =
        scale *
        base.col(static_cast<Eigen::Index>((k - independent) % independent));
  }
  return vectors;
}

/// Rebuild the flattened two-body tensor from its fragments.
Eigen::VectorXd reconstruct_two_body(
    const std::vector<TwoBodyFragment>& fragments, std::size_t norb) {
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
  return reconstructed;
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

std::shared_ptr<Hamiltonian> make_cholesky_hamiltonian(
    std::size_t norb, const Eigen::MatrixXd& vectors,
    double core_energy = -1.25) {
  return std::make_shared<Hamiltonian>(
      std::make_unique<CholeskyHamiltonianContainer>(
          make_one_body(norb, 7), vectors,
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

std::unique_ptr<DoubleFactorizer> make_cholesky_factorizer() {
  auto factorizer = DoubleFactorizerFactory::create("qdk");
  factorizer->settings().set("method", "cholesky");
  return factorizer;
}

}  // namespace

TEST(DoubleFactorizerTest, MetaDataAndFactoryRegistration) {
  auto factorizer = DoubleFactorizerFactory::create("qdk");
  ASSERT_NE(factorizer, nullptr);
  EXPECT_EQ(factorizer->type_name(), "double_factorizer");
  EXPECT_EQ(factorizer->name(), "qdk");
  EXPECT_TRUE(factorizer->settings().has("truncation_threshold"));
  EXPECT_TRUE(factorizer->settings().has("method"));
  EXPECT_EQ(factorizer->settings().get<std::string>("method"),
            "eigen_decomposition");

  const auto available = DoubleFactorizerFactory::available();
  EXPECT_NE(std::find(available.begin(), available.end(), "qdk"),
            available.end());
  EXPECT_THROW(DoubleFactorizerFactory::create("nonexistent_factorizer"),
               std::runtime_error);
}

TEST(DoubleFactorizerTest, RejectsInvalidInput) {
  constexpr std::size_t norb = 4;
  auto hamiltonian = make_hamiltonian(norb, make_two_body(norb, {1.0}, 31));
  auto factorizer = DoubleFactorizerFactory::create("qdk");

  EXPECT_THROW(factorizer->settings().set("truncation_threshold", -1.0),
               std::exception);
  EXPECT_THROW(eigen_decompose_two_body(Eigen::VectorXd::Zero(10), norb),
               std::invalid_argument);

  EXPECT_THROW(eigen_decompose_two_body(Eigen::VectorXd(), 0),
               std::invalid_argument);
  const Eigen::VectorXd tensor = make_two_body(norb, {1.0}, 31);
  EXPECT_THROW(eigen_decompose_two_body(tensor, norb, -1.0),
               std::invalid_argument);
  EXPECT_THROW(eigen_decompose_two_body(
                   tensor, norb, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_FALSE(eigen_decompose_two_body(tensor, norb, 0.0).empty());
  Eigen::VectorXd with_nan = make_two_body(norb, {1.0}, 31);
  with_nan[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(eigen_decompose_two_body(with_nan, norb), std::invalid_argument);
  Eigen::VectorXd with_inf = make_two_body(norb, {1.0}, 31);
  with_inf[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(eigen_decompose_two_body(with_inf, norb), std::invalid_argument);

  EXPECT_THROW(factorizer->run(nullptr), std::invalid_argument);
  EXPECT_THROW(factorizer->run(make_unrestricted_hamiltonian(norb)),
               std::invalid_argument);
  auto truncating = DoubleFactorizerFactory::create("qdk");
  truncating->settings().set("truncation_threshold", 1e6);
  EXPECT_THROW(truncating->run(hamiltonian), std::invalid_argument);
}

TEST(DoubleFactorizerTest, EigenDecomposeFragmentsReconstructTensor) {
  constexpr std::size_t norb = 3;
  const auto two_body = make_two_body(norb, {1.0, -1.0}, 17);
  const auto fragments = eigen_decompose_two_body(two_body, norb);
  ASSERT_FALSE(fragments.empty());

  const Eigen::VectorXd reconstructed = reconstruct_two_body(fragments, norb);
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

  auto factorized = DoubleFactorizerFactory::create("qdk")->run(hamiltonian);
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
  auto factorized = DoubleFactorizerFactory::create("qdk")->run(hamiltonian);

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

  auto exact = DoubleFactorizerFactory::create("qdk");
  const auto num_ranks_exact =
      as_factorized(exact->run(hamiltonian)).get_num_ranks();

  auto truncated = DoubleFactorizerFactory::create("qdk");
  truncated->settings().set("truncation_threshold", 1e-2);
  auto truncated_hamiltonian = truncated->run(hamiltonian);
  const auto& truncated_container = as_factorized(truncated_hamiltonian);

  EXPECT_LT(truncated_container.get_num_ranks(), num_ranks_exact);
  EXPECT_GT(truncated_container.get_num_ranks(), 0u);

  auto [g_aaaa, g_aabb, g_bbbb] =
      truncated_hamiltonian->get_two_body_integrals();
  EXPECT_FALSE(g_aaaa.isApprox(two_body, kReconstructionTolerance));
  EXPECT_LT((g_aaaa - two_body).cwiseAbs().maxCoeff(), 1e-2);
}

TEST(DoubleFactorizerCholeskyTest, MetaDataAndFactoryRegistration) {
  auto factorizer = make_cholesky_factorizer();
  ASSERT_NE(factorizer, nullptr);
  EXPECT_EQ(factorizer->type_name(), "double_factorizer");
  EXPECT_EQ(factorizer->name(), "qdk");
  EXPECT_EQ(factorizer->settings().get<std::string>("method"), "cholesky");

  // The method is a constrained setting rather than a separate registered
  // algorithm, so an unknown value has to be rejected at set() time.
  EXPECT_THROW(factorizer->settings().set("method", "not_a_method"),
               std::exception);
}

TEST(DoubleFactorizerCholeskyTest, FragmentsReconstructPositiveTensor) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, 1.0, 1.0}, 17);
  const auto fragments = cholesky_decompose_two_body(two_body, norb);
  ASSERT_FALSE(fragments.empty());

  const Eigen::VectorXd reconstructed = reconstruct_two_body(fragments, norb);
  for (Eigen::Index i = 0; i < two_body.size(); ++i) {
    EXPECT_NEAR(reconstructed[i], two_body[i], kReconstructionTolerance);
  }

  for (const auto& fragment : fragments) {
    EXPECT_DOUBLE_EQ(fragment.sign, 1.0)
        << "a Cholesky factorization cannot produce a negative fragment";
  }
}

TEST(DoubleFactorizerCholeskyTest, AgreesWithEigenDecomposition) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, 1.0, 1.0}, 29);

  const auto eigen_fragments = eigen_decompose_two_body(two_body, norb);
  const auto cholesky_fragments = cholesky_decompose_two_body(two_body, norb);

  // The threshold selects on ||eps||^2 in both methods, so a positive
  // semi-definite tensor has to yield the same number of fragments.
  EXPECT_EQ(cholesky_fragments.size(), eigen_fragments.size());

  const Eigen::VectorXd from_eigen =
      reconstruct_two_body(eigen_fragments, norb);
  const Eigen::VectorXd from_cholesky =
      reconstruct_two_body(cholesky_fragments, norb);
  EXPECT_LT((from_eigen - from_cholesky).cwiseAbs().maxCoeff(),
            kReconstructionTolerance);
}

TEST(DoubleFactorizerCholeskyTest, SortsFragmentsByDecreasingWeight) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, 1.0, 1.0}, 23);
  const auto fragments = cholesky_decompose_two_body(two_body, norb);
  ASSERT_GE(fragments.size(), 2u);

  for (std::size_t r = 1; r < fragments.size(); ++r) {
    EXPECT_LE(fragments[r].eps.squaredNorm(),
              fragments[r - 1].eps.squaredNorm() + 1e-12);
  }
}

// A fixed machine-epsilon breakdown tolerance passes at norb = 2 but reports a
// positive semi-definite supermatrix as indefinite once the accumulated
// roundoff in the residual diagonal outgrows it, which silently routes every
// realistic input through the O(norb^6) fallback. Cover several orbital counts
// so that regression is caught rather than hidden behind a correct answer.
TEST(DoubleFactorizerCholeskyTest, DoesNotFallBackAsOrbitalCountGrows) {
  const std::vector<std::size_t> orbital_counts = {2, 3, 4, 6, 8};
  for (const std::size_t norb : orbital_counts) {
    const auto two_body = make_two_body(norb, {1.0, 1.0, 1.0}, 41);
    const auto fragments = cholesky_decompose_two_body(two_body, norb);
    ASSERT_FALSE(fragments.empty()) << "norb=" << norb;

    for (const auto& fragment : fragments) {
      ASSERT_DOUBLE_EQ(fragment.sign, 1.0)
          << "norb=" << norb
          << ": fell back to the eigendecomposition for a positive "
             "semi-definite tensor";
    }

    const Eigen::VectorXd reconstructed = reconstruct_two_body(fragments, norb);
    EXPECT_LT((reconstructed - two_body).cwiseAbs().maxCoeff(),
              kReconstructionTolerance)
        << "norb=" << norb;
  }
}

TEST(DoubleFactorizerCholeskyTest, FallsBackForIndefiniteTensor) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, -1.0, 1.0}, 5);
  const auto fragments = cholesky_decompose_two_body(two_body, norb);
  ASSERT_FALSE(fragments.empty());

  // No Cholesky decomposition exists here, so a negative fragment is proof
  // that the eigendecomposition took over.
  const bool has_negative_fragment =
      std::any_of(fragments.begin(), fragments.end(),
                  [](const TwoBodyFragment& f) { return f.sign < 0.0; });
  EXPECT_TRUE(has_negative_fragment)
      << "expected a fallback to the eigendecomposition";

  const Eigen::VectorXd reconstructed = reconstruct_two_body(fragments, norb);
  EXPECT_LT((reconstructed - two_body).cwiseAbs().maxCoeff(),
            kReconstructionTolerance);
}

TEST(DoubleFactorizerCholeskyTest, TruncationDiscardsSmallFragments) {
  constexpr std::size_t norb = 4;
  const auto two_body = make_two_body(norb, {1.0, 1e-4, 1e-4}, 23);

  const auto exact = cholesky_decompose_two_body(two_body, norb);
  const auto truncated = cholesky_decompose_two_body(two_body, norb, 1e-2);

  EXPECT_LT(truncated.size(), exact.size());
  EXPECT_FALSE(truncated.empty());

  // The eigen path thresholds the same quantity, so it has to retain the same
  // number of fragments for the same threshold.
  EXPECT_EQ(truncated.size(),
            eigen_decompose_two_body(two_body, norb, 1e-2).size());
}

TEST(DoubleFactorizerCholeskyTest, RejectsInvalidInput) {
  constexpr std::size_t norb = 4;
  const Eigen::VectorXd tensor = make_two_body(norb, {1.0, 1.0}, 31);

  EXPECT_THROW(cholesky_decompose_two_body(Eigen::VectorXd::Zero(10), norb),
               std::invalid_argument);
  EXPECT_THROW(cholesky_decompose_two_body(Eigen::VectorXd(), 0),
               std::invalid_argument);
  EXPECT_THROW(cholesky_decompose_two_body(tensor, norb, -1.0),
               std::invalid_argument);
  EXPECT_THROW(cholesky_decompose_two_body(
                   tensor, norb, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);

  Eigen::VectorXd with_nan = tensor;
  with_nan[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(cholesky_decompose_two_body(with_nan, norb),
               std::invalid_argument);

  Eigen::VectorXd with_inf = tensor;
  with_inf[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(cholesky_decompose_two_body(with_inf, norb),
               std::invalid_argument);
}

TEST(DoubleFactorizerCholeskyTest, FragmentsFromStoredVectorsReconstruct) {
  constexpr std::size_t norb = 4;
  constexpr std::size_t naux = 3;
  const auto vectors = make_cholesky_vectors(norb, naux, 17);
  const auto two_body = make_two_body(norb, {1.0, 1.0, 1.0}, 17);

  const auto fragments = fragments_from_cholesky_vectors(vectors, norb);
  EXPECT_EQ(fragments.size(), naux);

  // Reusing stored vectors has to give the same tensor as decomposing the
  // dense one, since the vectors already are the first factorization.
  const Eigen::VectorXd reconstructed = reconstruct_two_body(fragments, norb);
  for (Eigen::Index i = 0; i < two_body.size(); ++i) {
    EXPECT_NEAR(reconstructed[i], two_body[i], kReconstructionTolerance);
  }
}

TEST(DoubleFactorizerCholeskyTest, FragmentsFromStoredVectorsRejectInvalid) {
  constexpr std::size_t norb = 4;
  const auto vectors = make_cholesky_vectors(norb, 2, 17);

  EXPECT_THROW(fragments_from_cholesky_vectors(vectors, 0),
               std::invalid_argument);
  EXPECT_THROW(fragments_from_cholesky_vectors(vectors, norb + 1),
               std::invalid_argument);
  EXPECT_THROW(fragments_from_cholesky_vectors(vectors, norb, -1.0),
               std::invalid_argument);

  Eigen::MatrixXd with_nan = vectors;
  with_nan(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(fragments_from_cholesky_vectors(with_nan, norb),
               std::invalid_argument);
}

TEST(DoubleFactorizerCholeskyTest, RunProducesEquivalentFactorizedContainer) {
  constexpr std::size_t norb = 4;
  constexpr double core_energy = -3.75;
  const auto two_body = make_two_body(norb, {1.0, 1.0, 1.0}, 17);
  auto hamiltonian = make_hamiltonian(norb, two_body, core_energy);

  auto factorized = make_cholesky_factorizer()->run(hamiltonian);
  ASSERT_NE(factorized, nullptr);
  EXPECT_EQ(factorized->get_container_type(), "factorized");

  const auto& container = as_factorized(factorized);
  const Eigen::VectorXd& signs = container.get_signs();
  ASSERT_EQ(static_cast<std::size_t>(signs.size()), container.get_num_ranks());
  EXPECT_TRUE((signs.array() > 0.0).all())
      << "a positive semi-definite tensor needs no negative fragment";

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(two_body, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - two_body).cwiseAbs().maxCoeff();

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
  EXPECT_TRUE(factorized->is_restricted());
}

TEST(DoubleFactorizerCholeskyTest, ReusesStoredThreeCenterIntegrals) {
  constexpr std::size_t norb = 3;
  constexpr std::size_t independent = 3;
  constexpr std::size_t stored = 5;
  constexpr double core_energy = 2.25;
  const auto vectors =
      make_rank_deficient_cholesky_vectors(norb, independent, stored, 17);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors, core_energy);

  const Eigen::VectorXd expected =
      std::get<0>(hamiltonian->get_two_body_integrals());

  auto factorized = make_cholesky_factorizer()->run(hamiltonian);
  ASSERT_NE(factorized, nullptr);

  const auto& container = as_factorized(factorized);

  // The stored vectors are rank deficient, so consuming them yields one
  // fragment per stored vector while decomposing the dense tensor would yield
  // only `independent`. That difference is what detects the fast path silently
  // ceasing to be taken: the fallback stays exact, so reconstruction alone
  // could never notice.
  EXPECT_EQ(container.get_num_ranks(), stored)
      << "expected the stored three-center integrals to be reused directly; "
         "getting "
      << independent
      << " fragments means the dense tensor was "
         "re-decomposed instead";

  const Eigen::VectorXd& signs = container.get_signs();
  EXPECT_TRUE((signs.array() > 0.0).all());

  const Eigen::VectorXd& reconstructed =
      std::get<0>(factorized->get_two_body_integrals());
  EXPECT_TRUE(reconstructed.isApprox(expected, kReconstructionTolerance))
      << "max abs deviation: "
      << (reconstructed - expected).cwiseAbs().maxCoeff();

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
}

TEST(DoubleFactorizerCholeskyTest, EigenDecompositionIgnoresStoredVectors) {
  constexpr std::size_t norb = 3;
  constexpr std::size_t independent = 3;
  constexpr std::size_t stored = 5;
  const auto vectors =
      make_rank_deficient_cholesky_vectors(norb, independent, stored, 17);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors);

  const Eigen::VectorXd expected =
      std::get<0>(hamiltonian->get_two_body_integrals());

  // The fast path belongs to the "cholesky" method alone, so the default
  // "eigen_decomposition" has to fall through to the dense tensor and recover
  // the true rank.
  auto factorized = DoubleFactorizerFactory::create("qdk")->run(hamiltonian);
  ASSERT_NE(factorized, nullptr);

  const auto& container = as_factorized(factorized);
  EXPECT_EQ(container.get_num_ranks(), independent);

  const Eigen::VectorXd& reconstructed =
      std::get<0>(factorized->get_two_body_integrals());
  EXPECT_TRUE(reconstructed.isApprox(expected, kReconstructionTolerance));
}
