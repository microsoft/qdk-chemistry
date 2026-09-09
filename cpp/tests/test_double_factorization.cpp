// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

constexpr double kReconstructionTolerance = 1e-10;

/// Random symmetric matrices: the factors F_k of a two-body tensor, and with
/// @p count 1 the one-body term as well.
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

/// Contract explicit factors into g_pqrs = sum_k signs[k] F_k(p,q) F_k(r,s).
/// All-positive signs make the supermatrix positive semi-definite, so the
/// factors are then also Cholesky vectors of it.
Eigen::VectorXd make_two_body(const std::vector<Eigen::MatrixXd>& factors,
                              const std::vector<double>& signs) {
  const std::size_t norb = static_cast<std::size_t>(factors.front().rows());
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

/// As above, over freshly generated random factors, one per sign.
Eigen::VectorXd make_two_body(std::size_t norb,
                              const std::vector<double>& signs, unsigned seed) {
  return make_two_body(make_factors(norb, signs.size(), seed), signs);
}

/// Pack random factors as norb^2 x naux Cholesky vectors, so that L L^T is the
/// supermatrix of make_two_body with all-positive signs.
///
/// @param independent Number of linearly independent columns, i.e. the true
///        rank of the tensor.
/// @param total Number of stored columns. Anything beyond @p independent is a
///        scaled copy of an earlier column, which leaves the tensor unchanged
///        but makes L rank-deficient, as stored vectors routinely are.
Eigen::MatrixXd make_cholesky_vectors(std::size_t norb, std::size_t independent,
                                      std::size_t total, unsigned seed) {
  const std::vector<Eigen::MatrixXd> factors =
      make_factors(norb, independent, seed);

  Eigen::MatrixXd vectors(static_cast<Eigen::Index>(norb * norb),
                          static_cast<Eigen::Index>(total));
  for (std::size_t k = 0; k < independent; ++k) {
    for (std::size_t p = 0; p < norb; ++p) {
      for (std::size_t q = 0; q < norb; ++q) {
        vectors(static_cast<Eigen::Index>(p * norb + q),
                static_cast<Eigen::Index>(k)) = factors[k](p, q);
      }
    }
  }
  for (std::size_t k = independent; k < total; ++k) {
    const double scale = 0.5 + 0.25 * static_cast<double>(k - independent);
    vectors.col(static_cast<Eigen::Index>(k)) =
        scale *
        vectors.col(static_cast<Eigen::Index>((k - independent) % independent));
  }
  return vectors;
}

std::shared_ptr<Hamiltonian> make_hamiltonian(std::size_t norb,
                                              const Eigen::VectorXd& two_body,
                                              double core_energy = -1.25) {
  return std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          make_factors(norb, 1, 7).front(), two_body,
          testing::create_test_orbitals(static_cast<int>(norb),
                                        static_cast<int>(norb)),
          core_energy, Eigen::MatrixXd::Zero(0, 0)));
}

std::shared_ptr<Hamiltonian> make_cholesky_hamiltonian(
    std::size_t norb, const Eigen::MatrixXd& vectors,
    double core_energy = -1.25) {
  return std::make_shared<Hamiltonian>(
      std::make_unique<CholeskyHamiltonianContainer>(
          make_factors(norb, 1, 7).front(), vectors,
          testing::create_test_orbitals(static_cast<int>(norb),
                                        static_cast<int>(norb)),
          core_energy, Eigen::MatrixXd::Zero(0, 0)));
}

std::shared_ptr<Hamiltonian> make_unrestricted_hamiltonian(std::size_t norb) {
  const auto two_body = make_two_body(norb, {1.0, 1.0}, 3);
  return std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          make_factors(norb, 1, 7).front(), make_factors(norb, 1, 11).front(),
          two_body, two_body, two_body,
          testing::create_test_orbitals(static_cast<int>(norb),
                                        static_cast<int>(norb)),
          0.0, Eigen::MatrixXd::Zero(0, 0), Eigen::MatrixXd::Zero(0, 0)));
}

const FactorizedHamiltonianContainer& as_factorized(
    const std::shared_ptr<Hamiltonian>& hamiltonian) {
  return hamiltonian->get_container<FactorizedHamiltonianContainer>();
}

/// Trivial (no-symmetry) index set over @p num_modes carrying @p indices, for
/// model orbitals that declare an explicit active/inactive space.
std::shared_ptr<const SymmetryBlockedIndexSet> trivial_index_set(
    std::size_t num_modes, const std::vector<std::size_t>& indices) {
  auto symmetries =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  std::unordered_map<SymmetryLabel, std::size_t> extents{
      {SymmetryLabel{}, num_modes}};
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> selected{
      {SymmetryLabel{},
       std::vector<std::uint32_t>(indices.begin(), indices.end())}};
  return std::make_shared<const SymmetryBlockedIndexSet>(symmetries, extents,
                                                         std::move(selected));
}

}  // namespace

TEST(DoubleFactorizerTest, MetaDataAndFactoryRegistration) {
  auto factorizer = DoubleFactorizerFactory::create("qdk");
  ASSERT_NE(factorizer, nullptr);
  EXPECT_EQ(factorizer->type_name(), "double_factorizer");
  EXPECT_EQ(factorizer->name(), "qdk");
  EXPECT_TRUE(factorizer->settings().has("truncation_threshold"));

  const auto available = DoubleFactorizerFactory::available();
  EXPECT_NE(std::find(available.begin(), available.end(), "qdk"),
            available.end());
  EXPECT_THROW(DoubleFactorizerFactory::create("nonexistent_factorizer"),
               std::runtime_error);

  // A Cholesky decomposition exists only for a positive semi-definite
  // supermatrix. Stopping at the breakdown would yield an exact factorization
  // of a *different* tensor and a silently wrong lambda, so an indefinite
  // input is rejected rather than approximated.
  EXPECT_THROW(factorizer->run(
                   make_hamiltonian(4, make_two_body(4, {1.0, -1.0, 1.0}, 5))),
               std::invalid_argument);
}

TEST(DoubleFactorizerTest, RejectsInvalidInput) {
  constexpr std::size_t norb = 4;
  auto hamiltonian = make_hamiltonian(norb, make_two_body(norb, {1.0}, 31));
  auto factorizer = DoubleFactorizerFactory::create("qdk");

  EXPECT_THROW(factorizer->settings().set("truncation_threshold", -1.0),
               std::exception);
  EXPECT_THROW(factorizer->run(nullptr), std::invalid_argument);
  EXPECT_THROW(factorizer->run(make_unrestricted_hamiltonian(norb)),
               std::invalid_argument);

  // The container does not screen the tensor it is handed, so the factorizer
  // has to: a non-finite entry would otherwise reach LAPACK.
  Eigen::VectorXd with_nan = make_two_body(norb, {1.0}, 31);
  with_nan[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(factorizer->run(make_hamiltonian(norb, with_nan)),
               std::invalid_argument);

  Eigen::VectorXd with_inf = make_two_body(norb, {1.0}, 31);
  with_inf[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(factorizer->run(make_hamiltonian(norb, with_inf)),
               std::invalid_argument);

  auto truncating = DoubleFactorizerFactory::create("qdk");
  truncating->settings().set("truncation_threshold", 1e6);
  EXPECT_THROW(truncating->run(hamiltonian), std::invalid_argument);
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

TEST(DoubleFactorizerTest, PreservesInactiveFockAcrossASmallerActiveSpace) {
  constexpr std::size_t nmo = 4;
  constexpr std::size_t nact = 2;
  constexpr double core_energy = 2.5;

  auto orbitals = std::make_shared<ModelOrbitals>(
      trivial_index_set(nmo, {1, 2}), trivial_index_set(nmo, {0}));

  const Eigen::MatrixXd one_body = make_factors(nact, 1, 7).front();
  const Eigen::VectorXd two_body = make_two_body(nact, {1.0, 1.0}, 29);

  // Distinct entries, so a matrix that was dropped, zeroed or transposed on
  // the way through cannot still compare equal.
  Eigen::MatrixXd inactive_fock(nmo, nmo);
  for (std::size_t p = 0; p < nmo; ++p) {
    for (std::size_t q = 0; q < nmo; ++q) {
      inactive_fock(static_cast<Eigen::Index>(p),
                    static_cast<Eigen::Index>(q)) =
          1.0 + static_cast<double>(p * nmo + q);
    }
  }

  auto hamiltonian = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body, two_body, orbitals, core_energy, inactive_fock));
  ASSERT_TRUE(hamiltonian->has_inactive_fock_matrix());

  auto factorized = DoubleFactorizerFactory::create("qdk")->run(hamiltonian);

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
  EXPECT_EQ(as_factorized(factorized).get_num_orbitals(), nact);

  auto [factorized_h_alpha, factorized_h_beta] =
      factorized->get_one_body_integrals();
  EXPECT_TRUE(factorized_h_alpha.isApprox(one_body, kReconstructionTolerance));

  ASSERT_TRUE(factorized->has_inactive_fock_matrix());
  auto [factorized_fock, factorized_fock_beta] =
      factorized->get_inactive_fock_matrix();
  EXPECT_EQ(factorized_fock.rows(), static_cast<Eigen::Index>(nmo));
  EXPECT_EQ(factorized_fock.cols(), static_cast<Eigen::Index>(nmo));
  EXPECT_TRUE(factorized_fock.isApprox(inactive_fock, kReconstructionTolerance))
      << "the inactive Fock matrix did not survive factorization";

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(two_body, kReconstructionTolerance));
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

TEST(DoubleFactorizerTest, RunProducesEquivalentFactorizedContainer) {
  constexpr std::size_t norb = 4;
  constexpr double core_energy = -3.75;
  const auto two_body = make_two_body(norb, {1.0, 1.0, 1.0}, 17);
  auto hamiltonian = make_hamiltonian(norb, two_body, core_energy);

  auto factorized = DoubleFactorizerFactory::create("qdk")->run(hamiltonian);
  ASSERT_NE(factorized, nullptr);
  EXPECT_EQ(factorized->get_container_type(), "factorized");

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(two_body, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - two_body).cwiseAbs().maxCoeff();

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
  EXPECT_TRUE(factorized->is_restricted());
}

// The remaining tests cover the path that consumes a
// CholeskyHamiltonianContainer's stored vectors as the first factorization
// instead of expanding them into a dense norb^4 tensor.
TEST(DoubleFactorizerTest, ReusesStoredThreeCenterIntegrals) {
  constexpr std::size_t norb = 3;
  // Stored as 5 vectors that only span rank 3. Getting 5 fragments back is
  // what proves the stored vectors were consumed as-is: decomposing the dense
  // tensor would have stopped at the numerical rank and returned 3.
  const Eigen::MatrixXd vectors = make_cholesky_vectors(norb, 3, 5, 23);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors);

  auto factorized = DoubleFactorizerFactory::create("qdk")->run(hamiltonian);
  const auto& container = as_factorized(factorized);
  EXPECT_EQ(container.get_num_ranks(), 5u);

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  auto [expected, expected_aabb, expected_bbbb] =
      hamiltonian->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(expected, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - expected).cwiseAbs().maxCoeff();
}

TEST(DoubleFactorizerTest, IgnoresTruncationThresholdForStoredVectors) {
  // truncation_threshold is the pivoted-Cholesky stopping cutoff, and stored
  // vectors skip that step entirely. A threshold large enough to discard every
  // fragment of the equivalent dense tensor therefore has to change nothing.
  constexpr std::size_t norb = 3;
  const Eigen::MatrixXd vectors = make_cholesky_vectors(norb, 3, 5, 23);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors);

  auto factorizer = DoubleFactorizerFactory::create("qdk");
  factorizer->settings().set("truncation_threshold", 1e6);
  auto factorized = factorizer->run(hamiltonian);

  EXPECT_EQ(as_factorized(factorized).get_num_ranks(), 5u);

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  auto [expected, expected_aabb, expected_bbbb] =
      hamiltonian->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(expected, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - expected).cwiseAbs().maxCoeff();
}

TEST(DoubleFactorizerTest, StoredVectorsMatchDenseDecomposition) {
  constexpr std::size_t norb = 3;
  const Eigen::MatrixXd vectors = make_cholesky_vectors(norb, 3, 3, 23);

  // The same tensor reached two ways: from the stored vectors, which skip the
  // first factorization, and from the dense tensor, which runs it. The
  // shortcut is only sound if the two agree, so this is what pins it.
  auto stored = make_cholesky_hamiltonian(norb, vectors);
  auto [dense_tensor, dense_aabb, dense_bbbb] =
      stored->get_two_body_integrals();
  auto dense = make_hamiltonian(norb, dense_tensor);

  auto from_stored = DoubleFactorizerFactory::create("qdk")->run(stored);
  auto from_dense = DoubleFactorizerFactory::create("qdk")->run(dense);

  EXPECT_EQ(as_factorized(from_stored).get_num_ranks(),
            as_factorized(from_dense).get_num_ranks());

  auto [stored_g, stored_aabb, stored_bbbb] =
      from_stored->get_two_body_integrals();
  auto [dense_g, dense_g_aabb, dense_g_bbbb] =
      from_dense->get_two_body_integrals();
  EXPECT_TRUE(stored_g.isApprox(dense_g, kReconstructionTolerance))
      << "max abs deviation: " << (stored_g - dense_g).cwiseAbs().maxCoeff();
}
