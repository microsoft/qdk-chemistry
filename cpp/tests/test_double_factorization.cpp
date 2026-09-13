// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <random>
#include <stdexcept>
#include <string>
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

/// Contract supplied pair-layout factors into g_pqrs = sum_Q L_(pq),Q L_(rs),Q.
Eigen::VectorXd make_two_body(const Eigen::MatrixXd& vectors,
                              std::size_t norb) {
  Eigen::VectorXd two_body = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  for (std::size_t p = 0; p < norb; ++p) {
    for (std::size_t q = 0; q < norb; ++q) {
      for (std::size_t r = 0; r < norb; ++r) {
        for (std::size_t s = 0; s < norb; ++s) {
          double value = 0.0;
          for (Eigen::Index k = 0; k < vectors.cols(); ++k) {
            value += vectors(p * norb + q, k) * vectors(r * norb + s, k);
          }
          two_body[((p * norb + q) * norb + r) * norb + s] = value;
        }
      }
    }
  }
  return two_body;
}

/// Pack random symmetric factors as norb^2 x naux Cholesky vectors.
///
/// @param independent Number of linearly independent columns, i.e. the true
///        rank of the tensor.
/// @param total Number of stored columns. Anything beyond @p independent is a
///        scaled copy of an earlier column, making L rank-deficient.
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

const DFTHCHamiltonianContainer& as_factorized(
    const std::shared_ptr<Hamiltonian>& hamiltonian) {
  return hamiltonian->get_container<DFTHCHamiltonianContainer>();
}

}  // namespace

TEST(DoubleFactorizationTest, MetadataAndFactoryRegistration) {
  auto factorizer =
      HamiltonianFactorizationFactory::create("double_factorization");
  ASSERT_NE(factorizer, nullptr);
  EXPECT_EQ(factorizer->type_name(), "hamiltonian_factorization");
  EXPECT_EQ(factorizer->name(), "double_factorization");
  EXPECT_FALSE(factorizer->settings().has("truncation_threshold"));
  EXPECT_THROW(factorizer->settings().set("truncation_threshold", 1e-8),
               SettingNotFound);

  const auto available = HamiltonianFactorizationFactory::available();
  EXPECT_NE(
      std::find(available.begin(), available.end(), "double_factorization"),
      available.end());
}

TEST(DoubleFactorizationTest, RejectsInvalidInput) {
  constexpr std::size_t norb = 4;
  auto factorizer =
      HamiltonianFactorizationFactory::create("double_factorization");

  EXPECT_THROW(factorizer->run(nullptr), std::invalid_argument);

  const Eigen::MatrixXd vectors = make_cholesky_vectors(norb, 2, 2, 31);
  auto canonical = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          make_factors(norb, 1, 7).front(), make_two_body(vectors, norb),
          testing::create_test_orbitals(norb, norb), 0.0,
          Eigen::MatrixXd::Zero(0, 0)));
  ASSERT_TRUE(canonical->is_restricted());
  EXPECT_THROW(factorizer->run(canonical), std::invalid_argument);

  auto unrestricted = std::make_shared<Hamiltonian>(
      std::make_unique<CholeskyHamiltonianContainer>(
          make_factors(norb, 1, 7).front(), make_factors(norb, 1, 11).front(),
          vectors, 2.0 * vectors,
          testing::create_test_orbitals(norb, norb, false), 0.0,
          Eigen::MatrixXd::Zero(0, 0), Eigen::MatrixXd::Zero(0, 0)));
  ASSERT_FALSE(unrestricted->is_restricted());
  EXPECT_THROW(factorizer->run(unrestricted), std::invalid_argument);

  for (double non_finite : {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()}) {
    Eigen::MatrixXd invalid_vectors = vectors;
    invalid_vectors(0, 0) = non_finite;
    auto invalid = make_cholesky_hamiltonian(norb, invalid_vectors);
    EXPECT_THROW(factorizer->run(invalid), std::invalid_argument);
  }

  Eigen::MatrixXd asymmetric_vectors = vectors;
  asymmetric_vectors(1, 1) += 0.1;
  auto asymmetric = make_cholesky_hamiltonian(norb, asymmetric_vectors);
  EXPECT_THROW(factorizer->run(asymmetric), std::invalid_argument);
}

TEST(DoubleFactorizationTest, RejectsIncompatibleThreeCenterLayout) {
  constexpr std::size_t norb = 2;
  auto original =
      make_cholesky_hamiltonian(norb, make_cholesky_vectors(norb, 2, 2, 31));
  const auto& container =
      original->get_container<CholeskyHamiltonianContainer>();
  const auto& three_center = container.three_center();

  // Rank-3 tensors validate total size, not the producer's row/column split.
  SymmetryBlockedTensor<3>::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha(), SymmetryLabel{}}] =
      std::make_shared<const Eigen::MatrixXd>(
          container.get_three_center_integrals().first.transpose());
  SymmetryBlockedTensor<3> incompatible(
      three_center.symmetries(), three_center.extents(), std::move(blocks));
  auto hamiltonian = std::make_shared<Hamiltonian>(
      std::make_unique<CholeskyHamiltonianContainer>(
          container.one_body_integrals(), std::move(incompatible),
          original->get_orbitals(), original->get_core_energy(), nullptr));
  ASSERT_TRUE(hamiltonian->is_restricted());

  auto factorizer =
      HamiltonianFactorizationFactory::create("double_factorization");
  try {
    factorizer->run(hamiltonian);
    FAIL() << "incompatible three-center layout was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("expected norb^2 = 4 rows"),
              std::string::npos);
  }
}

TEST(DoubleFactorizationTest, PreservesOneBodyTermAndCoreEnergy) {
  constexpr std::size_t norb = 4;
  constexpr double core_energy = -3.75;
  auto hamiltonian = make_cholesky_hamiltonian(
      norb, make_cholesky_vectors(norb, 2, 2, 17), core_energy);

  auto [h_alpha, h_beta] = hamiltonian->get_one_body_integrals();
  auto factorized =
      HamiltonianFactorizationFactory::create("double_factorization")
          ->run(hamiltonian);

  auto [factorized_h_alpha, factorized_h_beta] =
      factorized->get_one_body_integrals();

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
  EXPECT_TRUE(factorized_h_alpha.isApprox(h_alpha, kReconstructionTolerance));
  EXPECT_TRUE(factorized_h_beta.isApprox(h_beta, kReconstructionTolerance));
  EXPECT_EQ(factorized->get_orbitals(), hamiltonian->get_orbitals());
  EXPECT_EQ(factorized->get_type(), hamiltonian->get_type());
  EXPECT_FALSE(factorized->has_inactive_fock_matrix());
  EXPECT_TRUE(factorized->is_restricted());
}

TEST(DoubleFactorizationTest, PreservesInactiveFockAcrossASmallerActiveSpace) {
  constexpr std::size_t nmo = 4;
  constexpr std::size_t nact = 2;
  constexpr double core_energy = 2.5;

  auto orbitals = std::make_shared<ModelOrbitals>(
      testing::restricted_index_set(nmo, {1, 2}),
      testing::restricted_index_set(nmo, {0}));

  const Eigen::MatrixXd one_body = make_factors(nact, 1, 7).front();
  const Eigen::MatrixXd vectors = make_cholesky_vectors(nact, 2, 2, 29);
  const Eigen::VectorXd two_body = make_two_body(vectors, nact);

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
      std::make_unique<CholeskyHamiltonianContainer>(
          one_body, vectors, orbitals, core_energy, inactive_fock, std::nullopt,
          HamiltonianType::NonHermitian));
  ASSERT_TRUE(hamiltonian->has_inactive_fock_matrix());

  auto factorized =
      HamiltonianFactorizationFactory::create("double_factorization")
          ->run(hamiltonian);

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
  EXPECT_EQ(as_factorized(factorized).get_num_orbitals(), nact);
  EXPECT_EQ(factorized->get_orbitals(), orbitals);
  EXPECT_EQ(factorized->get_type(), HamiltonianType::NonHermitian);

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

TEST(DoubleFactorizationTest, PreservesSmallStoredFragments) {
  constexpr std::size_t norb = 4;
  Eigen::MatrixXd vectors = make_cholesky_vectors(norb, 3, 3, 23);
  vectors.rightCols(2) *= 1e-2;
  const Eigen::VectorXd two_body = make_two_body(vectors, norb);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors);

  auto factorizer =
      HamiltonianFactorizationFactory::create("double_factorization");
  auto factorized = factorizer->run(hamiltonian);
  EXPECT_EQ(as_factorized(factorized).get_num_ranks(), 3u);

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(two_body, kReconstructionTolerance));
}

TEST(DoubleFactorizationTest, RunProducesEquivalentFactorizedContainer) {
  constexpr std::size_t norb = 4;
  constexpr double core_energy = -3.75;
  const Eigen::MatrixXd vectors = make_cholesky_vectors(norb, 3, 3, 17);
  const Eigen::VectorXd two_body = make_two_body(vectors, norb);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors, core_energy);

  auto factorized =
      HamiltonianFactorizationFactory::create("double_factorization")
          ->run(hamiltonian);
  ASSERT_NE(factorized, nullptr);
  EXPECT_EQ(factorized->get_container_type(), "factorized");

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  EXPECT_TRUE(g_aaaa.isApprox(two_body, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - two_body).cwiseAbs().maxCoeff();

  EXPECT_DOUBLE_EQ(factorized->get_core_energy(), core_energy);
  EXPECT_TRUE(factorized->is_restricted());
}

TEST(DoubleFactorizationTest, ReusesStoredThreeCenterIntegrals) {
  constexpr std::size_t norb = 3;
  // Stored as 5 vectors that only span rank 3. Getting 5 fragments back is
  // what proves the stored vectors were consumed as-is: decomposing the dense
  // tensor would have stopped at the numerical rank and returned 3.
  const Eigen::MatrixXd vectors = make_cholesky_vectors(norb, 3, 5, 23);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors);

  auto factorized =
      HamiltonianFactorizationFactory::create("double_factorization")
          ->run(hamiltonian);
  const auto& container = as_factorized(factorized);
  EXPECT_EQ(container.get_num_ranks(), 5u);

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  const Eigen::VectorXd expected = make_two_body(vectors, norb);
  EXPECT_TRUE(g_aaaa.isApprox(expected, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - expected).cwiseAbs().maxCoeff();
}

TEST(DoubleFactorizationTest, PreservesRedundantCustomFactorization) {
  constexpr std::size_t norb = 2;
  Eigen::MatrixXd vectors(norb * norb, 2);
  vectors.col(0) << 0.7, 0.0, 0.0, 0.2;
  vectors.col(1) = 2.0 * vectors.col(0);
  auto hamiltonian = make_cholesky_hamiltonian(norb, vectors);
  const auto input_hash = hamiltonian->content_hash();

  auto factorizer =
      HamiltonianFactorizationFactory::create("double_factorization");
  auto factorized = factorizer->run(hamiltonian);
  const auto& container = as_factorized(factorized);

  ASSERT_EQ(container.get_num_ranks(), 2u);
  EXPECT_EQ(container.get_num_bases(), norb);
  EXPECT_EQ(container.get_num_copies(), 1u);

  auto [g_aaaa, g_aabb, g_bbbb] = factorized->get_two_body_integrals();
  const Eigen::VectorXd expected = make_two_body(vectors, norb);
  EXPECT_TRUE(g_aaaa.isApprox(expected, kReconstructionTolerance))
      << "max abs deviation: " << (g_aaaa - expected).cwiseAbs().maxCoeff();
  EXPECT_EQ(hamiltonian->content_hash(), input_hash);
  EXPECT_TRUE(hamiltonian->get_container<CholeskyHamiltonianContainer>()
                  .get_three_center_integrals()
                  .first == vectors);
}
