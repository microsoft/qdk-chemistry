// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/ducc.hpp"
#include "ut_common.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

std::shared_ptr<Wavefunction> amplitudes(std::shared_ptr<Orbitals> orbitals,
                                         const std::string& occupation,
                                         double alpha = 0.0, double beta = 0.0,
                                         double doubles = 0.0) {
  auto determinant =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string(occupation), orbitals));
  const auto occupied = std::count(occupation.begin(), occupation.end(), '2');
  const auto singles = occupied * (occupation.size() - occupied);
  const Eigen::VectorXd ta = Eigen::VectorXd::Constant(singles, alpha);
  const Eigen::VectorXd tb = Eigen::VectorXd::Constant(singles, beta);
  const Eigen::VectorXd tab =
      Eigen::VectorXd::Constant(singles * singles, doubles);
  const Eigen::VectorXd same = Eigen::VectorXd::Zero(singles * singles);
  return std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
      orbitals, determinant, AmplitudeType::CoupledCluster, ta, tb, tab, same,
      same));
}

std::shared_ptr<Hamiltonian> hamiltonian(std::shared_ptr<Orbitals> orbitals,
                                         const Eigen::MatrixXd& one,
                                         const Eigen::VectorXd& opposite,
                                         double core = 0.0) {
  const Eigen::VectorXd same = Eigen::VectorXd::Zero(opposite.size());
  return std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one, one, same, opposite, same, orbitals, core, Eigen::MatrixXd(),
          Eigen::MatrixXd()));
}

void expect_integrals(const Hamiltonian& actual, const Hamiltonian& expected,
                      double tolerance = 1e-12) {
  EXPECT_EQ(actual.is_restricted(), expected.is_restricted());
  EXPECT_NEAR(actual.get_core_energy(), expected.get_core_energy(), tolerance);
  const auto [a, b] = actual.get_one_body_integrals();
  const auto [ea, eb] = expected.get_one_body_integrals();
  ASSERT_EQ(a.rows(), ea.rows());
  ASSERT_EQ(a.cols(), ea.cols());
  ASSERT_EQ(b.rows(), eb.rows());
  ASSERT_EQ(b.cols(), eb.cols());
  EXPECT_LE((a - ea).cwiseAbs().maxCoeff(), tolerance);
  EXPECT_LE((b - eb).cwiseAbs().maxCoeff(), tolerance);
  const auto [aa, ab, bb] = actual.get_two_body_integrals();
  const auto [eaa, eab, ebb] = expected.get_two_body_integrals();
  ASSERT_EQ(aa.size(), eaa.size());
  ASSERT_EQ(ab.size(), eab.size());
  ASSERT_EQ(bb.size(), ebb.size());
  EXPECT_LE((aa - eaa).cwiseAbs().maxCoeff(), tolerance);
  EXPECT_LE((ab - eab).cwiseAbs().maxCoeff(), tolerance);
  EXPECT_LE((bb - ebb).cwiseAbs().maxCoeff(), tolerance);
}

class DuccTest : public ::testing::Test {
 protected:
  std::shared_ptr<Orbitals> orbitals = std::make_shared<ModelOrbitals>(
      testing::unrestricted_index_set(2, {0, 1}, {0, 1}));
  std::shared_ptr<const SymmetryBlockedIndexSet> target =
      testing::unrestricted_index_set(2, {0}, {0});
  Eigen::MatrixXd one =
      (Eigen::MatrixXd(2, 2) << -1.0, 0.03, 0.03, 0.5).finished();
  Eigen::VectorXd opposite = Eigen::VectorXd::Constant(16, 0.07);
  microsoft::DuccConstructor constructor;
};

TEST_F(DuccTest, MetadataFactoryAndConstrainedLevel) {
  EXPECT_EQ(constructor.name(), "qdk_ducc");
  EXPECT_EQ(constructor.type_name(), "effective_hamiltonian_constructor");
  EXPECT_EQ(constructor.settings().get<int64_t>("level"), 2);
  auto factory = EffectiveHamiltonianConstructorFactory::create("qdk_ducc");
  ASSERT_NE(factory, nullptr);
  EXPECT_EQ(factory->name(), constructor.name());
  for (int level = 0; level <= 2; ++level)
    EXPECT_NO_THROW(constructor.settings().set("level", int64_t{level}));
  for (int level : {-1, 3, 4})
    EXPECT_THROW(constructor.settings().set("level", int64_t{level}),
                 std::invalid_argument);
}

TEST_F(DuccTest, RejectsUnsupportedLevels) {
  auto ham = hamiltonian(orbitals, one, opposite);
  auto reference = amplitudes(orbitals, "20");
  for (int level : {-1, 3, 4}) {
    SCOPED_TRACE(level);
    microsoft::DuccConstructor builder;
    EXPECT_THROW(
        {
          builder.settings().set("level", level);
          builder.run(reference, ham, target);
        },
        std::invalid_argument);
  }
}

TEST_F(DuccTest, RejectsMissingAmplitudesAndNonAmplitudeReference) {
  auto ham = hamiltonian(orbitals, one, opposite);
  auto determinant =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string("20"), orbitals));
  EXPECT_THROW(constructor.run(determinant, ham, target),
               std::invalid_argument);
  const std::optional<AmplitudeContainer::VectorVariant> zero =
      Eigen::VectorXd::Zero(1).eval();
  for (int missing : {0, 1, 2}) {
    const auto t1 = missing == 1 ? zero : std::nullopt;
    const auto t2 = missing == 2 ? zero : std::nullopt;
    auto reference =
        std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
            orbitals, determinant, AmplitudeType::CoupledCluster, t1, t1, t2,
            t2, t2));
    EXPECT_THROW(constructor.run(reference, ham, target),
                 std::invalid_argument);
  }
}

TEST_F(DuccTest, FockDressingRetainsReferenceEnergy) {
  one.setZero();
  one(1, 1) = 1.0;
  opposite.setZero();
  auto ham = hamiltonian(orbitals, one, opposite);
  auto reference = amplitudes(orbitals, "20", 0.2);
  for (int level : {1, 2}) {
    SCOPED_TRACE(level);
    microsoft::DuccConstructor builder;
    builder.settings().set("level", int64_t{level});
    auto result = builder.run(reference, ham, target);
    const auto [a, b] = result->get_one_body_integrals();
    const auto [aa, ab, bb] = result->get_two_body_integrals();
    // Include the active determinant; the scalar alone is not its energy.
    const double energy = result->get_core_energy() + a(0, 0) + b(0, 0) + ab(0);
    EXPECT_NEAR(energy, 0.04, 1e-12);
    EXPECT_FALSE(result->is_restricted());
  }
}

TEST_F(DuccTest, LevelZeroPreservesNoncontiguousWindowAndFrozenCore) {
  orbitals = std::make_shared<ModelOrbitals>(
      testing::unrestricted_index_set(5, {1, 3, 4}, {1, 3, 4}),
      testing::unrestricted_index_set(5, {0}, {0}));
  one = Eigen::Vector3d(-2.0, -1.0, 0.5).asDiagonal();
  opposite = Eigen::VectorXd::Zero(81);
  opposite(0) = 0.4;
  opposite(4) = opposite(36) = 0.2;
  opposite(40) = 0.3;
  auto ham = hamiltonian(orbitals, one, opposite, 7.0);
  auto reference = amplitudes(orbitals, "220", 0.1, -0.1, 0.05);
  target = testing::unrestricted_index_set(5, {3, 4}, {3, 4});
  constructor.settings().set("level", 0);
  auto result = constructor.run(reference, ham, target);
  for (const auto& spin : {axes::alpha(), axes::beta()}) {
    EXPECT_EQ(
        spin_channel_indices(result->get_orbitals()->active_indices(), spin),
        (std::vector<std::size_t>{3, 4}));
    EXPECT_EQ(
        spin_channel_indices(result->get_orbitals()->inactive_indices(), spin),
        (std::vector<std::size_t>{0, 1}));
  }
  Eigen::MatrixXd expected_one = Eigen::Vector2d(-0.8, 0.5).asDiagonal();
  Eigen::VectorXd expected_two = Eigen::VectorXd::Zero(16);
  expected_two(0) = 0.3;
  auto expected =
      hamiltonian(result->get_orbitals(), expected_one, expected_two, 3.4);
  expect_integrals(*result, *expected);
  EXPECT_NEAR(result->get_core_energy() +
                  result->get_one_body_element(0, 0, SpinChannel::aa) +
                  result->get_one_body_element(0, 0, SpinChannel::bb) +
                  result->get_two_body_element(0, 0, 0, 0, SpinChannel::aabb),
              2.1, 1e-12);
  expect_integrals(*Hamiltonian::from_json(result->to_json()), *result);
}

TEST_F(DuccTest, EqualZeroUnrestrictedDoublesAreAccepted) {
  auto ham = hamiltonian(orbitals, one, opposite);
  auto reference = amplitudes(orbitals, "20");
  constructor.settings().set("level", 0);
  auto bare = constructor.run(reference, ham, target);
  microsoft::DuccConstructor builder;
  auto result = builder.run(reference, ham, target);
  EXPECT_FALSE(result->is_restricted());
  expect_integrals(*result, *bare);
}

TEST_F(DuccTest, RejectsAlphaAndBetaBasisMismatch) {
  auto basis = testing::create_random_basis_set(2);
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2, 2);
  orbitals = std::make_shared<Orbitals>(identity, identity, std::nullopt,
                                        std::nullopt, std::nullopt, basis);
  auto ham = hamiltonian(orbitals, one, opposite);
  for (int spin : {0, 1}) {
    SCOPED_TRACE(spin);
    Eigen::MatrixXd changed = identity;
    changed.col(0).swap(changed.col(1));
    auto other = std::make_shared<Orbitals>(
        spin == 0 ? changed : identity, spin == 1 ? changed : identity,
        std::nullopt, std::nullopt, std::nullopt, basis);
    auto reference = amplitudes(other, "20");
    EXPECT_THROW(constructor.run(reference, ham, target),
                 std::invalid_argument);
  }
}

TEST_F(DuccTest, RejectsNonFiniteCoreEnergy) {
  auto reference = amplitudes(orbitals, "20");
  for (double core : {std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(core);
    EXPECT_THROW(
        constructor.run(reference, hamiltonian(orbitals, one, opposite, core),
                        target),
        std::invalid_argument);
  }
}

TEST_F(DuccTest, AllActiveLevelsCollapseToInput) {
  const auto g = [](int p, int q, int r, int s) {
    return p == q && r == s ? 0.08 : 0.07;
  };
  Eigen::VectorXd same(16), antisymmetrized(16);
  for (int i = 0; i < 16; ++i) {
    const int p = i / 8, q = i / 4 % 2, r = i / 2 % 2, s = i % 2;
    same(i) = g(p, q, r, s);
    antisymmetrized(i) = 0.5 * (g(p, q, r, s) - g(p, s, r, q));
  }
  const auto build = [&](const Eigen::VectorXd& same_spin) {
    return std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one, one, same_spin, opposite, same_spin, orbitals, 0.3,
            Eigen::MatrixXd(), Eigen::MatrixXd()));
  };
  auto ham = build(same);
  auto expected = build(antisymmetrized);
  auto reference = amplitudes(orbitals, "20", 0.12, -0.08, 0.09);
  for (int level : {0, 1, 2}) {
    SCOPED_TRACE(level);
    microsoft::DuccConstructor builder;
    builder.settings().set("level", int64_t{level});
    auto result = builder.run(reference, ham, orbitals->active_indices());
    expect_integrals(*result, *expected);
  }
}

TEST_F(DuccTest, FrozenCoreMatchesCoreFreeWindow) {
  auto with_core = std::make_shared<ModelOrbitals>(
      testing::unrestricted_index_set(3, {1, 2}, {1, 2}),
      testing::unrestricted_index_set(3, {0}, {0}));
  for (int level : {1, 2}) {
    SCOPED_TRACE(level);
    microsoft::DuccConstructor builder;
    builder.settings().set("level", int64_t{level});
    auto expected =
        builder.run(amplitudes(orbitals, "20", 0.12, -0.08, 0.09),
                    hamiltonian(orbitals, one, opposite, 0.3), target);
    auto result = builder.run(amplitudes(with_core, "20", 0.12, -0.08, 0.09),
                              hamiltonian(with_core, one, opposite, 0.3),
                              testing::unrestricted_index_set(3, {1}, {1}));
    expect_integrals(*result, *expected);
  }
}

#ifdef _OPENMP
TEST_F(DuccTest, LevelTwoIsDeterministicAcrossWorkers) {
  struct RestoreThreads {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    ~RestoreThreads() {
      omp_set_num_threads(threads);
      omp_set_dynamic(dynamic);
    }
  } restore;
  omp_set_dynamic(0);
  auto wide = std::make_shared<ModelOrbitals>(
      testing::unrestricted_index_set(4, {0, 1, 2, 3}, {0, 1, 2, 3}));
  Eigen::MatrixXd wide_one = Eigen::Vector4d(-1.2, -0.5, 0.3, 1.1).asDiagonal();
  wide_one(0, 2) = wide_one(2, 0) = 0.03;
  // Different alpha and beta target orbitals give each term several plans.
  const std::vector<
      std::tuple<std::shared_ptr<Hamiltonian>, std::shared_ptr<Wavefunction>,
                 std::shared_ptr<const SymmetryBlockedIndexSet>>>
      cases{{hamiltonian(orbitals, one, opposite, 0.3),
             amplitudes(orbitals, "20", 0.12, -0.08, 0.09), target},
            {hamiltonian(wide, wide_one, Eigen::VectorXd::Constant(256, 0.07),
                         0.3),
             amplitudes(wide, "2200", 0.12, -0.08, 0.09),
             testing::unrestricted_index_set(4, {1, 2}, {0, 1})}};
  constructor.settings().set("level", int64_t{2});
  for (const auto& [ham, reference, p_indices] : cases) {
    omp_set_num_threads(1);
    auto expected = constructor.run(reference, ham, p_indices);
    for (int workers : {1, 2, 4}) {
      SCOPED_TRACE(workers);
      omp_set_num_threads(workers);
      int observed_workers = 0;
#pragma omp parallel
      {
#pragma omp single
        observed_workers = omp_get_num_threads();
      }
      ASSERT_EQ(observed_workers, workers);
      for (int repeat = 0; repeat < 3; ++repeat) {
        SCOPED_TRACE(repeat);
        auto result = constructor.run(reference, ham, p_indices);
        expect_integrals(*result, *expected, 0.0);
      }
    }
  }
}
#endif

}  // namespace
