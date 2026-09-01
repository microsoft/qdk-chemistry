// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

class DuccTest : public ::testing::Test {
 protected:
  static constexpr std::size_t kNumOrbitals = 3;

  void SetUp() override {
    orbitals = testing::create_test_orbitals(kNumOrbitals, kNumOrbitals, true);
    determinant_reference =
        std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
            Configuration::from_spin_half_string("200"), orbitals));

    Eigen::VectorXd t1(2);
    t1 << 0.08, -0.03;
    Eigen::VectorXd t2(4);
    t2 << 0.04, -0.02, 0.01, 0.03;
    amplitude_reference = make_amplitude_reference(t1, t2);

    one_body.resize(kNumOrbitals, kNumOrbitals);
    one_body << -1.0, 0.08, -0.03, 0.08, -0.4, 0.05, -0.03, 0.05, 0.2;

    two_body.resize(kNumOrbitals * kNumOrbitals * kNumOrbitals * kNumOrbitals);
    for (std::size_t p = 0; p < kNumOrbitals; ++p)
      for (std::size_t q = 0; q < kNumOrbitals; ++q)
        for (std::size_t r = 0; r < kNumOrbitals; ++r)
          for (std::size_t s = 0; s < kNumOrbitals; ++s) {
            const auto index =
                ((p * kNumOrbitals + q) * kNumOrbitals + r) * kNumOrbitals + s;
            two_body[static_cast<Eigen::Index>(index)] =
                0.12 / static_cast<double>(1 + p + q + r + s);
          }

    hamiltonian = std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals, 0.35, Eigen::MatrixXd{}));
  }

  std::shared_ptr<Wavefunction> make_amplitude_reference(
      const Eigen::VectorXd& t1, const Eigen::VectorXd& t2,
      AmplitudeType amplitude_type = AmplitudeType::CoupledCluster,
      std::shared_ptr<Wavefunction> reference = nullptr) const {
    return std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
        orbitals, reference ? std::move(reference) : determinant_reference,
        amplitude_type, t1, t2));
  }

  std::shared_ptr<Hamiltonian> run_ducc(
      std::int64_t level,
      std::shared_ptr<const SymmetryBlockedIndexSet> p_space,
      std::shared_ptr<Wavefunction> reference = nullptr) const {
    auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
    constructor->settings().set("ducc_level", level);
    return constructor->run(
        reference ? std::move(reference) : amplitude_reference, hamiltonian,
        std::move(p_space));
  }

  static void expect_hamiltonians_near(const Hamiltonian& actual,
                                       const Hamiltonian& expected) {
    const auto [actual_one_a, actual_one_b] = actual.get_one_body_integrals();
    const auto [expected_one_a, expected_one_b] =
        expected.get_one_body_integrals();
    EXPECT_TRUE(
        actual_one_a.isApprox(expected_one_a, testing::integral_tolerance));
    EXPECT_TRUE(
        actual_one_b.isApprox(expected_one_b, testing::integral_tolerance));

    const auto [actual_two_aa, actual_two_ab, actual_two_bb] =
        actual.get_two_body_integrals();
    const auto [expected_two_aa, expected_two_ab, expected_two_bb] =
        expected.get_two_body_integrals();
    EXPECT_TRUE(
        actual_two_aa.isApprox(expected_two_aa, testing::integral_tolerance));
    EXPECT_TRUE(
        actual_two_ab.isApprox(expected_two_ab, testing::integral_tolerance));
    EXPECT_TRUE(
        actual_two_bb.isApprox(expected_two_bb, testing::integral_tolerance));
    EXPECT_NEAR(actual.get_core_energy(), expected.get_core_energy(),
                testing::integral_tolerance);
  }

  std::shared_ptr<Orbitals> orbitals;
  std::shared_ptr<Wavefunction> determinant_reference;
  std::shared_ptr<Wavefunction> amplitude_reference;
  std::shared_ptr<Hamiltonian> hamiltonian;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;
};

TEST_F(DuccTest, Settings) {
  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
  EXPECT_EQ(constructor->settings().get<std::int64_t>("ducc_level"), 2);
  EXPECT_NO_THROW(constructor->settings().set("ducc_level", std::int64_t{0}));
  EXPECT_NO_THROW(constructor->settings().set("ducc_level", std::int64_t{2}));
  EXPECT_THROW(constructor->settings().set("ducc_level", std::int64_t{-1}),
               std::invalid_argument);
  EXPECT_THROW(constructor->settings().set("ducc_level", std::int64_t{3}),
               std::invalid_argument);
}

TEST_F(DuccTest, RejectsNullPSpace) {
  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
  EXPECT_THROW(constructor->run(amplitude_reference, hamiltonian, nullptr),
               std::invalid_argument);
}

TEST_F(DuccTest, RejectsEmptyPSpace) {
  const std::array<std::shared_ptr<const SymmetryBlockedIndexSet>, 3>
      empty_p_spaces{
          testing::restricted_index_set(kNumOrbitals, {}),
          testing::unrestricted_index_set(kNumOrbitals, {}, {0}),
          testing::unrestricted_index_set(kNumOrbitals, {0}, {}),
      };
  for (const std::int64_t level : std::array<std::int64_t, 3>{0, 1, 2}) {
    for (const auto& p_space : empty_p_spaces) {
      EXPECT_THROW(run_ducc(level, p_space), std::invalid_argument);
    }
  }
}

TEST_F(DuccTest, RejectsDifferentOrbitals) {
  auto other_orbitals =
      testing::create_test_orbitals(kNumOrbitals, kNumOrbitals, true);
  auto other_determinant =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string("200"), other_orbitals));
  Eigen::VectorXd t1 = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd t2 = Eigen::VectorXd::Zero(4);
  auto other_reference =
      std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
          other_orbitals, other_determinant, AmplitudeType::CoupledCluster, t1,
          t2));

  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
  EXPECT_THROW(
      constructor->run(other_reference, hamiltonian,
                       testing::restricted_index_set(kNumOrbitals, {0, 1})),
      std::invalid_argument);
}

TEST_F(DuccTest, AcceptsEquivalentOrbitalObjects) {
  auto equivalent_orbitals = std::make_shared<Orbitals>(*orbitals);
  auto equivalent_determinant =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string("200"), equivalent_orbitals));
  Eigen::VectorXd t1(2);
  t1 << 0.08, -0.03;
  Eigen::VectorXd t2(4);
  t2 << 0.04, -0.02, 0.01, 0.03;
  auto equivalent_reference =
      std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
          equivalent_orbitals, equivalent_determinant,
          AmplitudeType::CoupledCluster, t1, t2));

  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
  EXPECT_NO_THROW(
      constructor->run(equivalent_reference, hamiltonian,
                       testing::restricted_index_set(kNumOrbitals, {0, 1})));
}

TEST_F(DuccTest, RejectsHamiltonianWithInactiveFockMatrix) {
  auto active_hamiltonian = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body, two_body, orbitals, 0.35,
          Eigen::MatrixXd::Identity(kNumOrbitals, kNumOrbitals)));
  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");

  EXPECT_THROW(
      constructor->run(amplitude_reference, active_hamiltonian,
                       testing::restricted_index_set(kNumOrbitals, {0, 1})),
      std::runtime_error);
}

TEST_F(DuccTest, RejectsComplexAmplitudes) {
  Eigen::VectorXcd t1 = Eigen::VectorXcd::Zero(2);
  Eigen::VectorXcd t2 = Eigen::VectorXcd::Zero(4);
  t1[0] = std::complex<double>{0.08, 0.01};
  auto complex_reference =
      std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
          orbitals, determinant_reference, AmplitudeType::CoupledCluster,
          std::optional<AmplitudeContainer::VectorVariant>{t1},
          std::optional<AmplitudeContainer::VectorVariant>{t2}));
  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");

  EXPECT_THROW(
      constructor->run(complex_reference, hamiltonian,
                       testing::restricted_index_set(kNumOrbitals, {0, 1})),
      std::runtime_error);
}

TEST_F(DuccTest, RejectsNonCoupledClusterAmplitudes) {
  Eigen::VectorXd t1 = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd t2 = Eigen::VectorXd::Zero(4);

  for (const auto amplitude_type :
       {AmplitudeType::MollerPlesset, AmplitudeType::Unspecified}) {
    const auto reference = make_amplitude_reference(t1, t2, amplitude_type);
    EXPECT_THROW(
        run_ducc(1, testing::restricted_index_set(kNumOrbitals, {0, 1}),
                 reference),
        std::invalid_argument);
  }
}

TEST_F(DuccTest, RejectsMultiDeterminantReference) {
  Eigen::VectorXd coefficients(2);
  coefficients << 1.0, 0.0;
  auto multi_determinant =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          coefficients,
          std::vector<Configuration>{
              Configuration::from_spin_half_string("200"),
              Configuration::from_spin_half_string("020")},
          orbitals));
  Eigen::VectorXd t1 = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd t2 = Eigen::VectorXd::Zero(4);
  const auto reference = make_amplitude_reference(
      t1, t2, AmplitudeType::CoupledCluster, multi_determinant);

  EXPECT_THROW(run_ducc(1, testing::restricted_index_set(kNumOrbitals, {0, 1}),
                        reference),
               std::invalid_argument);
}

TEST_F(DuccTest, RejectsNonPrefixOccupiedOrbitals) {
  auto non_prefix =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string("020"), orbitals));
  Eigen::VectorXd t1 = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd t2 = Eigen::VectorXd::Zero(4);
  const auto reference = make_amplitude_reference(
      t1, t2, AmplitudeType::CoupledCluster, non_prefix);

  EXPECT_THROW(run_ducc(1, testing::restricted_index_set(kNumOrbitals, {0, 1}),
                        reference),
               std::invalid_argument);
}

TEST_F(DuccTest, RejectsNonHermitianHamiltonian) {
  Eigen::MatrixXd non_hermitian_one_body = one_body;
  non_hermitian_one_body(0, 1) += 1.0;
  auto non_hermitian = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          non_hermitian_one_body, two_body, orbitals, 0.35, Eigen::MatrixXd{},
          HamiltonianType::NonHermitian));
  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");

  EXPECT_THROW(
      constructor->run(amplitude_reference, non_hermitian,
                       testing::restricted_index_set(kNumOrbitals, {0, 1})),
      std::invalid_argument);
}

TEST_F(DuccTest, AcceptsCholeskyHamiltonian) {
  const Eigen::MatrixXd three_center =
      Eigen::MatrixXd::Zero(kNumOrbitals * kNumOrbitals, 1);
  auto cholesky = std::make_shared<Hamiltonian>(
      std::make_unique<CholeskyHamiltonianContainer>(
          one_body, three_center, orbitals, 0.35, Eigen::MatrixXd{}));
  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
  constructor->settings().set("ducc_level", std::int64_t{0});

  const auto output =
      constructor->run(amplitude_reference, cholesky,
                       testing::restricted_index_set(kNumOrbitals, {0, 1}));
  const auto [one_a, one_b] = output->get_one_body_integrals();
  EXPECT_TRUE(one_a.isApprox(one_body.topLeftCorner(2, 2),
                             testing::integral_tolerance));
  EXPECT_TRUE(one_b.isApprox(one_body.topLeftCorner(2, 2),
                             testing::integral_tolerance));
  const auto [two_aa, two_ab, two_bb] = output->get_two_body_integrals();
  EXPECT_TRUE(two_aa.isZero(testing::integral_tolerance));
  EXPECT_TRUE(two_ab.isZero(testing::integral_tolerance));
  EXPECT_TRUE(two_bb.isZero(testing::integral_tolerance));
}

TEST_F(DuccTest, RejectsUnequalSpinPSpaceSizes) {
  auto unrestricted_orbitals =
      testing::create_test_orbitals(kNumOrbitals, kNumOrbitals, false);
  auto unrestricted_determinant =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string("200"), unrestricted_orbitals));
  Eigen::VectorXd t1(2);
  t1 << 0.08, -0.03;
  Eigen::VectorXd t2(4);
  t2 << 0.04, -0.02, 0.01, 0.03;
  auto unrestricted_reference =
      std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
          unrestricted_orbitals, unrestricted_determinant,
          AmplitudeType::CoupledCluster,
          std::optional<AmplitudeContainer::VectorVariant>{t1},
          std::optional<AmplitudeContainer::VectorVariant>{t1},
          std::optional<AmplitudeContainer::VectorVariant>{t2},
          std::optional<AmplitudeContainer::VectorVariant>{t2},
          std::optional<AmplitudeContainer::VectorVariant>{t2}));
  auto unrestricted_hamiltonian = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body, one_body, two_body, two_body, two_body,
          unrestricted_orbitals, 0.35, Eigen::MatrixXd{}, Eigen::MatrixXd{}));
  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
  EXPECT_THROW(constructor->run(
                   unrestricted_reference, unrestricted_hamiltonian,
                   testing::unrestricted_index_set(kNumOrbitals, {0, 1}, {0})),
               std::runtime_error);
}

TEST_F(DuccTest, LevelZeroProjectsBareHamiltonianToPSpace) {
  const std::array<std::size_t, 2> active{0, 2};
  const auto output = run_ducc(
      0, testing::restricted_index_set(kNumOrbitals, {active[0], active[1]}));
  const auto [one_a, one_b] = output->get_one_body_integrals();

  Eigen::MatrixXd expected_one(2, 2);
  for (std::size_t p = 0; p < active.size(); ++p)
    for (std::size_t q = 0; q < active.size(); ++q)
      expected_one(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)) =
          one_body(static_cast<Eigen::Index>(active[p]),
                   static_cast<Eigen::Index>(active[q]));

  EXPECT_TRUE(one_a.isApprox(expected_one, testing::integral_tolerance));
  EXPECT_TRUE(one_b.isApprox(expected_one, testing::integral_tolerance));
  EXPECT_FALSE(output->is_restricted());

  const auto output_orbitals = output->get_orbitals();
  EXPECT_EQ(
      spin_channel_indices(output_orbitals->active_indices(), axes::alpha()),
      (std::vector<std::size_t>{0, 2}));
  EXPECT_EQ(
      spin_channel_indices(output_orbitals->inactive_indices(), axes::alpha()),
      (std::vector<std::size_t>{}));
  const auto [virtual_a, virtual_b] =
      output_orbitals->get_virtual_space_indices();
  EXPECT_EQ(virtual_a, (std::vector<std::size_t>{1}));
  EXPECT_EQ(virtual_b, (std::vector<std::size_t>{1}));

  const auto [two_aa, two_ab, two_bb] = output->get_two_body_integrals();
  ASSERT_EQ(two_aa.size(), 16);
  ASSERT_EQ(two_ab.size(), 16);
  ASSERT_EQ(two_bb.size(), 16);
  for (std::size_t p = 0; p < active.size(); ++p)
    for (std::size_t q = 0; q < active.size(); ++q)
      for (std::size_t r = 0; r < active.size(); ++r)
        for (std::size_t s = 0; s < active.size(); ++s) {
          const auto output_index =
              ((p * active.size() + q) * active.size() + r) * active.size() + s;
          const auto input_index =
              ((active[p] * kNumOrbitals + active[q]) * kNumOrbitals +
               active[r]) *
                  kNumOrbitals +
              active[s];
          EXPECT_NEAR(two_ab[static_cast<Eigen::Index>(output_index)],
                      two_body[static_cast<Eigen::Index>(input_index)],
                      testing::integral_tolerance);
          EXPECT_NEAR(two_aa[static_cast<Eigen::Index>(output_index)], 0.0,
                      testing::integral_tolerance);
          EXPECT_NEAR(two_bb[static_cast<Eigen::Index>(output_index)], 0.0,
                      testing::integral_tolerance);
        }
  EXPECT_NEAR(output->get_core_energy(), 0.35, testing::integral_tolerance);
}

TEST_F(DuccTest, OutputOrbitalsClassifyOccupiedQAsInactive) {
  const auto output =
      run_ducc(0, testing::restricted_index_set(kNumOrbitals, {1, 2}));
  const auto output_orbitals = output->get_orbitals();

  EXPECT_EQ(
      spin_channel_indices(output_orbitals->active_indices(), axes::alpha()),
      (std::vector<std::size_t>{1, 2}));
  EXPECT_EQ(
      spin_channel_indices(output_orbitals->inactive_indices(), axes::alpha()),
      (std::vector<std::size_t>{0}));
  const auto [virtual_a, virtual_b] =
      output_orbitals->get_virtual_space_indices();
  EXPECT_TRUE(virtual_a.empty());
  EXPECT_TRUE(virtual_b.empty());
}

TEST_F(DuccTest, TreatsPSpaceAsAbsoluteOrbitalIndices) {
  auto base_orbitals = testing::create_test_orbitals(4, 4, true);
  auto window_orbitals =
      testing::with_active_space(base_orbitals, {0, 2, 3}, {});
  auto window_determinant =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string("200"), window_orbitals));
  Eigen::VectorXd t1(2);
  t1 << 0.08, -0.03;
  Eigen::VectorXd t2(4);
  t2 << 0.04, -0.02, 0.01, 0.03;
  auto window_reference =
      std::make_shared<Wavefunction>(std::make_unique<AmplitudeContainer>(
          window_orbitals, window_determinant, AmplitudeType::CoupledCluster,
          t1, t2));
  auto window_hamiltonian = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body, two_body, window_orbitals, 0.35, Eigen::MatrixXd{}));

  auto constructor = EffectiveHamiltonianConstructorFactory::create("ducc");
  constructor->settings().set("ducc_level", std::int64_t{0});
  const auto output =
      constructor->run(window_reference, window_hamiltonian,
                       testing::restricted_index_set(4, {0, 3}));
  const auto [one_a, one_b] = output->get_one_body_integrals();
  const std::array<std::size_t, 2> local_positions{0, 2};
  Eigen::MatrixXd expected_one(2, 2);
  for (std::size_t p = 0; p < local_positions.size(); ++p)
    for (std::size_t q = 0; q < local_positions.size(); ++q)
      expected_one(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)) =
          one_body(static_cast<Eigen::Index>(local_positions[p]),
                   static_cast<Eigen::Index>(local_positions[q]));

  EXPECT_TRUE(one_a.isApprox(expected_one, testing::integral_tolerance));
  EXPECT_TRUE(one_b.isApprox(expected_one, testing::integral_tolerance));
  EXPECT_EQ(spin_channel_indices(output->get_orbitals()->active_indices(),
                                 axes::alpha()),
            (std::vector<std::size_t>{0, 3}));
}

TEST_F(DuccTest, AllActiveSpaceReducesToLevelZero) {
  const auto p_space = testing::restricted_index_set(kNumOrbitals, {0, 1, 2});
  const auto level_zero = run_ducc(0, p_space);

  for (const std::int64_t level : std::array<std::int64_t, 2>{1, 2}) {
    const auto dressed = run_ducc(level, p_space);
    expect_hamiltonians_near(*dressed, *level_zero);
  }
}

TEST_F(DuccTest, ExternalAmplitudesDressPSpaceHamiltonian) {
  const auto p_space = testing::restricted_index_set(kNumOrbitals, {0, 1});
  const auto level_zero = run_ducc(0, p_space);
  const auto level_one = run_ducc(1, p_space);
  const auto level_two = run_ducc(2, p_space);

  const auto [zero_one_a, zero_one_b] = level_zero->get_one_body_integrals();
  const auto [one_one_a, one_one_b] = level_one->get_one_body_integrals();
  const auto [two_one_a, two_one_b] = level_two->get_one_body_integrals();
  const double level_one_difference =
      (one_one_a - zero_one_a).norm() + (one_one_b - zero_one_b).norm() +
      std::abs(level_one->get_core_energy() - level_zero->get_core_energy());
  const double level_two_difference =
      (two_one_a - zero_one_a).norm() + (two_one_b - zero_one_b).norm() +
      std::abs(level_two->get_core_energy() - level_zero->get_core_energy());

  EXPECT_GT(level_one_difference, testing::integral_tolerance);
  EXPECT_GT(level_two_difference, testing::integral_tolerance);
}

}  // namespace
