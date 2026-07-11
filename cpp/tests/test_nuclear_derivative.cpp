// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../src/qdk/chemistry/algorithms/nuclear_derivative_detail.hpp"
#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace qdk::chemistry::tests::test_support {

std::vector<std::vector<size_t>> recorded_active_spaces;
std::vector<std::vector<size_t>> recorded_inactive_spaces;
std::vector<std::vector<size_t>> recorded_localized_alpha_spaces;
std::vector<std::vector<size_t>> recorded_localized_beta_spaces;
constexpr double dft_analytic_numeric_gradient_tolerance = 5.0e-5;

class RecordingMultiConfigurationCalculator
    : public MultiConfigurationCalculator {
 public:
  std::string name() const override {
    return "_test_nuclear_derivative_recording_mc";
  }

 protected:
  std::pair<double, std::shared_ptr<Wavefunction>> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian, unsigned int n_alpha,
      unsigned int n_beta) const override {
    auto orbitals = hamiltonian->get_orbitals();
    recorded_active_spaces.push_back(
        orbitals->get_active_space_indices().first);
    recorded_inactive_spaces.push_back(
        orbitals->get_inactive_space_indices().first);

    auto determinant = Configuration::canonical_hf_configuration(
        n_alpha, n_beta, orbitals->get_active_space_indices().first.size());
    auto container =
        std::make_unique<StateVectorContainer>(determinant, orbitals);
    return {0.0, std::make_shared<Wavefunction>(std::move(container))};
  }
};

class RecordingLocalizer : public Localizer {
 public:
  std::string name() const override {
    return "_test_nuclear_derivative_recording_localizer";
  }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override {
    recorded_localized_alpha_spaces.push_back(loc_indices_a);
    recorded_localized_beta_spaces.push_back(loc_indices_b);
    return wavefunction;
  }
};

template <typename Factory>
class ScopedFactoryRegistration {
 public:
  explicit ScopedFactoryRegistration(std::string key) : key_(std::move(key)) {
    Factory::unregister_instance(key_);
  }

  ~ScopedFactoryRegistration() { Factory::unregister_instance(key_); }

  ScopedFactoryRegistration(const ScopedFactoryRegistration&) = delete;
  ScopedFactoryRegistration& operator=(const ScopedFactoryRegistration&) =
      delete;

 private:
  std::string key_;
};

void expect_same_structure(const std::shared_ptr<Structure>& left,
                           const std::shared_ptr<Structure>& right) {
  ASSERT_NE(left, nullptr);
  ASSERT_NE(right, nullptr);
  ASSERT_EQ(left->get_num_atoms(), right->get_num_atoms());
  EXPECT_TRUE(
      left->get_coordinates().isApprox(right->get_coordinates(), 1.0e-12));
  EXPECT_EQ(left->get_elements(), right->get_elements());
}

NuclearDerivativeResult run_qdk_derivative_for_functional(
    const std::string& calculator_name, const std::string& functional,
    const std::shared_ptr<Structure>& structure =
        testing::create_lih_structure(),
    int charge = 0, int spin_multiplicity = 1) {
  auto calculator = NuclearDerivativeCalculatorFactory::create(calculator_name);
  AlgorithmRef scf_ref("scf_solver", "qdk");
  scf_ref.get_settings()->set("method", functional);
  calculator->settings().set("energy_calculator", scf_ref);
  if (calculator->settings().has("finite_difference_step")) {
    calculator->settings().set("finite_difference_step", 1.0e-2);
  }

  return calculator->run(structure, charge, spin_multiplicity,
                         std::string("sto-3g"), 0);
}

void expect_qdk_analytic_gradient_matches_numeric(
    const std::string& functional,
    double gradient_tolerance = dft_analytic_numeric_gradient_tolerance,
    const std::shared_ptr<Structure>& structure =
        testing::create_lih_structure(),
    int charge = 0, int spin_multiplicity = 1) {
  auto [numeric_energy, numeric_gradients, numeric_hessian,
        numeric_wavefunction] =
      run_qdk_derivative_for_functional("qdk_finite_difference", functional,
                                        structure, charge, spin_multiplicity);
  auto [analytic_energy, analytic_gradients, analytic_hessian,
        analytic_wavefunction] =
      run_qdk_derivative_for_functional("qdk", functional, structure, charge,
                                        spin_multiplicity);

  EXPECT_NEAR(analytic_energy, numeric_energy, 1.0e-8);
  ASSERT_NE(numeric_gradients, nullptr);
  ASSERT_NE(analytic_gradients, nullptr);
  const auto num_atoms = structure->get_num_atoms();
  EXPECT_EQ(analytic_gradients->get_structure()->get_num_atoms(), num_atoms);
  EXPECT_EQ(analytic_gradients->get_values().size(),
            static_cast<Eigen::Index>(3 * num_atoms));
  EXPECT_TRUE(analytic_gradients->get_values().allFinite());
  EXPECT_TRUE(numeric_gradients->get_values().allFinite());
  EXPECT_FALSE(numeric_hessian.has_value());
  EXPECT_FALSE(analytic_hessian.has_value());
  ASSERT_TRUE(numeric_wavefunction.has_value());
  ASSERT_TRUE(analytic_wavefunction.has_value());

  const auto& numeric_values = numeric_gradients->get_values();
  const auto& analytic_values = analytic_gradients->get_values();
  ASSERT_EQ(analytic_values.size(), numeric_values.size());
  for (Eigen::Index i = 0; i < analytic_values.size(); ++i) {
    EXPECT_NEAR(analytic_values(i), numeric_values(i), gradient_tolerance)
        << "gradient component " << i;
  }
}

}  // namespace qdk::chemistry::tests::test_support

namespace test_support = qdk::chemistry::tests::test_support;
using test_support::dft_analytic_numeric_gradient_tolerance;
using test_support::expect_qdk_analytic_gradient_matches_numeric;
using test_support::expect_same_structure;
using test_support::recorded_active_spaces;
using test_support::recorded_inactive_spaces;
using test_support::recorded_localized_alpha_spaces;
using test_support::recorded_localized_beta_spaces;
using test_support::RecordingLocalizer;
using test_support::RecordingMultiConfigurationCalculator;
using test_support::ScopedFactoryRegistration;

TEST(NuclearDerivativeSettingsTest, SettingsHaveUserFacingDescriptions) {
  NuclearDerivativeSettings settings;
  for (const auto& key :
       {"energy_calculator", "orbital_solver", "hamiltonian_constructor",
        "compute_hessian", "suppress_child_algorithm_logging",
        "localize_reference_orbitals", "orbital_localizer"}) {
    EXPECT_TRUE(settings.has_description(key)) << key;
    EXPECT_FALSE(settings.get_description(key).empty()) << key;
  }
  EXPECT_FALSE(settings.has("reuse_seed_active_space"));
  EXPECT_FALSE(settings.has("finite_difference_step"));
}

TEST(NuclearDerivativeSettingsTest,
     FiniteDifferenceSettingsHaveUserFacingDescriptions) {
  auto calculator = NuclearDerivativeCalculatorFactory::create();
  for (const auto& key :
       {"reuse_seed_active_space", "finite_difference_step"}) {
    EXPECT_TRUE(calculator->settings().has_description(key)) << key;
    EXPECT_FALSE(calculator->settings().get_description(key).empty()) << key;
  }
}

TEST(NuclearDerivativeSettingsTest,
     ImplementationSettingsBelongToFiniteDifferenceOnly) {
  auto finite_difference = NuclearDerivativeCalculatorFactory::create();
  auto analytic = NuclearDerivativeCalculatorFactory::create("qdk");

  EXPECT_TRUE(finite_difference->settings().has("reuse_seed_active_space"));
  EXPECT_TRUE(finite_difference->settings().has("finite_difference_step"));
  EXPECT_FALSE(analytic->settings().has("reuse_seed_active_space"));
  EXPECT_FALSE(analytic->settings().has("finite_difference_step"));
}

TEST(NuclearDerivativeDataTest, GradientsReferenceStructureAndRoundTripJson) {
  auto structure = testing::create_water_structure();
  Eigen::VectorXd values(9);
  values << 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8;

  NuclearGradients gradients(structure, values);

  EXPECT_EQ(gradients.get_structure()->get_num_atoms(), 3);
  EXPECT_TRUE(gradients.get_values().isApprox(values));
  EXPECT_EQ(gradients.as_matrix().rows(), 3);
  EXPECT_EQ(gradients.as_matrix().cols(), 3);
  EXPECT_TRUE(
      gradients.get_atom_gradient(1).isApprox(Eigen::Vector3d(0.3, 0.4, 0.5)));
  EXPECT_THROW(gradients.get_atom_gradient(3), std::out_of_range);
  expect_same_structure(structure, gradients.get_structure());

  auto loaded = NuclearGradients::from_json(gradients.to_json());
  EXPECT_TRUE(loaded->get_values().isApprox(values));
  expect_same_structure(structure, loaded->get_structure());
}

TEST(NuclearDerivativeDataTest, HessianReferencesStructureAndRoundTripsJson) {
  auto structure = testing::create_water_structure();
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(9, 9);
  Eigen::Matrix3d atom_pair_block;
  atom_pair_block << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
  matrix.block<3, 3>(3, 6) = atom_pair_block;

  NuclearHessian hessian(structure, matrix);

  EXPECT_EQ(hessian.get_structure()->get_num_atoms(), 3);
  EXPECT_TRUE(hessian.get_matrix().isApprox(matrix));
  EXPECT_TRUE(hessian.get_atom_pair_block(1, 2).isApprox(atom_pair_block));
  EXPECT_THROW(hessian.get_atom_pair_block(3, 0), std::out_of_range);
  EXPECT_THROW(hessian.get_atom_pair_block(0, 3), std::out_of_range);
  expect_same_structure(structure, hessian.get_structure());

  auto loaded = NuclearHessian::from_json(hessian.to_json());
  EXPECT_TRUE(loaded->get_matrix().isApprox(matrix));
  expect_same_structure(structure, loaded->get_structure());
}

TEST(NuclearDerivativeCalculatorTest, FiniteDifferenceRunsRealScfForWater) {
  auto calculator = NuclearDerivativeCalculatorFactory::create();
  calculator->settings().set("compute_hessian", true);
  calculator->settings().set("finite_difference_step", 1.0e-3);

  auto [energy, gradients, hessian, wavefunction] = calculator->run(
      testing::create_water_structure(), 0, 1, std::string("sto-3g"), 0);

  ASSERT_TRUE(std::isfinite(energy));
  EXPECT_LT(energy, 0.0);
  ASSERT_NE(gradients, nullptr);
  EXPECT_EQ(gradients->get_structure()->get_num_atoms(), 3);
  EXPECT_EQ(gradients->get_values().size(), 9);
  EXPECT_TRUE(gradients->get_values().allFinite());

  auto gradient_matrix = gradients->as_matrix();
  EXPECT_NEAR(gradient_matrix.col(0).sum(), 0.0, 1.0e-5);
  EXPECT_NEAR(gradient_matrix.col(1).sum(), 0.0, 1.0e-5);
  EXPECT_NEAR(gradient_matrix.col(2).sum(), 0.0, 1.0e-5);

  ASSERT_TRUE(hessian.has_value());
  ASSERT_NE(*hessian, nullptr);
  EXPECT_EQ((*hessian)->get_structure()->get_num_atoms(), 3);
  EXPECT_EQ((*hessian)->get_matrix().rows(), 9);
  EXPECT_EQ((*hessian)->get_matrix().cols(), 9);
  EXPECT_TRUE((*hessian)->get_matrix().allFinite());
  EXPECT_TRUE((*hessian)->get_matrix().isApprox(
      (*hessian)->get_matrix().transpose(), 1.0e-8));

  ASSERT_TRUE(wavefunction.has_value());
  EXPECT_NE(*wavefunction, nullptr);
}

TEST(NuclearDerivativeCalculatorTest, NullStructureThrowsInvalidArgument) {
  for (const auto& calculator_name : {"qdk_finite_difference", "qdk"}) {
    auto calculator =
        NuclearDerivativeCalculatorFactory::create(calculator_name);
    try {
      calculator->run(nullptr, 0, 1, std::string("sto-3g"), 0);
      FAIL() << calculator_name << " accepted a null structure";
    } catch (const std::invalid_argument& ex) {
      EXPECT_STREQ(ex.what(), "Structure must not be null");
    }
  }
}

TEST(NuclearDerivativeCalculatorTest, ExcludesEcpCoreElectrons) {
  auto structure = testing::create_agh_structure();
  auto [n_alpha, n_beta] =
      qdk::chemistry::algorithms::detail::active_electron_counts(
          structure, 0, 1, std::string("def2-svp"), 0);

  EXPECT_EQ(n_alpha, 10);
  EXPECT_EQ(n_beta, 10);
}

TEST(NuclearDerivativeCalculatorTest, RejectsTooManyInactiveOrbitals) {
  for (const auto& calculator_name : {"qdk_finite_difference", "qdk"}) {
    auto calculator =
        NuclearDerivativeCalculatorFactory::create(calculator_name);
    EXPECT_THROW(calculator->run(testing::create_hydrogen_structure(), 0, 2,
                                 std::string("sto-3g"), 1),
                 std::invalid_argument);
  }
}

TEST(NuclearDerivativeCalculatorTest,
     MultiReferenceFiniteDifferenceReusesSeedActiveSpace) {
  recorded_active_spaces.clear();
  recorded_inactive_spaces.clear();
  [[maybe_unused]] ScopedFactoryRegistration<
      MultiConfigurationCalculatorFactory>
      mc_guard("_test_nuclear_derivative_recording_mc");
  MultiConfigurationCalculatorFactory::register_instance(
      []() -> MultiConfigurationCalculatorFactory::return_type {
        return std::make_unique<RecordingMultiConfigurationCalculator>();
      });

  auto structure = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [_, wavefunction] = scf_solver->run(structure, 0, 1, "sto-3g");
  auto seed_orbitals =
      testing::with_active_space(wavefunction->get_orbitals(),
                                 std::vector<size_t>{0}, std::vector<size_t>{});

  auto calculator = NuclearDerivativeCalculatorFactory::create();
  calculator->settings().set(
      "energy_calculator",
      AlgorithmRef("multi_configuration_calculator",
                   "_test_nuclear_derivative_recording_mc"));
  calculator->settings().set("finite_difference_step", 1.0e-2);

  auto [energy, gradients, hessian, result_wavefunction] =
      calculator->run(structure, 0, 1, seed_orbitals, 4);

  EXPECT_TRUE(std::isfinite(energy));
  ASSERT_NE(gradients, nullptr);
  EXPECT_FALSE(hessian.has_value());
  ASSERT_TRUE(result_wavefunction.has_value());
  ASSERT_FALSE(recorded_active_spaces.empty());
  for (const auto& active_space : recorded_active_spaces) {
    EXPECT_EQ(active_space, std::vector<size_t>{0});
  }
  for (const auto& inactive_space : recorded_inactive_spaces) {
    EXPECT_TRUE(inactive_space.empty());
  }
}

TEST(NuclearDerivativeCalculatorTest,
     MultiReferenceFiniteDifferenceLocalizesReferenceOrbitals) {
  recorded_active_spaces.clear();
  recorded_inactive_spaces.clear();
  recorded_localized_alpha_spaces.clear();
  recorded_localized_beta_spaces.clear();
  [[maybe_unused]] ScopedFactoryRegistration<
      MultiConfigurationCalculatorFactory>
      mc_guard("_test_nuclear_derivative_recording_mc");
  MultiConfigurationCalculatorFactory::register_instance(
      []() -> MultiConfigurationCalculatorFactory::return_type {
        return std::make_unique<RecordingMultiConfigurationCalculator>();
      });
  [[maybe_unused]] ScopedFactoryRegistration<LocalizerFactory> localizer_guard(
      "_test_nuclear_derivative_recording_localizer");
  LocalizerFactory::register_instance([]() -> LocalizerFactory::return_type {
    return std::make_unique<RecordingLocalizer>();
  });

  auto structure = testing::create_hydrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [_, wavefunction] = scf_solver->run(structure, -1, 1, "sto-3g");
  auto seed_orbitals =
      testing::with_active_space(wavefunction->get_orbitals(),
                                 std::vector<size_t>{0}, std::vector<size_t>{});

  auto calculator = NuclearDerivativeCalculatorFactory::create();
  calculator->settings().set(
      "energy_calculator",
      AlgorithmRef("multi_configuration_calculator",
                   "_test_nuclear_derivative_recording_mc"));
  calculator->settings().set(
      "orbital_localizer",
      AlgorithmRef("orbital_localizer",
                   "_test_nuclear_derivative_recording_localizer"));
  calculator->settings().set("localize_reference_orbitals", true);
  calculator->settings().set("finite_difference_step", 1.0e-2);

  auto [energy, gradients, hessian, result_wavefunction] =
      calculator->run(structure, -1, 1, seed_orbitals, 0);

  EXPECT_TRUE(std::isfinite(energy));
  ASSERT_NE(gradients, nullptr);
  EXPECT_FALSE(hessian.has_value());
  ASSERT_TRUE(result_wavefunction.has_value());
  ASSERT_FALSE(recorded_localized_alpha_spaces.empty());
  ASSERT_EQ(recorded_localized_alpha_spaces.size(),
            recorded_localized_beta_spaces.size());
  for (const auto& localized_space : recorded_localized_alpha_spaces) {
    EXPECT_EQ(localized_space, std::vector<size_t>{0});
  }
  for (const auto& localized_space : recorded_localized_beta_spaces) {
    EXPECT_EQ(localized_space, std::vector<size_t>{0});
  }
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientMatchesNumericHybridGgaDftForLithiumHydride) {
  expect_qdk_analytic_gradient_matches_numeric("b3lyp");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientMatchesNumericGgaDftForLithiumHydride) {
  expect_qdk_analytic_gradient_matches_numeric("pbe");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientMatchesNumericOpenShellHybridGgaDftForHydroxyl) {
  expect_qdk_analytic_gradient_matches_numeric(
      "b3lyp", dft_analytic_numeric_gradient_tolerance,
      testing::create_oh_structure(), 0, 2);
}

TEST(
    NuclearDerivativeCalculatorTest,
    QdkAnalyticGradientMatchesNumericRangeSeparatedHybridDftForLithiumHydride) {
  expect_qdk_analytic_gradient_matches_numeric(
      "wB97x", dft_analytic_numeric_gradient_tolerance);
}
