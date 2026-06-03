// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

std::vector<std::vector<size_t>> recorded_active_spaces;
std::vector<std::vector<size_t>> recorded_inactive_spaces;

std::string determinant_string(size_t n_active_orbitals, unsigned int n_alpha,
                               unsigned int n_beta) {
  std::string determinant(n_active_orbitals, '0');
  for (auto& occupation : determinant) {
    if (n_alpha > 0 && n_beta > 0) {
      occupation = '2';
      --n_alpha;
      --n_beta;
    } else if (n_alpha > 0) {
      occupation = 'u';
      --n_alpha;
    } else if (n_beta > 0) {
      occupation = 'd';
      --n_beta;
    }
  }
  return determinant;
}

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

    auto determinant = Configuration(determinant_string(
        orbitals->get_active_space_indices().first.size(), n_alpha, n_beta));
    auto container =
        std::make_unique<SlaterDeterminantContainer>(determinant, orbitals);
    return {0.0, std::make_shared<Wavefunction>(std::move(container))};
  }
};

std::shared_ptr<Structure> create_h2_structure() {
  std::vector<Eigen::Vector3d> coordinates = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
  std::vector<Element> elements = {Element::H, Element::H};
  return std::make_shared<Structure>(coordinates, elements);
}

void expect_same_structure(const std::shared_ptr<Structure>& left,
                           const std::shared_ptr<Structure>& right) {
  ASSERT_NE(left, nullptr);
  ASSERT_NE(right, nullptr);
  ASSERT_EQ(left->get_num_atoms(), right->get_num_atoms());
  EXPECT_TRUE(
      left->get_coordinates().isApprox(right->get_coordinates(), 1.0e-12));
  EXPECT_EQ(left->get_elements(), right->get_elements());
}

void expect_qdk_analytic_gradient_runs_for_functional(
    const std::string& functional) {
  auto calculator = NuclearDerivativeCalculatorFactory::create("qdk");
  AlgorithmRef scf_ref("scf_solver", "qdk");
  scf_ref.get_settings()->set("method", functional);
  calculator->settings().set("energy_calculator", scf_ref);

  auto [energy, gradients, hessian, wavefunction] =
      calculator->run(create_h2_structure(), 0, 1, std::string("sto-3g"));

  ASSERT_TRUE(std::isfinite(energy));
  EXPECT_LT(energy, 0.0);
  ASSERT_NE(gradients, nullptr);
  EXPECT_EQ(gradients->get_num_atoms(), 2);
  EXPECT_EQ(gradients->get_values().size(), 6);
  EXPECT_TRUE(gradients->get_values().allFinite());

  auto gradient_matrix = gradients->as_matrix();
  EXPECT_NEAR(gradient_matrix.col(0).sum(), 0.0, 1.0e-8);
  EXPECT_NEAR(gradient_matrix.col(1).sum(), 0.0, 1.0e-8);
  EXPECT_NEAR(gradient_matrix.col(2).sum(), 0.0, 1.0e-8);

  EXPECT_FALSE(hessian.has_value());
  ASSERT_TRUE(wavefunction.has_value());
  EXPECT_NE(*wavefunction, nullptr);
}

NuclearDerivativeResult run_qdk_derivative_for_functional(
    const std::string& calculator_name, const std::string& functional) {
  auto calculator = NuclearDerivativeCalculatorFactory::create(calculator_name);
  AlgorithmRef scf_ref("scf_solver", "qdk");
  scf_ref.get_settings()->set("method", functional);
  calculator->settings().set("energy_calculator", scf_ref);
  calculator->settings().set("finite_difference_step", 1.0e-2);

  return calculator->run(create_h2_structure(), 0, 1, std::string("sto-3g"));
}

void expect_qdk_analytic_gradient_matches_numeric(
    const std::string& functional) {
  auto [numeric_energy, numeric_gradients, numeric_hessian,
        numeric_wavefunction] =
      run_qdk_derivative_for_functional("finite_difference", functional);
  auto [analytic_energy, analytic_gradients, analytic_hessian,
        analytic_wavefunction] =
      run_qdk_derivative_for_functional("qdk", functional);

  EXPECT_NEAR(analytic_energy, numeric_energy, 1.0e-8);
  ASSERT_NE(numeric_gradients, nullptr);
  ASSERT_NE(analytic_gradients, nullptr);
  EXPECT_FALSE(numeric_hessian.has_value());
  EXPECT_FALSE(analytic_hessian.has_value());
  ASSERT_TRUE(numeric_wavefunction.has_value());
  ASSERT_TRUE(analytic_wavefunction.has_value());

  const auto& numeric_values = numeric_gradients->get_values();
  const auto& analytic_values = analytic_gradients->get_values();
  ASSERT_EQ(analytic_values.size(), numeric_values.size());
  for (Eigen::Index i = 0; i < analytic_values.size(); ++i) {
    EXPECT_NEAR(analytic_values(i), numeric_values(i), 1.0e-3)
        << "gradient component " << i;
  }
}

}  // namespace

TEST(NuclearDerivativeSettingsTest, SettingsHaveUserFacingDescriptions) {
  NuclearDerivativeSettings settings;
  for (const auto& key :
       {"energy_calculator", "orbital_solver", "hamiltonian_constructor",
        "compute_hessian", "finite_difference_step", "symmetrize_hessian",
        "reuse_seed_active_space", "localize_reference_orbitals",
        "orbital_localizer", "n_active_alpha_electrons",
        "n_active_beta_electrons"}) {
    EXPECT_TRUE(settings.has_description(key)) << key;
    EXPECT_FALSE(settings.get_description(key).empty()) << key;
  }
}

TEST(NuclearDerivativeDataTest, GradientsReferenceStructureAndRoundTripJson) {
  auto structure = create_h2_structure();
  Eigen::VectorXd values(6);
  values << 0.0, 0.1, 0.2, 0.3, 0.4, 0.5;

  NuclearGradients gradients(structure, values);

  EXPECT_EQ(gradients.get_num_atoms(), 2);
  EXPECT_TRUE(gradients.get_values().isApprox(values));
  EXPECT_EQ(gradients.as_matrix().rows(), 2);
  EXPECT_EQ(gradients.as_matrix().cols(), 3);
  expect_same_structure(structure, gradients.get_structure());

  auto loaded = NuclearGradients::from_json(gradients.to_json());
  EXPECT_TRUE(loaded->get_values().isApprox(values));
  expect_same_structure(structure, loaded->get_structure());
}

TEST(NuclearDerivativeDataTest, HessianReferencesStructureAndRoundTripsJson) {
  auto structure = create_h2_structure();
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(6, 6);

  NuclearHessian hessian(structure, matrix);

  EXPECT_EQ(hessian.get_num_atoms(), 2);
  EXPECT_TRUE(hessian.get_matrix().isApprox(matrix));
  expect_same_structure(structure, hessian.get_structure());

  auto loaded = NuclearHessian::from_json(hessian.to_json());
  EXPECT_TRUE(loaded->get_matrix().isApprox(matrix));
  expect_same_structure(structure, loaded->get_structure());
}

TEST(NuclearDerivativeCalculatorTest, FiniteDifferenceRunsRealScfForH2) {
  auto calculator = NuclearDerivativeCalculatorFactory::create();
  calculator->settings().set("compute_hessian", true);
  calculator->settings().set("finite_difference_step", 1.0e-3);

  auto [energy, gradients, hessian, wavefunction] =
      calculator->run(create_h2_structure(), 0, 1, std::string("sto-3g"));

  ASSERT_TRUE(std::isfinite(energy));
  EXPECT_LT(energy, 0.0);
  ASSERT_NE(gradients, nullptr);
  EXPECT_EQ(gradients->get_num_atoms(), 2);
  EXPECT_EQ(gradients->get_values().size(), 6);
  EXPECT_TRUE(gradients->get_values().allFinite());

  auto gradient_matrix = gradients->as_matrix();
  EXPECT_NEAR(gradient_matrix.col(0).sum(), 0.0, 1.0e-5);
  EXPECT_NEAR(gradient_matrix.col(1).sum(), 0.0, 1.0e-5);
  EXPECT_NEAR(gradient_matrix.col(2).sum(), 0.0, 1.0e-5);

  ASSERT_TRUE(hessian.has_value());
  ASSERT_NE(*hessian, nullptr);
  EXPECT_EQ((*hessian)->get_num_atoms(), 2);
  EXPECT_EQ((*hessian)->get_matrix().rows(), 6);
  EXPECT_EQ((*hessian)->get_matrix().cols(), 6);
  EXPECT_TRUE((*hessian)->get_matrix().allFinite());
  EXPECT_TRUE((*hessian)->get_matrix().isApprox(
      (*hessian)->get_matrix().transpose(), 1.0e-8));

  ASSERT_TRUE(wavefunction.has_value());
  EXPECT_NE(*wavefunction, nullptr);
}

TEST(NuclearDerivativeCalculatorTest,
     MultiReferenceFiniteDifferenceReusesSeedActiveSpace) {
  recorded_active_spaces.clear();
  recorded_inactive_spaces.clear();
  MultiConfigurationCalculatorFactory::unregister_instance(
      "_test_nuclear_derivative_recording_mc");
  MultiConfigurationCalculatorFactory::register_instance(
      []() -> MultiConfigurationCalculatorFactory::return_type {
        return std::make_unique<RecordingMultiConfigurationCalculator>();
      });

  auto structure = create_h2_structure();
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
  calculator->settings().set("n_active_alpha_electrons", 1);
  calculator->settings().set("n_active_beta_electrons", 1);
  calculator->settings().set("finite_difference_step", 1.0e-2);

  auto [energy, gradients, hessian, result_wavefunction] =
      calculator->run(structure, 0, 1, seed_orbitals);

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

  MultiConfigurationCalculatorFactory::unregister_instance(
      "_test_nuclear_derivative_recording_mc");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientRunsRealNonHybridDftForH2) {
  expect_qdk_analytic_gradient_runs_for_functional("pbe");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientRunsRealHybridDftForH2) {
  expect_qdk_analytic_gradient_runs_for_functional("b3lyp");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientRunsRealRangeSeparatedHybridDftForH2) {
  expect_qdk_analytic_gradient_runs_for_functional("wB97x");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientMatchesNumericHybridGgaDftForH2) {
  expect_qdk_analytic_gradient_matches_numeric("b3lyp");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientMatchesNumericGgaDftForH2) {
  expect_qdk_analytic_gradient_matches_numeric("pbe");
}

TEST(NuclearDerivativeCalculatorTest,
     QdkAnalyticGradientMatchesNumericRangeSeparatedHybridDftForH2) {
  expect_qdk_analytic_gradient_matches_numeric("wB97x");
}
