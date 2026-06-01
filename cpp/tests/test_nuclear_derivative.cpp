// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <vector>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

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

}  // namespace

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

TEST(NuclearDerivativeCalculatorTest, QdkAnalyticGradientRunsRealDftForH2) {
  auto calculator = NuclearDerivativeCalculatorFactory::create("qdk");
  AlgorithmRef scf_ref("scf_solver", "qdk");
  scf_ref.get_settings()->set("method", std::string("pbe"));
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
