// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <qdk/chemistry/algorithms/microsoft/qio/jacobi_optimizer.hpp>
#include <stdexcept>
#include <vector>

using qdk::chemistry::algorithms::microsoft::qio::detail::optimize_rotation;
using qdk::chemistry::algorithms::microsoft::qio::detail::OrbitalPair;

namespace {

struct TwoOrbitalRdm {
  Eigen::MatrixXd alpha;
  Eigen::MatrixXd beta;
  std::vector<double> aabb;
};

TwoOrbitalRdm delocalized_one_alpha_electron() {
  Eigen::MatrixXd alpha(2, 2);
  alpha << 0.5, 0.5, 0.5, 0.5;
  return {std::move(alpha), Eigen::MatrixXd::Zero(2, 2),
          std::vector<double>(16, 0.0)};
}

}  // namespace

TEST(QIOJacobiOptimizerTest, FullObjectiveLocalizesOneElectron) {
  auto rdm = delocalized_one_alpha_electron();
  auto result =
      optimize_rotation(std::move(rdm.alpha), std::move(rdm.beta),
                        std::move(rdm.aabb), 2, 20, 1e-12, 0.02, 201, 1e-12);

  EXPECT_GT(result.initial_objective, 1.0);
  EXPECT_NEAR(result.final_objective, 0.0, 1e-8);
  EXPECT_LT(result.final_objective, result.initial_objective);
  EXPECT_TRUE(result.converged);
  EXPECT_GT(result.cycles, 0);
  EXPECT_TRUE((result.rotation.transpose() * result.rotation)
                  .isApprox(Eigen::MatrixXd::Identity(2, 2), 1e-12));
}

TEST(QIOJacobiOptimizerTest, SubsetObjectiveUsesCrossBoundaryPair) {
  auto rdm = delocalized_one_alpha_electron();
  const std::vector<std::size_t> objective_indices{0};
  const std::vector<OrbitalPair> rotation_pairs{{0, 1}};
  auto result = optimize_rotation(std::move(rdm.alpha), std::move(rdm.beta),
                                  std::move(rdm.aabb), 2, objective_indices,
                                  rotation_pairs, 20, 1e-12, 0.02, 201, 1e-12);

  EXPECT_NEAR(result.initial_objective, std::log(2.0), 1e-12);
  EXPECT_NEAR(result.final_objective, 0.0, 1e-8);
}

TEST(QIOJacobiOptimizerTest, RejectsInvalidObjectiveIndex) {
  auto rdm = delocalized_one_alpha_electron();
  const std::vector<std::size_t> objective_indices{2};
  const std::vector<OrbitalPair> rotation_pairs{{0, 1}};

  EXPECT_THROW(optimize_rotation(std::move(rdm.alpha), std::move(rdm.beta),
                                 std::move(rdm.aabb), 2, objective_indices,
                                 rotation_pairs, 20, 1e-12, 0.02, 201, 1e-12),
               std::invalid_argument);
}
