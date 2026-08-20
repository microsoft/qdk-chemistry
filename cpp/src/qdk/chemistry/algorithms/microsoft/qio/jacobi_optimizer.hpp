// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::qio::detail {

struct RotationOptimizationResult {
  Eigen::MatrixXd rotation;
  Eigen::MatrixXd rdm_alpha;
  Eigen::MatrixXd rdm_beta;
};

RotationOptimizationResult optimize_rotation(
    Eigen::MatrixXd rdm_alpha, Eigen::MatrixXd rdm_beta,
    std::vector<double> rdm_aabb, std::size_t dim, std::size_t max_cycles,
    double convergence_tolerance, double coarse_angle_step, int fine_samples,
    double improvement_tolerance);

}  // namespace qdk::chemistry::algorithms::microsoft::qio::detail
