// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::qio::detail {

struct OrbitalPair {
  std::size_t first;
  std::size_t second;
};

/**
 * @brief Result of a Jacobi orbital-rotation optimization.
 */
struct RotationOptimizationResult {
  Eigen::MatrixXd rotation;
  Eigen::MatrixXd rdm_alpha;
  Eigen::MatrixXd rdm_beta;
  double initial_objective;
  double final_objective;
  std::size_t cycles;
  bool converged;
};

/**
 * @brief Minimize a selected single-orbital entropy sum over allowed pairs.
 *
 * The RDMs must span the full optimization window. Objective indices select
 * which orbital entropies contribute to the cost, while rotation pairs select
 * which Givens rotations may be applied.
 */
RotationOptimizationResult optimize_rotation(
    Eigen::MatrixXd rdm_alpha, Eigen::MatrixXd rdm_beta,
    std::vector<double> rdm_aabb, std::size_t dim,
    std::span<const std::size_t> objective_indices,
    std::span<const OrbitalPair> rotation_pairs, std::size_t max_cycles,
    double convergence_tolerance, double coarse_angle_step, int fine_samples,
    double improvement_tolerance);

/**
 * @brief Minimize the entropy sum over all orbitals and all unique pairs.
 */
RotationOptimizationResult optimize_rotation(
    Eigen::MatrixXd rdm_alpha, Eigen::MatrixXd rdm_beta,
    std::vector<double> rdm_aabb, std::size_t dim, std::size_t max_cycles,
    double convergence_tolerance, double coarse_angle_step, int fine_samples,
    double improvement_tolerance);

}  // namespace qdk::chemistry::algorithms::microsoft::qio::detail
