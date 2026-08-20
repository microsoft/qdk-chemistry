// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::qio::detail {

struct SpinResolvedRDMs {
  Eigen::MatrixXd alpha;
  Eigen::MatrixXd beta;
  std::vector<double> alpha_beta;
};

/**
 * @brief Embed active RDMs into a frozen-inactive, empty-virtual orbital
 * window.
 *
 * The input state is assumed to factor into doubly occupied inactive orbitals,
 * a correlated active-space state, and empty virtual orbitals.
 */
SpinResolvedRDMs build_frozen_space_rdms(
    const Eigen::MatrixXd& active_alpha, const Eigen::MatrixXd& active_beta,
    std::span<const double> active_alpha_beta, std::size_t full_dimension,
    std::span<const std::size_t> active_indices,
    std::span<const std::size_t> inactive_indices);

}  // namespace qdk::chemistry::algorithms::microsoft::qio::detail
