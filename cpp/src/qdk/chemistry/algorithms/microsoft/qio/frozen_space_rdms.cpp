// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "frozen_space_rdms.hpp"

#include <stdexcept>
#include <string>

namespace qdk::chemistry::algorithms::microsoft::qio::detail {

namespace {

std::size_t flat_index(std::size_t dim, std::size_t i, std::size_t j,
                       std::size_t k, std::size_t l) {
  return ((i * dim + j) * dim + k) * dim + l;
}

void validate_indices(std::size_t full_dimension,
                      std::span<const std::size_t> active_indices,
                      std::span<const std::size_t> inactive_indices) {
  std::vector<bool> assigned(full_dimension, false);
  const auto assign = [&](std::span<const std::size_t> indices,
                          const char* label) {
    for (const auto index : indices) {
      if (index >= full_dimension) {
        throw std::invalid_argument(std::string(label) +
                                    " orbital index is out of range");
      }
      if (assigned[index]) {
        throw std::invalid_argument(
            "Active and inactive orbital indices must be disjoint and unique");
      }
      assigned[index] = true;
    }
  };
  assign(active_indices, "Active");
  assign(inactive_indices, "Inactive");
}

}  // namespace

SpinResolvedRDMs build_frozen_space_rdms(
    const Eigen::MatrixXd& active_alpha, const Eigen::MatrixXd& active_beta,
    std::span<const double> active_alpha_beta, std::size_t full_dimension,
    std::span<const std::size_t> active_indices,
    std::span<const std::size_t> inactive_indices) {
  const std::size_t active_dimension = active_indices.size();
  const auto matrix_dimension = static_cast<Eigen::Index>(active_dimension);
  if (active_alpha.rows() != matrix_dimension ||
      active_alpha.cols() != matrix_dimension ||
      active_beta.rows() != matrix_dimension ||
      active_beta.cols() != matrix_dimension ||
      active_alpha_beta.size() != active_dimension * active_dimension *
                                      active_dimension * active_dimension) {
    throw std::invalid_argument(
        "Active RDM dimensions must match the active orbital count");
  }
  validate_indices(full_dimension, active_indices, inactive_indices);

  SpinResolvedRDMs result{
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(full_dimension),
                            static_cast<Eigen::Index>(full_dimension)),
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(full_dimension),
                            static_cast<Eigen::Index>(full_dimension)),
      std::vector<double>(
          full_dimension * full_dimension * full_dimension * full_dimension,
          0.0)};

  for (const auto index : inactive_indices) {
    const auto matrix_index = static_cast<Eigen::Index>(index);
    result.alpha(matrix_index, matrix_index) = 1.0;
    result.beta(matrix_index, matrix_index) = 1.0;
  }

  for (std::size_t p = 0; p < active_dimension; ++p) {
    const auto full_p = active_indices[p];
    const auto matrix_p = static_cast<Eigen::Index>(full_p);
    for (std::size_t q = 0; q < active_dimension; ++q) {
      const auto full_q = active_indices[q];
      const auto matrix_q = static_cast<Eigen::Index>(full_q);
      result.alpha(matrix_p, matrix_q) = active_alpha(
          static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q));
      result.beta(matrix_p, matrix_q) = active_beta(
          static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q));
      for (std::size_t r = 0; r < active_dimension; ++r) {
        const auto full_r = active_indices[r];
        for (std::size_t s = 0; s < active_dimension; ++s) {
          const auto full_s = active_indices[s];
          result.alpha_beta[flat_index(full_dimension, full_p, full_q, full_r,
                                       full_s)] =
              active_alpha_beta[flat_index(active_dimension, p, q, r, s)];
        }
      }
    }
  }

  for (const auto inactive : inactive_indices) {
    for (std::size_t p = 0; p < active_dimension; ++p) {
      const auto full_p = active_indices[p];
      for (std::size_t q = 0; q < active_dimension; ++q) {
        const auto full_q = active_indices[q];
        result.alpha_beta[flat_index(full_dimension, inactive, inactive, full_p,
                                     full_q)] =
            active_beta(static_cast<Eigen::Index>(p),
                        static_cast<Eigen::Index>(q));
        result.alpha_beta[flat_index(full_dimension, full_p, full_q, inactive,
                                     inactive)] =
            active_alpha(static_cast<Eigen::Index>(p),
                         static_cast<Eigen::Index>(q));
      }
    }
    for (const auto other_inactive : inactive_indices) {
      result.alpha_beta[flat_index(full_dimension, inactive, inactive,
                                   other_inactive, other_inactive)] = 1.0;
    }
  }

  return result;
}

}  // namespace qdk::chemistry::algorithms::microsoft::qio::detail
