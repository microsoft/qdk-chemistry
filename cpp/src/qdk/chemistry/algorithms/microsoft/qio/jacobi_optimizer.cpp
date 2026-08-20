// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "jacobi_optimizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <qdk/chemistry/data/orbital_entropy.hpp>
#include <utility>

namespace qdk::chemistry::algorithms::microsoft::qio::detail {

namespace {

std::size_t flat_index(std::size_t dim, std::size_t i, std::size_t j,
                       std::size_t k, std::size_t l) {
  return ((i * dim + j) * dim + k) * dim + l;
}

void rotate_two_rdm_axis(std::vector<double>& rdm_aabb, std::size_t dim,
                         int axis, std::size_t i, std::size_t j,
                         double cos_theta, double sin_theta) {
  const std::array<std::size_t, 4> strides{dim * dim * dim, dim * dim, dim, 1};
  const std::size_t axis_stride = strides[axis];
  const std::size_t axis_span = dim * axis_stride;
  const std::size_t block_count = dim * dim * dim;

#pragma omp parallel for schedule(static)
  for (std::size_t block = 0; block < block_count; ++block) {
    const std::size_t base =
        (block / axis_stride) * axis_span + block % axis_stride;
    double& elem_i = rdm_aabb[base + i * axis_stride];
    double& elem_j = rdm_aabb[base + j * axis_stride];
    const double val_i = elem_i;
    const double val_j = elem_j;
    elem_i = cos_theta * val_i + sin_theta * val_j;
    elem_j = -sin_theta * val_i + cos_theta * val_j;
  }
}

void rotate_two_rdm(std::vector<double>& rdm_aabb, std::size_t dim,
                    std::size_t i, std::size_t j, double cos_theta,
                    double sin_theta) {
  for (int axis = 0; axis < 4; ++axis) {
    rotate_two_rdm_axis(rdm_aabb, dim, axis, i, j, cos_theta, sin_theta);
  }
}

void rotate_one_rdm(Eigen::MatrixXd& rdm, std::size_t i, std::size_t j,
                    double cos_theta, double sin_theta) {
  const Eigen::Index ii = static_cast<Eigen::Index>(i);
  const Eigen::Index jj = static_cast<Eigen::Index>(j);
  for (Eigen::Index p = 0; p < rdm.cols(); ++p) {
    const double row_i = rdm(ii, p);
    const double row_j = rdm(jj, p);
    rdm(ii, p) = cos_theta * row_i + sin_theta * row_j;
    rdm(jj, p) = -sin_theta * row_i + cos_theta * row_j;
  }
  for (Eigen::Index p = 0; p < rdm.rows(); ++p) {
    const double col_i = rdm(p, ii);
    const double col_j = rdm(p, jj);
    rdm(p, ii) = cos_theta * col_i + sin_theta * col_j;
    rdm(p, jj) = -sin_theta * col_i + cos_theta * col_j;
  }
}

double total_single_orbital_entropy(const Eigen::MatrixXd& rdm_alpha,
                                    const Eigen::MatrixXd& rdm_beta,
                                    const std::vector<double>& rdm_aabb,
                                    std::size_t dim) {
  double entropy = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    const Eigen::Index ii = static_cast<Eigen::Index>(i);
    entropy += data::detail::single_orbital_entropy(
        rdm_alpha(ii, ii), rdm_beta(ii, ii),
        rdm_aabb[flat_index(dim, i, i, i, i)]);
  }
  return entropy;
}

}  // namespace

RotationOptimizationResult optimize_rotation(
    Eigen::MatrixXd rdm_alpha, Eigen::MatrixXd rdm_beta,
    std::vector<double> rdm_aabb, std::size_t dim, std::size_t max_cycles,
    double convergence_tolerance, double coarse_angle_step, int fine_samples,
    double improvement_tolerance) {
  Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(
      static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
  if (dim < 2) {
    return {std::move(rotation), std::move(rdm_alpha), std::move(rdm_beta)};
  }

  double entropy_prev =
      total_single_orbital_entropy(rdm_alpha, rdm_beta, rdm_aabb, dim);
  constexpr double angle_period = std::numbers::pi / 2.0;
  for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = i + 1; j < dim; ++j) {
        const Eigen::Index ii = static_cast<Eigen::Index>(i);
        const Eigen::Index jj = static_cast<Eigen::Index>(j);
        const std::array<std::size_t, 2> pair{i, j};
        const double a_ii = rdm_alpha(ii, ii), a_ij = rdm_alpha(ii, jj),
                     a_jj = rdm_alpha(jj, jj);
        const double b_ii = rdm_beta(ii, ii), b_ij = rdm_beta(ii, jj),
                     b_jj = rdm_beta(jj, jj);

        std::array<double, 5> double_occupation_coefficients{};
        for (std::size_t a = 0; a < 2; ++a) {
          for (std::size_t b = 0; b < 2; ++b) {
            for (std::size_t c = 0; c < 2; ++c) {
              for (std::size_t d = 0; d < 2; ++d) {
                double_occupation_coefficients[a + b + c + d] +=
                    rdm_aabb[flat_index(dim, pair[a], pair[b], pair[c],
                                        pair[d])];
              }
            }
          }
        }

        auto double_occupation = [&](double w0, double w1) {
          const double w0_squared = w0 * w0;
          const double w1_squared = w1 * w1;
          return double_occupation_coefficients[0] * w0_squared * w0_squared +
                 double_occupation_coefficients[1] * w0_squared * w0 * w1 +
                 double_occupation_coefficients[2] * w0_squared * w1_squared +
                 double_occupation_coefficients[3] * w0 * w1_squared * w1 +
                 double_occupation_coefficients[4] * w1_squared * w1_squared;
        };

        auto pair_entropy = [&](double theta) {
          const double cos_theta = std::cos(theta);
          const double sin_theta = std::sin(theta);
          const double occ_alpha_i = cos_theta * cos_theta * a_ii +
                                     2.0 * cos_theta * sin_theta * a_ij +
                                     sin_theta * sin_theta * a_jj;
          const double occ_beta_i = cos_theta * cos_theta * b_ii +
                                    2.0 * cos_theta * sin_theta * b_ij +
                                    sin_theta * sin_theta * b_jj;
          const double occ_alpha_j = sin_theta * sin_theta * a_ii -
                                     2.0 * cos_theta * sin_theta * a_ij +
                                     cos_theta * cos_theta * a_jj;
          const double occ_beta_j = sin_theta * sin_theta * b_ii -
                                    2.0 * cos_theta * sin_theta * b_ij +
                                    cos_theta * cos_theta * b_jj;
          const double double_occ_i = double_occupation(cos_theta, sin_theta);
          const double double_occ_j = double_occupation(-sin_theta, cos_theta);
          return data::detail::single_orbital_entropy(occ_alpha_i, occ_beta_i,
                                                      double_occ_i) +
                 data::detail::single_orbital_entropy(occ_alpha_j, occ_beta_j,
                                                      double_occ_j);
        };

        const double entropy_current = pair_entropy(0.0);
        double best_theta = 0.0;
        double best_entropy = entropy_current;
        for (double theta = 0.0; theta < angle_period;
             theta += coarse_angle_step) {
          const double value = pair_entropy(theta);
          if (value < best_entropy) {
            best_entropy = value;
            best_theta = theta;
          }
        }
        const double theta_lo = best_theta - coarse_angle_step;
        const double theta_hi = best_theta + coarse_angle_step;
        for (int k = 0; k < fine_samples; ++k) {
          const double theta =
              theta_lo +
              (theta_hi - theta_lo) * k / static_cast<double>(fine_samples - 1);
          const double value = pair_entropy(theta);
          if (value < best_entropy) {
            best_entropy = value;
            best_theta = theta;
          }
        }
        best_theta = std::fmod(best_theta, angle_period);
        if (best_theta < 0.0) {
          best_theta += angle_period;
        }

        if (best_entropy < entropy_current - improvement_tolerance) {
          const double cos_theta = std::cos(best_theta);
          const double sin_theta = std::sin(best_theta);
          rotate_one_rdm(rdm_alpha, i, j, cos_theta, sin_theta);
          rotate_one_rdm(rdm_beta, i, j, cos_theta, sin_theta);
          rotate_two_rdm(rdm_aabb, dim, i, j, cos_theta, sin_theta);
          for (Eigen::Index p = 0; p < rotation.rows(); ++p) {
            const double rot_i = rotation(p, ii);
            const double rot_j = rotation(p, jj);
            rotation(p, ii) = cos_theta * rot_i + sin_theta * rot_j;
            rotation(p, jj) = -sin_theta * rot_i + cos_theta * rot_j;
          }
        }
      }
    }
    const double entropy_now =
        total_single_orbital_entropy(rdm_alpha, rdm_beta, rdm_aabb, dim);
    if (std::abs(entropy_prev - entropy_now) < convergence_tolerance) {
      break;
    }
    entropy_prev = entropy_now;
  }
  return {std::move(rotation), std::move(rdm_alpha), std::move(rdm_beta)};
}

}  // namespace qdk::chemistry::algorithms::microsoft::qio::detail
