// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qio.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

/**
 * @brief Single-orbital (von Neumann) entropy from orbital occupations.
 *
 * Uses the four occupation eigenvalues {1 - na - nb + d, na - d, nb - d, d}.
 * Only strictly-positive eigenvalues contribute (w -> 0+ gives w*ln(w) -> 0),
 * matching WavefunctionContainer::get_single_orbital_entropies (w > 0).
 *
 * @param occ_alpha Alpha occupation gamma_{ii} of the orbital.
 * @param occ_beta Beta occupation gamma_{ibar,ibar} of the orbital.
 * @param double_occ Double occupation Gamma_{i,ibar,i,ibar} (aabb 2-RDM diag).
 * @return The single-orbital entropy S(rho_i) >= 0.
 */
double single_orbital_entropy(double occ_alpha, double occ_beta,
                              double double_occ) {
  const double omega[4] = {1.0 - occ_alpha - occ_beta + double_occ,
                           occ_alpha - double_occ, occ_beta - double_occ,
                           double_occ};
  double entropy = 0.0;
  for (double weight : omega) {
    if (weight > 0.0) {
      entropy -= weight * std::log(weight);
    }
  }
  return entropy;
}

/**
 * @brief Flat row-major offset into an (dim x dim x dim x dim) tensor.
 *
 * @param dim Dimension of each of the four tensor axes.
 * @param i First axis index.
 * @param j Second axis index.
 * @param k Third axis index.
 * @param l Fourth axis index.
 * @return The flattened row-major offset.
 */
inline std::size_t flat_index(std::size_t dim, std::size_t i, std::size_t j,
                              std::size_t k, std::size_t l) {
  return ((i * dim + j) * dim + k) * dim + l;
}

/**
 * @brief Apply an in-place Givens rotation to one axis of a rank-4 2-RDM
 * tensor.
 *
 * The transform on the active axis is v_i <- c*v_i + s*v_j and
 * v_j <- -s*v_i + c*v_j, with all four axes of dimension @p dim.
 *
 * @param rdm_aabb Flattened (dim^4) alpha-beta 2-RDM block, modified in place.
 * @param dim Dimension of each tensor axis.
 * @param axis Axis (0-3) to which the rotation is applied.
 * @param i First orbital of the rotation plane.
 * @param j Second orbital of the rotation plane.
 * @param cos_theta Cosine of the rotation angle.
 * @param sin_theta Sine of the rotation angle.
 */
void rotate_two_rdm_axis(std::vector<double>& rdm_aabb, std::size_t dim,
                         int axis, std::size_t i, std::size_t j,
                         double cos_theta, double sin_theta) {
  const std::size_t strides[4] = {dim * dim * dim, dim * dim, dim, 1};
  const std::size_t axis_stride = strides[axis];
  int other_axes[3];
  int count = 0;
  for (int ax = 0; ax < 4; ++ax) {
    if (ax != axis) {
      other_axes[count++] = ax;
    }
  }
  // Each (x, y, z, active in {i, j}) maps to a unique flat index, so every
  // touched element is written at most once: the (x, y) iterations are
  // independent and safe to run in parallel. OpenMP schedules the outer two
  // axes; when OpenMP is disabled the pragma is ignored and the loop is serial.
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t x = 0; x < dim; ++x) {
    for (std::size_t y = 0; y < dim; ++y) {
      for (std::size_t z = 0; z < dim; ++z) {
        const std::size_t base = x * strides[other_axes[0]] +
                                 y * strides[other_axes[1]] +
                                 z * strides[other_axes[2]];
        double& elem_i = rdm_aabb[base + i * axis_stride];
        double& elem_j = rdm_aabb[base + j * axis_stride];
        const double val_i = elem_i;
        const double val_j = elem_j;
        elem_i = cos_theta * val_i + sin_theta * val_j;
        elem_j = -sin_theta * val_i + cos_theta * val_j;
      }
    }
  }
}

/**
 * @brief Apply a Givens rotation to all four axes of a 2-RDM tensor.
 *
 * @param rdm_aabb Flattened (dim^4) alpha-beta 2-RDM block, modified in place.
 * @param dim Dimension of each tensor axis.
 * @param i First orbital of the rotation plane.
 * @param j Second orbital of the rotation plane.
 * @param cos_theta Cosine of the rotation angle.
 * @param sin_theta Sine of the rotation angle.
 */
void rotate_two_rdm(std::vector<double>& rdm_aabb, std::size_t dim,
                    std::size_t i, std::size_t j, double cos_theta,
                    double sin_theta) {
  for (int axis = 0; axis < 4; ++axis) {
    rotate_two_rdm_axis(rdm_aabb, dim, axis, i, j, cos_theta, sin_theta);
  }
}

/**
 * @brief Apply a Givens rotation to both axes of a (symmetric) 1-RDM.
 *
 * @param rdm The (dim x dim) 1-RDM matrix, modified in place.
 * @param i First orbital of the rotation plane.
 * @param j Second orbital of the rotation plane.
 * @param cos_theta Cosine of the rotation angle.
 * @param sin_theta Sine of the rotation angle.
 */
void rotate_one_rdm(Eigen::MatrixXd& rdm, std::size_t i, std::size_t j,
                    double cos_theta, double sin_theta) {
  const Eigen::Index ii = static_cast<Eigen::Index>(i);
  const Eigen::Index jj = static_cast<Eigen::Index>(j);
  // Left rotation: mix rows i and j (iterate over all columns p).
  for (Eigen::Index p = 0; p < rdm.cols(); ++p) {
    const double row_i = rdm(ii, p);
    const double row_j = rdm(jj, p);
    rdm(ii, p) = cos_theta * row_i + sin_theta * row_j;
    rdm(jj, p) = -sin_theta * row_i + cos_theta * row_j;
  }
  // Right rotation: mix columns i and j (iterate over all rows p).
  for (Eigen::Index p = 0; p < rdm.rows(); ++p) {
    const double col_i = rdm(p, ii);
    const double col_j = rdm(p, jj);
    rdm(p, ii) = cos_theta * col_i + sin_theta * col_j;
    rdm(p, jj) = -sin_theta * col_i + cos_theta * col_j;
  }
}

/**
 * @brief Total single-orbital entropy sum F_QI = sum_i S(rho_i).
 *
 * @param rdm_alpha Active alpha 1-RDM (dim x dim).
 * @param rdm_beta Active beta 1-RDM (dim x dim).
 * @param rdm_aabb Flattened (dim^4) alpha-beta 2-RDM block.
 * @param dim Active-space dimension.
 * @return The summed single-orbital entropy over all active orbitals.
 */
double total_single_orbital_entropy(const Eigen::MatrixXd& rdm_alpha,
                                    const Eigen::MatrixXd& rdm_beta,
                                    const std::vector<double>& rdm_aabb,
                                    std::size_t dim) {
  double entropy = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    const Eigen::Index ii = static_cast<Eigen::Index>(i);
    entropy += single_orbital_entropy(rdm_alpha(ii, ii), rdm_beta(ii, ii),
                                      rdm_aabb[flat_index(dim, i, i, i, i)]);
  }
  return entropy;
}

/**
 * @brief Minimize the single-orbital entropy sum by gradient-free Jacobi
 * sweeps.
 *
 * Each active orbital pair (i, j) is rotated by the angle that minimizes the
 * two entropy terms that change, located by a coarse-then-fine 1-D scan. The
 * plane rotation is applied to the cached RDMs and accumulated into a single
 * unitary, which is returned.
 *
 * @param rdm_alpha Active alpha 1-RDM (dim x dim), rotated in place.
 * @param rdm_beta Active beta 1-RDM (dim x dim), rotated in place.
 * @param rdm_aabb Flattened (dim^4) alpha-beta 2-RDM block, rotated in place.
 * @param dim Active-space dimension.
 * @param max_cycles Maximum number of Jacobi sweeps.
 * @param convergence_tolerance Sweep-to-sweep entropy-sum change to stop at.
 * @param coarse_angle_step Coarse angle-scan spacing (radians) over [0, pi).
 * @param fine_samples Number of samples in the fine refinement scan.
 * @param improvement_tolerance Minimum entropy decrease to accept a rotation.
 * @return The accumulated active-space rotation U (dim x dim).
 */
Eigen::MatrixXd optimize_rotation(Eigen::MatrixXd& rdm_alpha,
                                  Eigen::MatrixXd& rdm_beta,
                                  std::vector<double>& rdm_aabb,
                                  std::size_t dim, std::size_t max_cycles,
                                  double convergence_tolerance,
                                  double coarse_angle_step, int fine_samples,
                                  double improvement_tolerance) {
  Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(
      static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
  if (dim < 2) {
    return rotation;
  }

  double entropy_prev =
      total_single_orbital_entropy(rdm_alpha, rdm_beta, rdm_aabb, dim);
  for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = i + 1; j < dim; ++j) {
        const Eigen::Index ii = static_cast<Eigen::Index>(i);
        const Eigen::Index jj = static_cast<Eigen::Index>(j);
        const std::size_t pair[2] = {i, j};
        const double a_ii = rdm_alpha(ii, ii), a_ij = rdm_alpha(ii, jj),
                     a_jj = rdm_alpha(jj, jj);
        const double b_ii = rdm_beta(ii, ii), b_ij = rdm_beta(ii, jj),
                     b_jj = rdm_beta(jj, jj);

        // Double occupation Gamma for orbital weights (w0, w1) over the pair.
        auto double_occupation = [&](double w0, double w1) {
          const double weight[2] = {w0, w1};
          double value = 0.0;
          for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
              for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                  value += weight[a] * weight[b] * weight[c] * weight[d] *
                           rdm_aabb[flat_index(dim, pair[a], pair[b], pair[c],
                                               pair[d])];
                }
              }
            }
          }
          return value;
        };

        // Pair entropy after rotating by theta: i' = (c, s), j' = (-s, c).
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
          return single_orbital_entropy(occ_alpha_i, occ_beta_i, double_occ_i) +
                 single_orbital_entropy(occ_alpha_j, occ_beta_j, double_occ_j);
        };

        const double entropy_current = pair_entropy(0.0);
        double best_theta = 0.0;
        double best_entropy = entropy_current;
        for (double theta = 0.0; theta < std::numbers::pi;
             theta += coarse_angle_step) {
          const double value = pair_entropy(theta);
          if (value < best_entropy) {
            best_entropy = value;
            best_theta = theta;
          }
        }
        // Clamp the fine-scan window to the documented domain [0, pi) so the
        // refined angle never leaves the intended periodic range.
        const double theta_lo = std::max(0.0, best_theta - coarse_angle_step);
        const double theta_hi =
            std::min(std::numbers::pi, best_theta + coarse_angle_step);
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

        if (best_entropy < entropy_current - improvement_tolerance) {
          const double cos_theta = std::cos(best_theta);
          const double sin_theta = std::sin(best_theta);
          rotate_one_rdm(rdm_alpha, i, j, cos_theta, sin_theta);
          rotate_one_rdm(rdm_beta, i, j, cos_theta, sin_theta);
          rotate_two_rdm(rdm_aabb, dim, i, j, cos_theta, sin_theta);
          // Accumulate U <- U * G(i, j; c, s) (in-place column update).
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
  return rotation;
}

}  // namespace detail

std::shared_ptr<data::Wavefunction> QIOLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();

  // QIO produces a single spatial orbital set.
  if (loc_indices_a != loc_indices_b) {
    throw std::invalid_argument(
        "loc_indices_a and loc_indices_b must be identical for QIO "
        "localization.");
  }
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }
  if (std::adjacent_find(loc_indices_a.begin(), loc_indices_a.end()) !=
      loc_indices_a.end()) {
    throw std::invalid_argument("loc_indices_a contains duplicate indices");
  }

  // Empty selection is a no-op, but still returns the standard single-reference
  // (Aufbau determinant) carrier for consistency with the Localizer contract.
  if (loc_indices_a.empty()) {
    return algorithms::detail::new_aufbau_determinant_wavefunction(wavefunction,
                                                                   orbitals);
  }

  if (!orbitals->is_restricted()) {
    throw std::invalid_argument(
        "QIOLocalizer requires a single spatial orbital set (RHF/ROHF); "
        "unrestricted (UHF) orbitals are not supported.");
  }

  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "QIOLocalizer requires an active space to be defined in the orbitals.");
  }

  // The output Orbitals carry the AO overlap matrix; require it up front so a
  // missing overlap fails as std::invalid_argument (consistent with the other
  // input checks) rather than a std::runtime_error from get_overlap_matrix().
  if (!orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "QIOLocalizer requires an overlap matrix to be available in the "
        "orbitals.");
  }

  const auto active_index_set = orbitals->active_indices();
  const auto active_indices_a =
      data::spin_channel_indices(active_index_set, data::axes::alpha());
  const auto active_indices_b =
      data::spin_channel_indices(active_index_set, data::axes::beta());
  if (loc_indices_a != active_indices_a || loc_indices_b != active_indices_b) {
    throw std::invalid_argument(
        "QIOLocalizer requires loc_indices_a and loc_indices_b to match the "
        "orbitals' active-space indices.");
  }

  const std::size_t num_molecular_orbitals =
      orbitals->get_num_molecular_orbitals();
  if (loc_indices_a.back() >= num_molecular_orbitals) {
    throw std::invalid_argument(
        "loc_indices_a contains invalid orbital index >= "
        "num_molecular_orbitals");
  }

  // Single-orbital entropies require correlated spin-dependent RDMs.
  if (!wavefunction->has_one_rdm_spin_dependent() ||
      !wavefunction->has_two_rdm_spin_dependent()) {
    throw std::invalid_argument(
        "QIOLocalizer requires spin-dependent active 1- and 2-RDMs in the "
        "wavefunction.");
  }

  const std::size_t n = active_indices_a.size();

  // Extract the active alpha/beta 1-RDM and the alpha-beta (aabb) 2-RDM block.
  auto [rdm_aa_variant, rdm_bb_variant] =
      wavefunction->get_active_one_rdm_spin_dependent();
  auto [rdm_aaaa_variant, rdm_aabb_variant, rdm_bbbb_variant] =
      wavefunction->get_active_two_rdm_spin_dependent();
  (void)rdm_aaaa_variant;  // QIO only needs the alpha-beta (aabb) block.
  (void)rdm_bbbb_variant;

  const auto* rdm_aa = std::get_if<Eigen::MatrixXd>(&rdm_aa_variant);
  const auto* rdm_bb = std::get_if<Eigen::MatrixXd>(&rdm_bb_variant);
  const auto* rdm_aabb = std::get_if<Eigen::VectorXd>(&rdm_aabb_variant);
  if (!rdm_aa || !rdm_bb || !rdm_aabb) {
    throw std::invalid_argument(
        "QIOLocalizer requires real-valued active RDMs.");
  }
  if (static_cast<std::size_t>(rdm_aa->rows()) != n ||
      static_cast<std::size_t>(rdm_aa->cols()) != n ||
      static_cast<std::size_t>(rdm_bb->rows()) != n ||
      static_cast<std::size_t>(rdm_bb->cols()) != n ||
      static_cast<std::size_t>(rdm_aabb->size()) != n * n * n * n) {
    throw std::invalid_argument(
        "Active RDM dimensions do not match the active-space size.");
  }

  Eigen::MatrixXd rdm_alpha = *rdm_aa;
  Eigen::MatrixXd rdm_beta = *rdm_bb;
  std::vector<double> rdm_aabb_flat(rdm_aabb->data(),
                                    rdm_aabb->data() + rdm_aabb->size());

  // Minimize the single-orbital entropy sum -> accumulated rotation U.
  const auto max_cycles =
      static_cast<std::size_t>(_settings->get<int64_t>("max_cycles"));
  const double convergence_tolerance =
      _settings->get<double>("convergence_tolerance");
  const double coarse_angle_step = _settings->get<double>("coarse_angle_step");
  const auto fine_samples =
      static_cast<int>(_settings->get<int64_t>("fine_samples"));
  const double improvement_tolerance =
      _settings->get<double>("improvement_tolerance");
  // BoundConstraint range checks pass NaN (every comparison with NaN is false),
  // so reject non-finite double settings explicitly.
  const auto require_finite = [](const char* setting_name, double value) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string("QIOLocalizer setting '") +
                                  setting_name + "' must be finite.");
    }
  };
  require_finite("convergence_tolerance", convergence_tolerance);
  require_finite("coarse_angle_step", coarse_angle_step);
  require_finite("improvement_tolerance", improvement_tolerance);
  const Eigen::MatrixXd u = detail::optimize_rotation(
      rdm_alpha, rdm_beta, rdm_aabb_flat, n, max_cycles, convergence_tolerance,
      coarse_angle_step, fine_samples, improvement_tolerance);

  // Apply the rotation to the active orbital columns (alpha == beta basis).
  const auto& coeffs_alpha = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  Eigen::MatrixXd selected_coeffs(coeffs_alpha.rows(),
                                  static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) {
    selected_coeffs.col(static_cast<Eigen::Index>(i)) =
        coeffs_alpha.col(static_cast<Eigen::Index>(active_indices_a[i]));
  }
  const Eigen::MatrixXd rotated_coeffs = selected_coeffs * u;

  Eigen::MatrixXd coeffs = coeffs_alpha;
  for (std::size_t i = 0; i < n; ++i) {
    coeffs.col(static_cast<Eigen::Index>(active_indices_a[i])) =
        rotated_coeffs.col(static_cast<Eigen::Index>(i));
  }

  // Create output orbitals, preserving active/inactive metadata. Energies are
  // invalidated by the rotation.
  auto new_orbitals = std::make_shared<data::Orbitals>(
      coeffs,
      std::nullopt,  // no energies for entropy-optimized orbitals
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      orbitals->active_indices(), orbitals->inactive_indices());

  // Attach the rotated spin-traced active 1-RDM (alpha + beta) as a payload so
  // downstream consumers see the density in the new orbital basis. Unlike the
  // natural-orbital localizer, QIO does not diagonalize the 1-RDM, so this
  // payload is generally non-diagonal.
  const Eigen::MatrixXd rotated_one_rdm_spin_traced = rdm_alpha + rdm_beta;
  return algorithms::detail::new_aufbau_determinant_wavefunction(
      wavefunction, new_orbitals,
      data::ContainerTypes::MatrixVariant(rotated_one_rdm_spin_traced));
}

}  // namespace qdk::chemistry::algorithms::microsoft
