// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "active_space_qio.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../qio/jacobi_optimizer.hpp"

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> ActiveSpaceQIOLocalizer::_run_impl(
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
        "ActiveSpaceQIOLocalizer requires a single spatial orbital set "
        "(RHF/ROHF); "
        "unrestricted (UHF) orbitals are not supported.");
  }

  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "ActiveSpaceQIOLocalizer requires an active space to be defined in the "
        "orbitals.");
  }

  // The output Orbitals carry the AO overlap matrix; require it up front so a
  // missing overlap fails as std::invalid_argument (consistent with the other
  // input checks) rather than a std::runtime_error from get_overlap_matrix().
  if (!orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "ActiveSpaceQIOLocalizer requires an overlap matrix to be available in "
        "the "
        "orbitals.");
  }

  const auto active_index_set = orbitals->active_indices();
  const auto active_indices_a =
      data::spin_channel_indices(active_index_set, data::axes::alpha());
  const auto active_indices_b =
      data::spin_channel_indices(active_index_set, data::axes::beta());
  if (loc_indices_a != active_indices_a || loc_indices_b != active_indices_b) {
    throw std::invalid_argument(
        "ActiveSpaceQIOLocalizer requires loc_indices_a and loc_indices_b to "
        "match the "
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
        "ActiveSpaceQIOLocalizer requires spin-dependent active 1- and 2-RDMs "
        "in the "
        "wavefunction.");
  }

  const std::size_t n = active_indices_a.size();

  // Borrow the three required blocks directly, then make one mutable working
  // copy of each. The legacy by-value accessors would also copy the unused
  // aaaa and bbbb 2-RDM blocks before these working copies are made.
  const auto* active_one_rdm =
      std::get_if<data::SymmetryBlockedTensor<2, double>>(
          &wavefunction->active_one_rdm());
  const auto* active_two_rdm =
      std::get_if<data::SymmetryBlockedTensor<4, double>>(
          &wavefunction->active_two_rdm());
  if (!active_one_rdm || !active_two_rdm) {
    throw std::invalid_argument(
        "ActiveSpaceQIOLocalizer requires real-valued active RDMs.");
  }
  const auto& rdm_aa =
      active_one_rdm->block({data::axes::alpha(), data::axes::alpha()});
  const auto& rdm_bb =
      active_one_rdm->block({data::axes::beta(), data::axes::beta()});
  const auto& rdm_aabb =
      active_two_rdm->block({data::axes::alpha(), data::axes::alpha(),
                             data::axes::beta(), data::axes::beta()});
  if (static_cast<std::size_t>(rdm_aa.rows()) != n ||
      static_cast<std::size_t>(rdm_aa.cols()) != n ||
      static_cast<std::size_t>(rdm_bb.rows()) != n ||
      static_cast<std::size_t>(rdm_bb.cols()) != n ||
      static_cast<std::size_t>(rdm_aabb.size()) != n * n * n * n) {
    throw std::invalid_argument(
        "Active RDM dimensions do not match the active-space size.");
  }

  Eigen::MatrixXd rdm_alpha = rdm_aa;
  Eigen::MatrixXd rdm_beta = rdm_bb;
  std::vector<double> rdm_aabb_flat(rdm_aabb.data(),
                                    rdm_aabb.data() + rdm_aabb.size());

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
      throw std::invalid_argument(
          std::string("ActiveSpaceQIOLocalizer setting '") + setting_name +
          "' must be finite.");
    }
  };
  require_finite("convergence_tolerance", convergence_tolerance);
  require_finite("coarse_angle_step", coarse_angle_step);
  require_finite("improvement_tolerance", improvement_tolerance);
  auto optimization = qio::detail::optimize_rotation(
      std::move(rdm_alpha), std::move(rdm_beta), std::move(rdm_aabb_flat), n,
      max_cycles, convergence_tolerance, coarse_angle_step, fine_samples,
      improvement_tolerance);

  // Apply the rotation to the active orbital columns (alpha == beta basis).
  const auto& coeffs_alpha = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  Eigen::MatrixXd selected_coeffs(coeffs_alpha.rows(),
                                  static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) {
    selected_coeffs.col(static_cast<Eigen::Index>(i)) =
        coeffs_alpha.col(static_cast<Eigen::Index>(active_indices_a[i]));
  }
  const Eigen::MatrixXd rotated_coeffs =
      selected_coeffs * optimization.rotation;

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

  // Attach both spin-resolved and spin-traced active 1-RDM payloads in the new
  // orbital basis. Unlike the natural-orbital localizer, QIO does not
  // diagonalize the 1-RDM, so these payloads are generally non-diagonal.
  const Eigen::MatrixXd rotated_one_rdm_spin_traced =
      optimization.rdm_alpha + optimization.rdm_beta;
  auto rotated_active_one_rdm = data::make_spin_diagonal_rank2_sbt_variant(
      data::ContainerTypes::MatrixVariant(std::move(optimization.rdm_alpha)),
      data::ContainerTypes::MatrixVariant(std::move(optimization.rdm_beta)),
      false);
  return algorithms::detail::new_aufbau_determinant_wavefunction(
      wavefunction, new_orbitals,
      data::ContainerTypes::MatrixVariant(rotated_one_rdm_spin_traced),
      rotated_active_one_rdm);
}

}  // namespace qdk::chemistry::algorithms::microsoft
