// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "natural_orbitals.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <variant>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> NaturalOrbitalLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();

  if (loc_indices_a.empty() && loc_indices_b.empty()) {
    return detail::new_aufbau_determinant_wavefunction(wavefunction, orbitals);
  }

  // Natural orbitals are a single spatial orbital set.
  if (loc_indices_a != loc_indices_b) {
    throw std::invalid_argument(
        "loc_indices_a and loc_indices_b must be identical for natural orbital "
        "localization.");
  }

  // Validate selected orbital indices.
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }
  if (std::adjacent_find(loc_indices_a.begin(), loc_indices_a.end()) !=
      loc_indices_a.end()) {
    throw std::invalid_argument("loc_indices_a contains duplicate indices");
  }

  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  if (!loc_indices_a.empty() &&
      loc_indices_a.back() >= num_molecular_orbitals) {
    throw std::invalid_argument(
        "loc_indices_a contains invalid orbital index >= "
        "num_molecular_orbitals");
  }

  // Require an active space because the available 1-RDM is active-space only.
  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires an active space to be defined in the "
        "orbitals.");
  }

  const auto& [active_indices_a, active_indices_b] =
      orbitals->get_active_space_indices();
  const auto& [inactive_indices_a, inactive_indices_b] =
      orbitals->get_inactive_space_indices();

  if (active_indices_a != active_indices_b ||
      inactive_indices_a != inactive_indices_b) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires matching alpha and beta active and "
        "inactive spaces.");
  }

  if (loc_indices_a != active_indices_a || loc_indices_b != active_indices_b) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires loc_indices_a and loc_indices_b to "
        "match the orbitals' active-space indices.");
  }

  // Require AO overlap for unrestricted density projection and output orbitals.
  if (!orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires an overlap matrix to be available "
        "in the orbitals.");
  }

  const size_t num_active = active_indices_a.size();
  const auto& [coeffs_alpha, coeffs_beta] = orbitals->get_coefficients();

  // Extract active alpha coefficients as the target spatial orbital basis.
  Eigen::MatrixXd selected_coeffs(coeffs_alpha.rows(), num_active);
  for (size_t i = 0; i < num_active; ++i) {
    selected_coeffs.col(i) = coeffs_alpha.col(active_indices_a[i]);
  }

  // Build the active 1-RDM in the basis being diagonalized.
  Eigen::MatrixXd rdm_subset(num_active, num_active);
  if (orbitals->is_restricted()) {
    // Restricted inputs already provide a spin-traced active-space 1-RDM.
    if (!wavefunction->has_one_rdm_spin_traced()) {
      throw std::invalid_argument(
          "NaturalOrbitalLocalizer requires an active-space 1-RDM in the "
          "wavefunction.");
    }
    const auto& rdm_variant = wavefunction->get_active_one_rdm_spin_traced();
    const auto* rdm = std::get_if<Eigen::MatrixXd>(&rdm_variant);
    if (!rdm) {
      throw std::invalid_argument(
          "NaturalOrbitalLocalizer requires a real-valued active 1-RDM.");
    }
    if (static_cast<size_t>(rdm->rows()) != num_active ||
        static_cast<size_t>(rdm->cols()) != num_active) {
      throw std::invalid_argument(
          "1-RDM dimensions do not match the orbitals' active-space size.");
    }

    rdm_subset = *rdm;
  } else {
    // Unrestricted inputs require spin-dependent blocks in different MO bases.
    if (!wavefunction->has_one_rdm_spin_dependent()) {
      throw std::invalid_argument(
          "NaturalOrbitalLocalizer requires spin-dependent active 1-RDM "
          "blocks for unrestricted orbitals.");
    }

    auto [rdm_alpha_variant, rdm_beta_variant] =
        wavefunction->get_active_one_rdm_spin_dependent();
    const auto* rdm_alpha = std::get_if<Eigen::MatrixXd>(&rdm_alpha_variant);
    const auto* rdm_beta = std::get_if<Eigen::MatrixXd>(&rdm_beta_variant);
    if (!rdm_alpha || !rdm_beta) {
      throw std::invalid_argument(
          "NaturalOrbitalLocalizer requires real-valued active 1-RDM blocks.");
    }
    if (static_cast<size_t>(rdm_alpha->rows()) != num_active ||
        static_cast<size_t>(rdm_alpha->cols()) != num_active ||
        static_cast<size_t>(rdm_beta->rows()) != num_active ||
        static_cast<size_t>(rdm_beta->cols()) != num_active) {
      throw std::invalid_argument(
          "1-RDM dimensions do not match the orbitals' active-space size.");
    }

    // Reconstruct the active spin-summed density in the AO basis.
    Eigen::MatrixXd active_coeffs_alpha(coeffs_alpha.rows(), num_active);
    Eigen::MatrixXd active_coeffs_beta(coeffs_beta.rows(), num_active);
    for (size_t i = 0; i < num_active; ++i) {
      active_coeffs_alpha.col(i) = coeffs_alpha.col(active_indices_a[i]);
      active_coeffs_beta.col(i) = coeffs_beta.col(active_indices_b[i]);
    }

    Eigen::MatrixXd density =
        active_coeffs_alpha * (*rdm_alpha) * active_coeffs_alpha.transpose();
    density +=
        active_coeffs_beta * (*rdm_beta) * active_coeffs_beta.transpose();

    const auto& overlap = orbitals->get_overlap_matrix();
    rdm_subset = selected_coeffs.transpose() * overlap * density * overlap *
                 selected_coeffs;
  }

  // Symmetrize before diagonalization to remove numerical noise.
  rdm_subset = 0.5 * (rdm_subset + rdm_subset.transpose());

  // Diagonalize the active 1-RDM to obtain the natural orbital rotation.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(rdm_subset);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("Eigenvalue decomposition of the 1-RDM failed.");
  }

  // Eigenvalues are ascending; reverse eigenvectors for descending occupation.
  Eigen::VectorXd occupations = eigensolver.eigenvalues().reverse();
  Eigen::MatrixXd no_rotation = eigensolver.eigenvectors().rowwise().reverse();

  // Apply the natural orbital rotation only to the active columns.
  Eigen::MatrixXd no_coeffs = selected_coeffs * no_rotation;
  Eigen::MatrixXd coeffs = coeffs_alpha;
  for (size_t i = 0; i < num_active; ++i) {
    coeffs.col(active_indices_a[i]) = no_coeffs.col(i);
  }

  // Create output orbitals, preserving active/inactive metadata.
  std::shared_ptr<data::Orbitals> new_orbitals;
  if (orbitals->is_restricted()) {
    new_orbitals = std::make_shared<data::Orbitals>(
        coeffs,
        std::nullopt,  // no energies for natural orbitals
        orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
        orbitals->active_indices(), orbitals->inactive_indices());
  } else {
    const data::Orbitals::RestrictedCASIndices restricted_indices =
        std::make_tuple(std::vector<size_t>(active_indices_a.begin(),
                                            active_indices_a.end()),
                        std::vector<size_t>(inactive_indices_a.begin(),
                                            inactive_indices_a.end()));
    new_orbitals = std::make_shared<data::Orbitals>(
        coeffs,
        std::nullopt,  // no energies for natural orbitals
        orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
        restricted_indices);
  }

  Eigen::MatrixXd diagonal_one_rdm = occupations.asDiagonal();
  return detail::new_aufbau_determinant_wavefunction(
      wavefunction, new_orbitals,
      data::ContainerTypes::MatrixVariant(diagonal_one_rdm));
}

}  // namespace qdk::chemistry::algorithms::microsoft
