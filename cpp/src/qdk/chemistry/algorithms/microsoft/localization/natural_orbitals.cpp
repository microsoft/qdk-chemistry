// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "natural_orbitals.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <optional>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
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

  const auto active_index_set = orbitals->active_indices();
  const auto active_indices_a =
      data::spin_channel_indices(active_index_set, data::axes::alpha());
  const auto active_indices_b =
      data::spin_channel_indices(active_index_set, data::axes::beta());
  const auto inactive_index_set = orbitals->inactive_indices();
  const auto inactive_indices_a =
      data::spin_channel_indices(inactive_index_set, data::axes::alpha());
  const auto inactive_indices_b =
      data::spin_channel_indices(inactive_index_set, data::axes::beta());

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
  const auto& coeffs_alpha = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  const auto& coeffs_beta =
      orbitals->coefficients()->block({data::axes::beta(), data::axes::beta()});

  // Extract active alpha coefficients as the target spatial orbital basis.
  Eigen::MatrixXd selected_coeffs(coeffs_alpha.rows(), num_active);
  for (size_t i = 0; i < num_active; ++i) {
    selected_coeffs.col(i) = coeffs_alpha.col(active_indices_a[i]);
  }

  if (!wavefunction->has_one_rdm_spin_dependent()) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires spin-dependent active 1-RDM "
        "blocks.");
  }
  const auto* input_one_rdm =
      std::get_if<data::SymmetryBlockedTensor<2, double>>(
          &wavefunction->active_one_rdm());
  if (!input_one_rdm) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires real-valued active 1-RDM blocks.");
  }
  Eigen::MatrixXd rdm_alpha =
      input_one_rdm->block({data::axes::alpha(), data::axes::alpha()});
  Eigen::MatrixXd rdm_beta =
      input_one_rdm->block({data::axes::beta(), data::axes::beta()});
  if (static_cast<size_t>(rdm_alpha.rows()) != num_active ||
      static_cast<size_t>(rdm_alpha.cols()) != num_active ||
      static_cast<size_t>(rdm_beta.rows()) != num_active ||
      static_cast<size_t>(rdm_beta.cols()) != num_active) {
    throw std::invalid_argument(
        "1-RDM dimensions do not match the orbitals' active-space size.");
  }

  if (!orbitals->is_restricted()) {
    // Rebase both spin blocks into the alpha orbital basis being diagonalized.
    // Reconstruct the active alpha and beta densities in the AO basis.
    Eigen::MatrixXd active_coeffs_alpha(coeffs_alpha.rows(), num_active);
    Eigen::MatrixXd active_coeffs_beta(coeffs_beta.rows(), num_active);
    for (size_t i = 0; i < num_active; ++i) {
      active_coeffs_alpha.col(i) = coeffs_alpha.col(active_indices_a[i]);
      active_coeffs_beta.col(i) = coeffs_beta.col(active_indices_b[i]);
    }

    Eigen::MatrixXd density_alpha =
        active_coeffs_alpha * rdm_alpha * active_coeffs_alpha.transpose();
    Eigen::MatrixXd density_beta =
        active_coeffs_beta * rdm_beta * active_coeffs_beta.transpose();

    const auto& overlap = orbitals->get_overlap_matrix();
    rdm_alpha = selected_coeffs.transpose() * overlap * density_alpha *
                overlap * selected_coeffs;
    rdm_beta = selected_coeffs.transpose() * overlap * density_beta * overlap *
               selected_coeffs;
  }

  // Symmetrize before diagonalization to remove numerical noise.
  Eigen::MatrixXd rdm = rdm_alpha + rdm_beta;
  rdm = 0.5 * (rdm + rdm.transpose());

  // Diagonalize the active 1-RDM to obtain the natural orbital rotation.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(rdm);
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
  Eigen::MatrixXd transformed_alpha =
      no_rotation.transpose() * rdm_alpha * no_rotation;
  Eigen::MatrixXd transformed_beta =
      no_rotation.transpose() * rdm_beta * no_rotation;
  transformed_alpha = 0.5 * (transformed_alpha + transformed_alpha.transpose());
  transformed_beta = 0.5 * (transformed_beta + transformed_beta.transpose());
  auto active_one_rdm = data::make_spin_diagonal_rank2_sbt_variant(
      data::ContainerTypes::MatrixVariant(transformed_alpha),
      data::ContainerTypes::MatrixVariant(transformed_beta), false);
  return detail::new_aufbau_determinant_wavefunction(
      wavefunction, new_orbitals,
      data::ContainerTypes::MatrixVariant(diagonal_one_rdm), active_one_rdm);
}

}  // namespace qdk::chemistry::algorithms::microsoft
