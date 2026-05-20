// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "natural_orbitals.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> NaturalOrbitalLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();

  // If both index vectors are empty, return original wavefunction unchanged
  if (loc_indices_a.empty() && loc_indices_b.empty()) {
    return wavefunction;
  }

  // Natural orbitals are a single set — alpha and beta indices must be
  // identical regardless of whether the input orbitals are restricted or not.
  if (!(loc_indices_a == loc_indices_b)) {
    throw std::invalid_argument(
        "loc_indices_a and loc_indices_b must be identical for natural orbital "
        "localization.");
  }

  // Validate that indices are sorted
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }

  // The biggest loc_index should be less than num_molecular_orbitals
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  if (!loc_indices_a.empty() &&
      loc_indices_a.back() >= num_molecular_orbitals) {
    throw std::invalid_argument(
        "loc_indices_a contains invalid orbital index >= "
        "num_molecular_orbitals");
  }

  // Require a spin-traced 1-RDM
  if (!wavefunction->has_one_rdm_spin_traced()) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires a spin-traced 1-RDM in the "
        "wavefunction.");
  }

  // Require an active space
  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires an active space to be defined in the "
        "orbitals.");
  }

  // For unrestricted orbitals we currently rely on the spin-traced 1-RDM being
  // expressed in the alpha MO basis, which is the convention enforced by
  // SlaterDeterminantContainer::get_active_one_rdm_spin_traced().
  if (!orbitals->is_restricted() &&
      !wavefunction->has_container_type<data::SlaterDeterminantContainer>()) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer with unrestricted orbitals is only supported "
        "for SlaterDeterminantContainer wavefunctions.");
  }

  // Get the spin-traced 1-RDM (active space only)
  const auto& rdm_variant = wavefunction->get_active_one_rdm_spin_traced();
  const auto* rdm_ptr = std::get_if<Eigen::MatrixXd>(&rdm_variant);
  if (!rdm_ptr) {
    throw std::invalid_argument(
        "NaturalOrbitalLocalizer requires a real-valued spin-traced 1-RDM.");
  }
  const Eigen::MatrixXd& rdm = *rdm_ptr;

  const size_t num_active = loc_indices_a.size();
  if (static_cast<size_t>(rdm.rows()) != num_active ||
      static_cast<size_t>(rdm.cols()) != num_active) {
    throw std::invalid_argument(
        "1-RDM dimensions do not match the number of selected active "
        "orbitals.");
  }

  // Diagonalize the spin-traced 1-RDM to get natural orbitals
  // The eigenvectors are the natural orbital rotation matrix,
  // and the eigenvalues are the occupation numbers.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(rdm);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("Eigenvalue decomposition of the 1-RDM failed.");
  }

  // Eigenvalues are in ascending order; reverse for descending occupation
  Eigen::VectorXd occupations = eigensolver.eigenvalues().reverse();
  Eigen::MatrixXd no_rotation = eigensolver.eigenvectors().rowwise().reverse();

  // Transform the active orbital coefficients.
  // For an unrestricted Slater determinant the 1-RDM is expressed in the
  // alpha MO basis, so we use the alpha coefficients as the reference.
  const auto& full_coeffs = orbitals->get_coefficients().first;
  const size_t num_atomic_orbitals = full_coeffs.rows();

  // Extract active subspace coefficients
  Eigen::MatrixXd active_coeffs(num_atomic_orbitals, num_active);
  for (size_t i = 0; i < num_active; ++i) {
    active_coeffs.col(i) = full_coeffs.col(loc_indices_a[i]);
  }

  // Rotate active coefficients to natural orbital basis
  Eigen::MatrixXd no_coeffs = active_coeffs * no_rotation;

  // Construct full coefficient matrix with only active orbitals replaced
  Eigen::MatrixXd coeffs = full_coeffs;
  for (size_t i = 0; i < num_active; ++i) {
    coeffs.col(loc_indices_a[i]) = no_coeffs.col(i);
  }

  // Preserve active space indices from input orbitals
  const auto& [active_indices_a, active_indices_b] =
      orbitals->get_active_space_indices();
  std::optional<data::Orbitals::RestrictedCASIndices> restricted_indices;
  const auto& inactive = orbitals->get_inactive_space_indices().first;
  restricted_indices = std::make_tuple(
      std::vector<size_t>(active_indices_a.begin(), active_indices_a.end()),
      std::vector<size_t>(inactive.begin(), inactive.end()));

  // Create new restricted orbitals with the natural orbital coefficients.
  // Natural orbitals are always a single (restricted) set regardless of
  // whether the input was unrestricted.
  auto new_orbitals = std::make_shared<data::Orbitals>(
      coeffs,
      std::nullopt,  // no energies for natural orbitals
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      restricted_indices);

  return detail::new_wavefunction(wavefunction, new_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft
