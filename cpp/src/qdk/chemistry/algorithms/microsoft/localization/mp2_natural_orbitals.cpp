// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "mp2_natural_orbitals.hpp"

#include <algorithm>
#include <macis/mcscf/orbital_energies.hpp>
#include <macis/util/moller_plesset.hpp>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <stdexcept>

#include "../scf/src/util/blas.h"

namespace qdk::chemistry::algorithms::microsoft {

// Bring BLAS wrapper into scope
using qdk::chemistry::scf::blas::gemm;

std::shared_ptr<data::Wavefunction> MP2NaturalOrbitalLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  auto orbitals = wavefunction->get_orbitals();
  // Get electron counts from settings
  auto [nalpha, nbeta] = wavefunction->get_total_num_electrons();

  // Check if electron counts have been set
  if (nalpha < 0 || nbeta < 0) {
    throw std::invalid_argument(
        "n_alpha_electrons and n_beta_electrons must be set in localizer "
        "settings before calling run()");
  }

  // Check that the input orbitals are canonical (have orbital energies)
  if (!orbitals->has_energies()) {
    throw std::invalid_argument(
        "Input orbitals must be canonical (have orbital energies) before "
        "localization");
  }

  // If both index vectors are empty, return original orbitals unchanged
  if (loc_indices_a.size() == 0 && loc_indices_b.size() == 0) {
    return wavefunction;
  }

  if (nalpha == 0 && nbeta == 0) {
    throw std::invalid_argument(
        "MP2 localization requires at least one occupied orbital.");
  }

  // Check for closed shell system
  if (nalpha != nbeta) {
    throw std::invalid_argument(
        "MP2NaturalOrbitalLocalizer only supports closed-shell systems (nalpha "
        "== nbeta).");
  }

  // Sanity checks
  if (not orbitals->is_restricted()) {
    throw std::invalid_argument(
        "MP2NaturalOrbitalLocalizer only supports restricted orbitals.");
  }

  // For restricted orbitals, alpha and beta indices must be identical
  if (!(loc_indices_a == loc_indices_b)) {
    throw std::invalid_argument(
        "For restricted orbitals, loc_indices_a and loc_indices_b must be "
        "identical");
  }

  // TODO (DBWY): Wrap consistency checking into a common function and reused
  // when needed. Work Item: 41816
  // Validate that indices are sorted
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }

  // the biggest loc_indice should be less than num_molecular_orbitals
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  if (!loc_indices_a.empty() &&
      loc_indices_a.back() >= num_molecular_orbitals) {
    throw std::invalid_argument(
        "loc_indices_a contains invalid orbital index >= "
        "num_molecular_orbitals");
  }

  // Separate indices into occupied and virtual orbitals
  std::vector<size_t> occ_indices, virt_indices;
  for (size_t idx : loc_indices_a) {
    if (idx < nalpha) {
      occ_indices.push_back(idx);
    } else {
      virt_indices.push_back(idx);
    }
  }

  // Extract selected orbitals for MP2 natural orbital calculation
  const auto& full_coeffs = orbitals->get_coefficients().first;
  const size_t num_orbitals = loc_indices_a.size();
  const size_t num_occupied = occ_indices.size();
  const size_t num_virtual = virt_indices.size();

  // Check that both occupied and virtual orbitals are present
  if (num_occupied == 0 || num_virtual == 0) {
    throw std::invalid_argument(
        "MP2 natural orbital calculation requires both occupied and virtual "
        "orbitals in the selected subspace");
  }

  // Extract subspace orbital coefficients
  Eigen::MatrixXd selected_coeffs(full_coeffs.rows(), num_orbitals);
  for (size_t i = 0; i < num_orbitals; ++i) {
    selected_coeffs.col(i) = full_coeffs.col(loc_indices_a[i]);
  }

  // Create selected orbitals object for Hamiltonian construction
  auto selected_orbitals = std::make_shared<data::Orbitals>(
      selected_coeffs,
      std::nullopt,  // no energies needed
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::nullopt);  // no active space indices

  // Construct Hamiltonian from selected orbitals
  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(selected_orbitals);

  // Compute MP2 Natural Orbitals for selected subspace
  const auto& one_body_int = H->get_one_body_integrals();
  const auto& two_body_int = H->get_two_body_integrals();

  Eigen::MatrixXd mp2_natural_orbitals(num_orbitals, num_orbitals);
  mp2_natural_orbitals.setZero();
  Eigen::VectorXd mp2_natural_orbital_occupations(num_orbitals);
  macis::mp2_natural_orbitals(macis::NumOrbital(num_orbitals),
                              macis::NumCanonicalOccupied(num_occupied),
                              macis::NumCanonicalVirtual(num_virtual),
                              one_body_int.data(), num_orbitals,
                              two_body_int.data(), num_orbitals,
                              mp2_natural_orbital_occupations.data(),
                              mp2_natural_orbitals.data(), num_orbitals);

  // Transform selected orbitals with MP2 natural orbital rotation
  const size_t num_basis_funcs = selected_coeffs.rows();
  Eigen::MatrixXd selected_no_coeffs =
      Eigen::MatrixXd::Zero(num_basis_funcs, num_orbitals);
  gemm("N", "N", num_basis_funcs, num_orbitals, num_orbitals, 1.0,
       selected_coeffs.data(), num_basis_funcs, mp2_natural_orbitals.data(),
       num_orbitals, 0.0, selected_no_coeffs.data(), num_basis_funcs);

  // Form final orbitals by updating only the selected orbitals
  Eigen::MatrixXd coeffs = full_coeffs;  // Start with original coefficients
  for (size_t i = 0; i < num_orbitals; ++i) {
    coeffs.col(loc_indices_a[i]) = selected_no_coeffs.col(i);
  }

  // Create new orbitals with MP2 natural orbital data
  auto new_orbitals = std::make_shared<data::Orbitals>(
      coeffs,
      std::nullopt,  // no energies for natural orbitals
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::nullopt);  // no active space indices
  return detail::new_wavefunction(wavefunction, new_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft
