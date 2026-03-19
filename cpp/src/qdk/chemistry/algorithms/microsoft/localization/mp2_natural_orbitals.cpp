// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "mp2_natural_orbitals.hpp"

#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>

#include <algorithm>
#include <blas.hh>
#include <macis/util/moller_plesset.hpp>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "../utils.hpp"

namespace qcs = qdk::chemistry::scf;

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> MP2NaturalOrbitalLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
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

  // Build separate C_occ and C_virt coefficient matrices (row-major for MOERI)
  const size_t num_atomic_orbitals = full_coeffs.rows();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      C_occ_rm(num_atomic_orbitals, num_occupied);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      C_virt_rm(num_atomic_orbitals, num_virtual);
  for (size_t i = 0; i < num_occupied; ++i) {
    C_occ_rm.col(i) = selected_coeffs.col(i);
  }
  for (size_t i = 0; i < num_virtual; ++i) {
    C_virt_rm.col(i) = selected_coeffs.col(num_occupied + i);
  }

  // Extract canonical orbital energies for selected orbitals
  const auto& [full_energies_a, full_energies_b] = orbitals->get_energies();
  Eigen::VectorXd eps_occ(num_occupied), eps_virt(num_virtual);
  for (size_t i = 0; i < num_occupied; ++i) {
    eps_occ[i] = full_energies_a[occ_indices[i]];
  }
  for (size_t i = 0; i < num_virtual; ++i) {
    eps_virt[i] = full_energies_a[virt_indices[i]];
  }

  // Create ERI engine directly (bypass Hamiltonian constructor)
  auto basis_set = orbitals->get_basis_set();
  auto internal_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  auto scf_config = std::make_unique<qcs::SCFConfig>();
  scf_config->mpi = qcs::mpi_default_input();
  scf_config->require_gradient = false;
  scf_config->basis = internal_basis_set->name;
  scf_config->cartesian = !internal_basis_set->pure;
  scf_config->scf_orbital_type = qcs::SCFOrbitalType::Restricted;

  std::string method_name = _settings->get<std::string>("eri_method");
  if (!method_name.compare("incore")) {
    scf_config->eri.method = qcs::ERIMethod::Incore;
    scf_config->k_eri.method = qcs::ERIMethod::Incore;
  } else if (!method_name.compare("direct")) {
    scf_config->eri.method = qcs::ERIMethod::Libint2Direct;
    scf_config->k_eri.method = qcs::ERIMethod::Libint2Direct;
  } else {
    throw std::runtime_error("Unsupported ERI method '" + method_name + "'");
  }

  auto eri = qcs::ERIMultiplexer::create(*internal_basis_set, *scf_config, 0.0);

  // Compute only (ia|jb) integrals via generalized MOERI transform
  qcs::MOERI moeri(eri);
  std::vector<double> V_iajb(num_occupied * num_virtual * num_occupied *
                             num_virtual);
  moeri.compute(num_atomic_orbitals, num_occupied, C_occ_rm.data(), num_virtual,
                C_virt_rm.data(), num_occupied, C_occ_rm.data(), num_virtual,
                C_virt_rm.data(), V_iajb.data());

  // Compute MP2 Natural Orbitals directly from (ia|jb) + orbital energies
  Eigen::MatrixXd mp2_natural_orbitals(num_orbitals, num_orbitals);
  mp2_natural_orbitals.setZero();
  Eigen::VectorXd mp2_natural_orbital_occupations(num_orbitals);
  macis::mp2_natural_orbitals_ov(macis::NumOrbital(num_orbitals),
                                 macis::NumCanonicalOccupied(num_occupied),
                                 macis::NumCanonicalVirtual(num_virtual),
                                 eps_occ.data(), eps_virt.data(), V_iajb.data(),
                                 mp2_natural_orbital_occupations.data(),
                                 mp2_natural_orbitals.data(), num_orbitals);

  // Transform selected orbitals with MP2 natural orbital rotation
  Eigen::MatrixXd selected_no_coeffs =
      Eigen::MatrixXd::Zero(num_atomic_orbitals, num_orbitals);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_atomic_orbitals, num_orbitals, num_orbitals, 1.0,
             selected_coeffs.data(), num_atomic_orbitals,
             mp2_natural_orbitals.data(), num_orbitals, 0.0,
             selected_no_coeffs.data(), num_atomic_orbitals);

  // Form final orbitals by updating only the selected orbitals
  Eigen::MatrixXd coeffs = full_coeffs;  // Start with original coefficients
  for (size_t i = 0; i < num_orbitals; ++i) {
    coeffs.col(loc_indices_a[i]) = selected_no_coeffs.col(i);
  }

  // Preserve active space indices from input orbitals if they exist
  // MP2 natural orbitals only supports restricted orbitals (alpha == beta)
  std::optional<data::Orbitals::RestrictedCASIndices> restricted_indices;
  if (orbitals->has_active_space()) {
    const auto& active = orbitals->get_active_space_indices().first;
    const auto& inactive = orbitals->get_inactive_space_indices().first;
    restricted_indices =
        std::make_tuple(std::vector<size_t>(active.begin(), active.end()),
                        std::vector<size_t>(inactive.begin(), inactive.end()));
  }

  // Create new orbitals with MP2 natural orbital data
  auto new_orbitals = std::make_shared<data::Orbitals>(
      coeffs,
      std::nullopt,  // no energies for natural orbitals
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      restricted_indices);  // preserve active space indices from input
  return detail::new_wavefunction(wavefunction, new_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft
