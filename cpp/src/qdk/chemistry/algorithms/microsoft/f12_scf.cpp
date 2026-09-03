// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "f12_scf.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <string>
#include <vector>

#include "ctf12_f12.hpp"

namespace qdk::chemistry::algorithms::microsoft {

std::pair<double, std::shared_ptr<data::Wavefunction>>
CtF12ScfSolver::_run_impl(std::shared_ptr<data::Structure> structure,
                          int charge, int multiplicity,
                          BasisOrGuessType basis_or_guess) const {
  QDK_LOG_TRACE_ENTERING();

  // Canonical Hartree-Fock reference from the configured SCF sub-step.
  auto canonical_scf = _create_nested<ScfSolverFactory>("canonical_scf");
  auto [e_hf, reference] =
      canonical_scf->run(structure, charge, multiplicity, basis_or_guess);

  const double gamma = _settings->get<double>("gamma");
  const std::string cabs_basis = _settings->get<std::string>("cabs_basis");
  const auto frozen_core =
      static_cast<std::size_t>(_settings->get<std::int64_t>("frozen_core"));

  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, cabs_basis, frozen_core);
  const ctf12::F12HartreeFockResult f12 = ctf12::run_f12_hf(input);

  const std::size_t n = f12.n_mo;
  const std::size_t nc = f12.n_core;
  const std::size_t nocc = f12.n_occupied;

  // Relaxed F12-HF orbitals with the frozen core marked inactive.
  auto reference_orbitals = reference->get_orbitals();
  std::optional<Eigen::MatrixXd> ao_overlap;
  if (reference_orbitals->has_overlap_matrix())
    ao_overlap = reference_orbitals->get_overlap_matrix();

  std::vector<std::size_t> active_indices, inactive_indices;
  for (std::size_t i = 0; i < nc; ++i) inactive_indices.push_back(i);
  for (std::size_t i = nc; i < n; ++i) active_indices.push_back(i);

  const Eigen::MatrixXd relaxed_coefficients =
      input.mo_coefficients * f12.relaxation;
  auto orbitals = std::make_shared<data::Orbitals>(
      relaxed_coefficients, std::make_optional(f12.relaxed_energies),
      ao_overlap, reference_orbitals->get_basis_set(),
      ctf12::restricted_index_set(n, active_indices),
      ctf12::restricted_index_set(n, inactive_indices));

  // Closed-shell Hartree-Fock determinant over the active space.
  std::string config_str(n - nc, '0');
  for (std::size_t i = 0; i < nocc - nc; ++i) config_str[i] = '2';
  auto determinant = data::Configuration::from_spin_half_string(config_str);

  auto container = std::make_unique<data::StateVectorContainer>(
      determinant, orbitals, "electrons");
  auto relaxed_reference =
      std::make_shared<data::Wavefunction>(std::move(container));

  const double total_energy = e_hf + (f12.e_f12hf - f12.e_hf);
  return {total_energy, relaxed_reference};
}

}  // namespace qdk::chemistry::algorithms::microsoft
