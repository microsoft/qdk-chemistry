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
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "ctf12_f12.hpp"
#include "ctf12_support.hpp"

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
  const ctf12::DressedHamiltonian dressed =
      ctf12::build_dressed_hamiltonian(input, /*relax_orbitals=*/true);

  const std::size_t n = dressed.n_mo;
  const std::size_t nc = dressed.n_core;
  const std::size_t nocc = dressed.n_occupied;

  // Relaxed F12-HF orbitals with the frozen core marked inactive.
  auto reference_orbitals = reference->get_orbitals();
  std::optional<Eigen::MatrixXd> ao_overlap;
  if (reference_orbitals->has_overlap_matrix())
    ao_overlap = reference_orbitals->get_overlap_matrix();

  std::vector<std::size_t> active_indices, inactive_indices;
  for (std::size_t i = 0; i < nc; ++i) inactive_indices.push_back(i);
  for (std::size_t i = nc; i < n; ++i) active_indices.push_back(i);

  auto orbitals = std::make_shared<data::Orbitals>(
      dressed.mo_coefficients, std::make_optional(dressed.orbital_energies),
      ao_overlap, reference_orbitals->get_basis_set(),
      std::make_optional(std::make_tuple(active_indices, inactive_indices)));

  // Closed-shell Hartree-Fock determinant over the active space.
  std::string config_str(n - nc, '0');
  for (std::size_t i = 0; i < nocc - nc; ++i) config_str[i] = '2';
  auto determinant = data::Configuration::from_spin_half_string(config_str);

  auto container = std::make_unique<data::StateVectorContainer>(
      determinant, orbitals, "electrons");
  auto relaxed_reference =
      std::make_shared<data::Wavefunction>(std::move(container));

  const double total_energy =
      dressed.e_f12hf + structure->calculate_nuclear_repulsion_energy();
  return {total_energy, relaxed_reference};
}

}  // namespace qdk::chemistry::algorithms::microsoft
