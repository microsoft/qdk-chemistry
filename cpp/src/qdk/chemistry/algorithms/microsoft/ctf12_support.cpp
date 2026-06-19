// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ctf12_support.hpp"

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/util/cabs.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <algorithm>
#include <cctype>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft::ctf12 {

F12HartreeFockInput f12_input_from_wavefunction(
    const data::Wavefunction& reference, double gamma,
    const std::string& cabs_basis, std::size_t frozen_core) {
  QDK_LOG_TRACE_ENTERING();

  auto orbitals = reference.get_orbitals();
  if (!orbitals) {
    throw std::invalid_argument(
        "CT-F12: reference wavefunction has no orbitals");
  }
  if (orbitals->is_unrestricted()) {
    throw std::invalid_argument(
        "CT-F12 requires a closed-shell (restricted) reference");
  }
  auto basis_set = orbitals->get_basis_set();
  if (!basis_set) {
    throw std::invalid_argument(
        "CT-F12: reference orbitals have no associated basis set");
  }

  const auto [n_alpha, n_beta] = reference.get_total_num_electrons();
  if (n_alpha != n_beta) {
    throw std::invalid_argument(
        "CT-F12 requires a closed-shell reference (equal alpha and beta "
        "electrons)");
  }
  if (frozen_core >= n_alpha) {
    throw std::invalid_argument(
        "CT-F12: number of frozen core orbitals must be smaller than the "
        "number of occupied orbitals");
  }

  // Derive the CABS auxiliary basis name when none is supplied.
  std::string cabs_name = cabs_basis;
  if (cabs_name.empty()) {
    cabs_name = basis_set->get_name();
    std::transform(
        cabs_name.begin(), cabs_name.end(), cabs_name.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    cabs_name += "-optri";
  }

  auto obs_scf = utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  auto mol = obs_scf->mol;
  auto obs_libint = scf::libint2_util::convert_to_libint_basisset(*obs_scf);
  auto aux_scf = scf::BasisSet::from_database_json(mol, cabs_name,
                                                   scf::BasisMode::PSI4, true);
  auto aux_libint = scf::libint2_util::convert_to_libint_basisset(*aux_scf);
  auto cabs = scf::cabs::build_cabs(obs_libint, aux_libint);

  F12HartreeFockInput input;
  input.obs = obs_libint;
  input.mo_coefficients = orbitals->get_coefficients_alpha();
  input.orbital_energies = orbitals->get_energies_alpha();
  input.n_occupied = static_cast<std::size_t>(n_alpha);
  input.n_core = frozen_core;
  input.cabs_ri_basis = cabs.ri_basis;
  input.cabs_coefficients = cabs.cabs_coeff;
  input.gamma = gamma;
  for (std::size_t a = 0; a < mol->n_atoms; ++a) {
    input.nuclei.emplace_back(static_cast<double>(mol->atomic_charges[a]),
                              mol->coords[a]);
  }
  return input;
}

}  // namespace qdk::chemistry::algorithms::microsoft::ctf12
