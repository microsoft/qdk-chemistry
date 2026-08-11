// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <limits>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::algorithms {

namespace {

using data::Orbitals;
using data::SymmetryBlockedIndexSet;

[[noreturn]] void throw_incompatible(const std::string& detail) {
  throw std::invalid_argument("effective Hamiltonian: " + detail);
}

bool is_subset(const SymmetryBlockedIndexSet& subset,
               const SymmetryBlockedIndexSet& superset) {
  if (*subset.symmetries() != *superset.symmetries() ||
      subset.extents() != superset.extents()) {
    return false;
  }

  for (const auto& label : subset.labels()) {
    const auto selected = subset.indices(label);
    if (!superset.has(label)) {
      if (!selected.empty()) return false;
      continue;
    }

    const auto available = superset.indices(label);
    if (!std::includes(available.begin(), available.end(), selected.begin(),
                       selected.end())) {
      return false;
    }
  }
  return true;
}

void validate_orbital_basis(const Orbitals& hamiltonian_orbitals,
                            const Orbitals& wavefunction_orbitals) {
  if (&hamiltonian_orbitals == &wavefunction_orbitals) return;

  if (hamiltonian_orbitals.get_num_molecular_orbitals() !=
      wavefunction_orbitals.get_num_molecular_orbitals()) {
    throw_incompatible(
        "Hamiltonian and wavefunction must use the same MO universe");
  }
  if (hamiltonian_orbitals.get_num_atomic_orbitals() !=
      wavefunction_orbitals.get_num_atomic_orbitals()) {
    throw_incompatible(
        "Hamiltonian and wavefunction must use the same AO basis size");
  }
  if (hamiltonian_orbitals.has_basis_set() !=
      wavefunction_orbitals.has_basis_set()) {
    throw_incompatible(
        "Hamiltonian and wavefunction must both carry an AO basis set");
  }
  if (hamiltonian_orbitals.has_basis_set() &&
      hamiltonian_orbitals.get_basis_set()->content_hash() !=
          wavefunction_orbitals.get_basis_set()->content_hash()) {
    throw_incompatible(
        "Hamiltonian and wavefunction must use the same AO basis set");
  }
  if (hamiltonian_orbitals.is_restricted() !=
      wavefunction_orbitals.is_restricted()) {
    throw_incompatible(
        "Hamiltonian and wavefunction must have the same spin restriction");
  }
  if (*hamiltonian_orbitals.symmetries() !=
      *wavefunction_orbitals.symmetries()) {
    throw_incompatible(
        "Hamiltonian and wavefunction must have the same orbital symmetries");
  }

  const auto hamiltonian_coefficients = hamiltonian_orbitals.coefficients();
  const auto wavefunction_coefficients = wavefunction_orbitals.coefficients();
  for (const auto& spin : {data::axes::alpha(), data::axes::beta()}) {
    const auto& hamiltonian_block =
        hamiltonian_coefficients->block({spin, spin});
    const auto& wavefunction_block =
        wavefunction_coefficients->block({spin, spin});
    if (hamiltonian_block.rows() != wavefunction_block.rows() ||
        hamiltonian_block.cols() != wavefunction_block.cols() ||
        (hamiltonian_block - wavefunction_block).norm() >
            std::numeric_limits<double>::epsilon()) {
      throw_incompatible(
          "Hamiltonian and wavefunction must use the same orbital "
          "coefficients");
    }
  }

  if (hamiltonian_orbitals.has_energies() !=
      wavefunction_orbitals.has_energies()) {
    throw_incompatible(
        "Hamiltonian and wavefunction must have matching orbital energies");
  }
  if (hamiltonian_orbitals.has_energies()) {
    const auto hamiltonian_energies = hamiltonian_orbitals.energies();
    const auto wavefunction_energies = wavefunction_orbitals.energies();
    for (const auto& spin : {data::axes::alpha(), data::axes::beta()}) {
      const auto& hamiltonian_block = hamiltonian_energies->block({spin});
      const auto& wavefunction_block = wavefunction_energies->block({spin});
      if (hamiltonian_block.size() != wavefunction_block.size() ||
          (hamiltonian_block - wavefunction_block).norm() > 1e-12) {
        throw_incompatible(
            "Hamiltonian and wavefunction must have matching orbital "
            "energies");
      }
    }
  }
}

}  // namespace

void EffectiveHamiltonianConstructor::_validate_inputs(
    const std::shared_ptr<data::Wavefunction>& reference,
    const std::shared_ptr<data::Hamiltonian>& hamiltonian,
    const std::shared_ptr<const data::SymmetryBlockedIndexSet>& p_indices)
    const {
  if (!reference) throw_incompatible("reference must not be null");
  if (!hamiltonian) throw_incompatible("Hamiltonian must not be null");
  if (!p_indices) throw_incompatible("P-space indices must not be null");

  const auto hamiltonian_orbitals = hamiltonian->get_orbitals();
  const auto wavefunction_orbitals = reference->get_orbitals();
  validate_orbital_basis(*hamiltonian_orbitals, *wavefunction_orbitals);

  if (!is_subset(*wavefunction_orbitals->active_indices(),
                 *hamiltonian_orbitals->active_indices())) {
    throw_incompatible(
        "wavefunction active orbitals must be a subset of the Hamiltonian "
        "active orbitals");
  }
  if (!is_subset(*p_indices, *wavefunction_orbitals->active_indices())) {
    throw_incompatible(
        "P-space indices must be a subset of the wavefunction active "
        "orbitals");
  }
}

void EffectiveHamiltonianConstructorFactory::register_default_instances() {}

}  // namespace qdk::chemistry::algorithms
