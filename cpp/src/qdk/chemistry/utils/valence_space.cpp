// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <spdlog/spdlog.h>

#include <algorithm>
#include <qdk/chemistry/utils/valence_space.hpp>

using namespace qdk::chemistry::data;

namespace qdk::chemistry::utils {

// Noble gas elements and their atomic numbers
static const std::vector<std::pair<Element, size_t>> NOBLE_GASES = {
    {Element::He, 2},  {Element::Ne, 10}, {Element::Ar, 18},
    {Element::Kr, 36}, {Element::Xe, 54}, {Element::Rn, 86}};

// Valence orbitals by period
static const std::vector<size_t> VALENCE_ORBITALS_BY_PERIOD = {
    1,  // Period 1 (H, He): 1s orbital
    4,  // Period 2 (Li-Ne): 2s, 3*2p
    4,  // Period 3 (Na-Ar): 3s, 3*3p
    9,  // Period 4 (K-Kr): 4s, 5*3d, 3*4p
    9,  // Period 5 (Rb-Xe): 5s, 5*4d, 3*5p
    16  // Period 6 (Cs-Rn): 6s, 7*4f, 5*5d, 3*6p
};

// Helper function to calculate valence electrons for an element
size_t calculate_valence_electrons(Element element) {
  size_t atomic_number = static_cast<size_t>(element);
  if ((atomic_number == 1) || (atomic_number == 2)) {  // H and He
    return atomic_number;
  }

  // Find the nearest smaller noble gas
  size_t num_valence_electrons = 0;
  for (const auto& noble_gas : NOBLE_GASES) {
    if (noble_gas.second < atomic_number) {
      num_valence_electrons = atomic_number - noble_gas.second;
    } else {
      break;
    }
  }

  return num_valence_electrons;
}

// Helper function to calculate valence orbitals for an element
size_t calculate_valence_orbitals(Element element) {
  size_t atomic_number = static_cast<size_t>(element);

  // Determine the period
  if (atomic_number <= 2) return VALENCE_ORBITALS_BY_PERIOD[0];   // Period 1
  if (atomic_number <= 10) return VALENCE_ORBITALS_BY_PERIOD[1];  // Period 2
  if (atomic_number <= 18) return VALENCE_ORBITALS_BY_PERIOD[2];  // Period 3
  if (atomic_number <= 36) return VALENCE_ORBITALS_BY_PERIOD[3];  // Period 4
  if (atomic_number <= 54) return VALENCE_ORBITALS_BY_PERIOD[4];  // Period 5
  return VALENCE_ORBITALS_BY_PERIOD[5];                           // Period 6
}

std::pair<size_t, size_t> compute_valence_space_parameters(
    std::shared_ptr<Wavefunction> wavefunction, int charge) {
  // Extract structure from wavefunction
  std::shared_ptr<Structure> structure =
      wavefunction->get_orbitals()->get_basis_set()->get_structure();

  size_t total_valence_electrons = 0;
  size_t total_valence_orbitals = 0;

  for (size_t i = 0; i < structure->get_num_atoms(); ++i) {
    const Element element = structure->get_atom_element(i);
    if (((unsigned)element > 86)) {
      spdlog::warn(
          "valence active parameters are only implemented up through period-6 "
          "elements. "
          "Element atomic number: {}",
          static_cast<unsigned>(element));
    }
    auto valence_electron = calculate_valence_electrons(element);
    auto valence_orbital = calculate_valence_orbitals(element);
    total_valence_electrons += valence_electron;
    total_valence_orbitals += valence_orbital;
  }

  std::pair<size_t, size_t> num_electrons =
      wavefunction->get_total_num_electrons();
  size_t num_core_mos = (num_electrons.first + num_electrons.second + charge -
                         total_valence_electrons) /
                        2;
  std::shared_ptr<Orbitals> orbitals = wavefunction->get_orbitals();
  size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  // number of available valence orbitals should be the upper bound of active
  // space
  size_t num_active_valence_orbitals =
      std::min(num_molecular_orbitals - num_core_mos, total_valence_orbitals);

  size_t num_active_valence_electrons =
      total_valence_electrons -
      charge;  // total valence electrons must be smaller than
               // or equal to num_active_valence_orbitals*2

  // Return as pair: [num_active_valence_electrons, num_active_valence_orbitals]
  return {num_active_valence_electrons, num_active_valence_orbitals};
}

}  // namespace qdk::chemistry::utils
