// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qdk::chemistry::scf {
/**
 * @brief Molecular structure data
 *
 * Contains atomic coordinates, charges, and molecular properties
 * for quantum chemistry calculations. All units are atomic
 */
struct Molecule {
  uint64_t n_atoms;                      ///< Number of atoms in the molecule
  uint64_t total_nuclear_charge;         ///< Sum of all atomic numbers
  uint64_t n_electrons;                  ///< Total number of electrons
  std::vector<uint64_t> atomic_nums;     ///< Atomic numbers for each atom
  std::vector<uint64_t> atomic_charges;  ///< Nuclear charges for each atom

  std::vector<std::array<double, 3>>
      coords;  ///< Cartesian coordinates in Bohr for each atom

  int64_t charge = 0;        ///< Molecular charge
  int64_t multiplicity = 1;  ///< Spin multiplicity (2S+1)
};
}  // namespace qdk::chemistry::scf
