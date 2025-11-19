// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>

namespace qdk::chemistry::scf {
/**
 * @brief Generate initial density matrix from atomic densities
 * @param basis_set Basis set for the calculation
 * @param mol Molecular structure
 * @param D Output density matrix
 */
void atom_guess(const BasisSet& basis_set, const Molecule& mol, double* D);
}  // namespace qdk::chemistry::scf
