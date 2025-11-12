// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace qdk::chemistry::scf {
/**
 * @brief Initialize density matrix using SOAD method
 *
 * Generates initial SCF guess by superimposing atomic densities.
 *
 * @param density_matrix Output density matrix
 * @param n_atomic_orbitals Number of basis functions
 * @param atomic_nums Atomic numbers for each atom
 * @param n_atoms Number of atoms
 */
void soad_initialize_density_matrix(double *density_matrix,
                                    uint64_t n_atomic_orbitals,
                                    const uint64_t *atomic_nums,
                                    uint64_t n_atoms);
}  // namespace qdk::chemistry::scf
