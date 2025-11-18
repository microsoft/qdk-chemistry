/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>

namespace macis {

/**
 * @brief Compute canonical orbital energies
 *
 * Computes the canonical orbital energies for the MCSCF molecular orbitals.
 * These are the eigenvalues of the generalized Fock matrix and provide
 * insight into the orbital energy ordering.
 *
 * @param[in]  norb   Number of total molecular orbitals
 * @param[in]  ninact Number of inactive (doubly occupied) orbitals
 * @param[in]  T      One-electron integrals matrix
 * @param[in]  LDT    Leading dimension of the T matrix
 * @param[in]  V      Two-electron integrals tensor
 * @param[in]  LDV    Leading dimension of the V tensor
 * @param[out] eps    Array to store the computed orbital energies (size norb)
 */
void canonical_orbital_energies(NumOrbital norb, NumInactive ninact,
                                const double* T, size_t LDT, const double* V,
                                size_t LDV, double* eps);

}  // namespace macis
