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

using NumCanonicalOccupied =
    NamedType<size_t, struct num_occupied_orbitals_canon_type>;
using NumCanonicalVirtual = NamedType<size_t, struct nvir_canon_type>;

/**
 *  @brief Form MP2 T2 Amplitudes
 *
 *  @param[in] num_occupied_orbitals Number of occupied orbitals
 *  @param[in] num_virtual_orbitals Number of virtual orbitals
 *  @param[in] V    The two-body Hamiltonian
 *  @param[in] LDV  The leading dimension of `V`
 *  @param[in] eps  Orbital eigenenergies
 *  @param[out] T2  MP2 T2 amplitudes
 *  @param[in] shift Energy shift to apply in the MP2 denominator (default 0.0)
 */
void mp2_t2(NumCanonicalOccupied num_occupied_orbitals,
            NumCanonicalVirtual num_virtual_orbitals, const double* V,
            size_t LDV, const double* eps, double* T2, double shift = 0.0);

/**
 *  @brief Form the MP2 1-RDM
 *
 *  @param[in] norb Number of orbitals
 *  @param[in] num_occupied_orbitals Number of occupied orbitals
 *  @param[in] num_virtual_orbitals Number of virtual orbitals
 *  @param[in] T    The one-body Hamiltonian
 *  @param[in] LDT  The leading dimension of `T`
 *  @param[in] V    The two-body Hamiltonian
 *  @param[in] LDV  The leading dimension of `V`
 *  @param[out] ORDM The MP2 1-RDM
 *  @param[in]  LDD  The leading dimension of `ORDM`
 *  @param[in] shift Energy shift to apply in the MP2 denominator (default 0.0)
 */
void mp2_1rdm(NumOrbital norb, NumCanonicalOccupied num_occupied_orbitals,
              NumCanonicalVirtual num_virtual_orbitals, const double* T,
              size_t LDT, const double* V, size_t LDV, double* ORDM, size_t LDD,
              double shift = 0.0);

/**
 *  @brief Form the MP2 Natural Orbitals
 *
 *  @param[in] norb Number of orbitals
 *  @param[in] num_occupied_orbitals Number of occupied orbitals
 *  @param[in] num_virtual_orbitals Number of virtual orbitals
 *  @param[in] T    The one-body Hamiltonian
 *  @param[in] LDT  The leading dimension of `T`
 *  @param[in] V    The two-body Hamiltonian
 *  @param[in] LDV  The leading dimension of `V`
 *  @param[out[ ON   The MP2 natural orbital occupataion numbers
 *  @param[out] NO_C The MP2 natural orbital rotation matrix
 *  @param[in]  LDC  The leading dimension of `NO_C`
 *  @param[in] shift Energy shift to apply in the MP2 denominator (default 0.0)
 */
void mp2_natural_orbitals(NumOrbital norb,
                          NumCanonicalOccupied num_occupied_orbitals,
                          NumCanonicalVirtual num_virtual_orbitals,
                          const double* T, size_t LDT, const double* V,
                          size_t LDV, double* ON, double* NO_C, size_t LDC,
                          double shift = 0.0);

}  // namespace macis
