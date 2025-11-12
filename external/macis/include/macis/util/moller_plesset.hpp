/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>

namespace macis {

using NumCanonicalOccupied = NamedType<size_t, struct nocc_canon_type>;
using NumCanonicalVirtual = NamedType<size_t, struct nvir_canon_type>;

/**
 *  @brief Form MP2 T2 Amplitudes
 *
 *  @param[in] nocc Number of occupied orbitals
 *  @param[in] nvir Number of virtual orbitals
 *  @param[in] V    The two-body Hamiltonian
 *  @param[in] LDV  The leading dimension of `V`
 *  @param[in] eps  Orbital eigenenergies
 *  @param[out] T2  MP2 T2 amplitudes
 */
void mp2_t2(NumCanonicalOccupied nocc, NumCanonicalVirtual nvir,
            const double* V, size_t LDV, const double* eps, double* T2);

/**
 *  @brief Form the MP2 1-RDM
 *
 *  @param[in] norb Number of orbitals
 *  @param[in] nocc Number of occupied orbitals
 *  @param[in] nvir Number of virtual orbitals
 *  @param[in] T    The one-body Hamiltonian
 *  @param[in] LDT  The leading dimension of `T`
 *  @param[in] V    The two-body Hamiltonian
 *  @param[in] LDV  The leading dimension of `V`
 *  @param[out] ORDM The MP2 1-RDM
 *  @param[in]  LDD  The leading dimension of `ORDM`
 */
void mp2_1rdm(NumOrbital norb, NumCanonicalOccupied nocc,
              NumCanonicalVirtual nvir, const double* T, size_t LDT,
              const double* V, size_t LDV, double* ORDM, size_t LDD);

/**
 *  @brief Form the MP2 Natural Orbitals
 *
 *  @param[in] norb Number of orbitals
 *  @param[in] nocc Number of occupied orbitals
 *  @param[in] nvir Number of virtual orbitals
 *  @param[in] T    The one-body Hamiltonian
 *  @param[in] LDT  The leading dimension of `T`
 *  @param[in] V    The two-body Hamiltonian
 *  @param[in] LDV  The leading dimension of `V`
 *  @param[out[ ON   The MP2 natural orbital occupataion numbers
 *  @param[out] NO_C The MP2 natural orbital rotation matrix
 *  @param[in]  LDC  The leading dimension of `NO_C`
 */
void mp2_natural_orbitals(NumOrbital norb, NumCanonicalOccupied nocc,
                          NumCanonicalVirtual nvir, const double* T, size_t LDT,
                          const double* V, size_t LDV, double* ON, double* NO_C,
                          size_t LDC);

}  // namespace macis
