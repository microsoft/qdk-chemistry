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
 *  @brief Compute a preconditioned conjugate gradient orbital step for MCSCF
 *  optimization.
 *
 *  @param[in] norb   Total number of orbitals
 *  @param[in] ninact Number of inactive orbitals
 *  @param[in] nact   Number of active orbitals
 *  @param[in] nvirt  Number of virtual orbitals
 *  @param[in] Fi     Inactive Fock matrix
 *  @param[in] LDFi   Leading dimension of Fi
 *  @param[in] Fa     Active Fock matrix
 *  @param[in] LDFa   Leading dimension of Fa
 *  @param[in] F      Full Fock matrix
 *  @param[in] LDF    Leading dimension of F
 *  @param[in] A1RDM  Active space one-particle reduced density matrix
 *  @param[in] LDD    Leading dimension of A1RDM
 *  @param[in] OG     Orbital gradient vector
 *  @param[out] K_lin Linear orbital rotation vector output
 */
void precond_cg_orbital_step(NumOrbital norb, NumInactive ninact,
                             NumActive nact, NumVirtual num_virtual_orbitals,
                             const double* Fi, size_t LDFi, const double* Fa,
                             size_t LDFa, const double* F, size_t LDF,
                             const double* A1RDM, size_t LDD, const double* OG,
                             double* K_lin);

}  // namespace macis
