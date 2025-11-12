/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstddef>
#include <macis/types.hpp>

namespace macis {

void compute_orbital_rotation(NumOrbital _norb, double alpha, const double* K,
                              size_t LDK, double* U, size_t LDU);

void fock_to_gradient(NumOrbital _norb, NumInactive _ninact, NumActive _nact,
                      NumVirtual _nvirt, const double* F, size_t LDF,
                      double* OG, size_t LDOG);

void orbital_rotated_generalized_fock(
    NumOrbital norb, NumInactive ninact, NumActive nact, const double* T,
    size_t LDT, const double* V, size_t LDV, const double* A1RDM, size_t LDD1,
    const double* A2RDM, size_t LDD2, const double* U, size_t LDU,
    double* T_trans, size_t LDTT, double* V_trans, size_t LDVT, double* F,
    size_t LDF);

double orbital_rotated_energy(NumOrbital norb, NumInactive ninact,
                              NumActive nact, const double* T, size_t LDT,
                              const double* V, size_t LDV, const double* A1RDM,
                              size_t LDD1, const double* A2RDM, size_t LDD2,
                              const double* U, size_t LDU, double* T_trans,
                              size_t LDTT, double* V_trans, size_t LDVT,
                              double* F, size_t LDF);

double orbital_rotated_energy(NumOrbital _norb, NumInactive ninact,
                              NumActive nact, const double* T, size_t LDT,
                              const double* V, size_t LDV, const double* A1RDM,
                              size_t LDD1, const double* A2RDM, size_t LDD2,
                              const double* U, size_t LDU);

void numerical_orbital_gradient(NumOrbital norb, NumInactive ninact,
                                NumActive nact, const double* T, size_t LDT,
                                const double* V, size_t LDV,
                                const double* A1RDM, size_t LDD1,
                                const double* A2RDM, size_t LDD2, double* OG,
                                size_t LDOG);

void numerical_orbital_hessian(NumOrbital _norb, NumInactive ninact,
                               NumActive nact, const double* T, size_t LDT,
                               const double* V, size_t LDV, const double* A1RDM,
                               size_t LDD1, const double* A2RDM, size_t LDD2,
                               double* OH, size_t LDOH);

}  // namespace macis
