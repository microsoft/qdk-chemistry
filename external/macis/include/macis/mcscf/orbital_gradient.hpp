/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstddef>
#include <macis/types.hpp>

namespace macis {

/**
 * @brief Compute orbital rotation matrix from anti-symmetric matrix
 *
 * Computes the unitary orbital rotation matrix U = exp(-alpha * K) where K is
 * an anti-symmetric matrix and alpha is a scaling factor. This is used to
 * transform molecular orbitals during MCSCF optimization.
 *
 * @param[in]  _norb  Number of molecular orbitals
 * @param[in]  alpha  Scaling factor for the rotation
 * @param[in]  K      Anti-symmetric rotation matrix (size norb x norb)
 * @param[in]  LDK    Leading dimension of the K matrix
 * @param[out] U      Computed rotation matrix (size norb x norb)
 * @param[in]  LDU    Leading dimension of the U matrix
 */
void compute_orbital_rotation(NumOrbital _norb, double alpha, const double* K,
                              size_t LDK, double* U, size_t LDU);

/**
 * @brief Convert generalized Fock matrix to orbital gradient
 *
 * Extracts the orbital gradient from the generalized Fock matrix by selecting
 * the appropriate off-diagonal blocks corresponding to orbital rotations.
 * This is used in MCSCF optimization to determine the direction of orbital
 * updates.
 *
 * @param[in]  _norb   Number of total molecular orbitals
 * @param[in]  _ninact Number of inactive orbitals
 * @param[in]  _nact   Number of active orbitals
 * @param[in]  _nvirt  Number of virtual orbitals
 * @param[in]  F       Generalized Fock matrix (size norb x norb)
 * @param[in]  LDF     Leading dimension of the F matrix
 * @param[out] OG      Orbital gradient vector (linear storage)
 * @param[in]  LDOG    Leading dimension of the OG vector
 */
void fock_to_gradient(NumOrbital _norb, NumInactive _ninact, NumActive _nact,
                      NumVirtual _num_virtual_orbitals, const double* F,
                      size_t LDF, double* OG, size_t LDOG);

/**
 * @brief Compute orbital-rotated generalized Fock matrix
 *
 * Computes the generalized Fock matrix after applying an orbital rotation.
 * This involves transforming the one- and two-electron integrals with the
 * rotation matrix U, then computing the generalized Fock matrix in the
 * rotated basis.
 *
 * @param[in]  norb    Number of total molecular orbitals
 * @param[in]  ninact  Number of inactive orbitals
 * @param[in]  nact    Number of active orbitals
 * @param[in]  T       Original one-electron integrals
 * @param[in]  LDT     Leading dimension of T matrix
 * @param[in]  V       Original two-electron integrals
 * @param[in]  LDV     Leading dimension of V tensor
 * @param[in]  A1RDM   Active space 1-RDM
 * @param[in]  LDD1    Leading dimension of A1RDM
 * @param[in]  A2RDM   Active space 2-RDM
 * @param[in]  LDD2    Leading dimension of A2RDM
 * @param[in]  U       Orbital rotation matrix
 * @param[in]  LDU     Leading dimension of U matrix
 * @param[out] T_trans Transformed one-electron integrals
 * @param[in]  LDTT    Leading dimension of T_trans
 * @param[out] V_trans Transformed two-electron integrals
 * @param[in]  LDVT    Leading dimension of V_trans
 * @param[out] F       Generalized Fock matrix in rotated basis
 * @param[in]  LDF     Leading dimension of F matrix
 */
void orbital_rotated_generalized_fock(
    NumOrbital norb, NumInactive ninact, NumActive nact, const double* T,
    size_t LDT, const double* V, size_t LDV, const double* A1RDM, size_t LDD1,
    const double* A2RDM, size_t LDD2, const double* U, size_t LDU,
    double* T_trans, size_t LDTT, double* V_trans, size_t LDVT, double* F,
    size_t LDF);

/**
 * @brief Compute energy after orbital rotation (with workspace)
 *
 * Computes the total energy after applying an orbital rotation, using
 * provided workspace arrays for transformed integrals. This version
 * allows reusing workspace memory for efficiency.
 *
 * @param[in]    norb    Number of total molecular orbitals
 * @param[in]    ninact  Number of inactive orbitals
 * @param[in]    nact    Number of active orbitals
 * @param[in]    T       Original one-electron integrals
 * @param[in]    LDT     Leading dimension of T matrix
 * @param[in]    V       Original two-electron integrals
 * @param[in]    LDV     Leading dimension of V tensor
 * @param[in]    A1RDM   Active space 1-RDM
 * @param[in]    LDD1    Leading dimension of A1RDM
 * @param[in]    A2RDM   Active space 2-RDM
 * @param[in]    LDD2    Leading dimension of A2RDM
 * @param[in]    U       Orbital rotation matrix
 * @param[in]    LDU     Leading dimension of U matrix
 * @param[inout] T_trans Workspace for transformed one-electron integrals
 * @param[in]    LDTT    Leading dimension of T_trans
 * @param[inout] V_trans Workspace for transformed two-electron integrals
 * @param[in]    LDVT    Leading dimension of V_trans
 * @param[inout] F       Workspace for generalized Fock matrix
 * @param[in]    LDF     Leading dimension of F matrix
 *
 * @return Total energy after orbital rotation
 */
double orbital_rotated_energy(NumOrbital norb, NumInactive ninact,
                              NumActive nact, const double* T, size_t LDT,
                              const double* V, size_t LDV, const double* A1RDM,
                              size_t LDD1, const double* A2RDM, size_t LDD2,
                              const double* U, size_t LDU, double* T_trans,
                              size_t LDTT, double* V_trans, size_t LDVT,
                              double* F, size_t LDF);

/**
 * @brief Compute energy after orbital rotation (without workspace)
 *
 * Computes the total energy after applying an orbital rotation.
 * This version allocates its own workspace internally and is simpler
 * to use than the workspace version.
 *
 * @param[in] _norb   Number of total molecular orbitals
 * @param[in] ninact  Number of inactive orbitals
 * @param[in] nact    Number of active orbitals
 * @param[in] T       Original one-electron integrals
 * @param[in] LDT     Leading dimension of T matrix
 * @param[in] V       Original two-electron integrals
 * @param[in] LDV     Leading dimension of V tensor
 * @param[in] A1RDM   Active space 1-RDM
 * @param[in] LDD1    Leading dimension of A1RDM
 * @param[in] A2RDM   Active space 2-RDM
 * @param[in] LDD2    Leading dimension of A2RDM
 * @param[in] U       Orbital rotation matrix
 * @param[in] LDU     Leading dimension of U matrix
 *
 * @return Total energy after orbital rotation
 */
double orbital_rotated_energy(NumOrbital _norb, NumInactive ninact,
                              NumActive nact, const double* T, size_t LDT,
                              const double* V, size_t LDV, const double* A1RDM,
                              size_t LDD1, const double* A2RDM, size_t LDD2,
                              const double* U, size_t LDU);

/**
 * @brief Compute orbital gradient using numerical differentiation
 *
 * Computes the orbital gradient numerically using finite differences.
 * This is primarily used for testing and validation of analytic gradient
 * implementations. The gradient indicates the direction of steepest energy
 * descent.
 *
 * @param[in]  norb   Number of total molecular orbitals
 * @param[in]  ninact Number of inactive orbitals
 * @param[in]  nact   Number of active orbitals
 * @param[in]  T      One-electron integrals matrix
 * @param[in]  LDT    Leading dimension of T matrix
 * @param[in]  V      Two-electron integrals tensor
 * @param[in]  LDV    Leading dimension of V tensor
 * @param[in]  A1RDM  Active space 1-RDM
 * @param[in]  LDD1   Leading dimension of A1RDM
 * @param[in]  A2RDM  Active space 2-RDM
 * @param[in]  LDD2   Leading dimension of A2RDM
 * @param[out] OG     Computed orbital gradient (linear storage)
 * @param[in]  LDOG   Leading dimension of OG vector
 */
void numerical_orbital_gradient(NumOrbital norb, NumInactive ninact,
                                NumActive nact, const double* T, size_t LDT,
                                const double* V, size_t LDV,
                                const double* A1RDM, size_t LDD1,
                                const double* A2RDM, size_t LDD2, double* OG,
                                size_t LDOG);

/**
 * @brief Compute the numerical orbital Hessian for MCSCF calculations
 *
 * Calculates the second derivatives of the energy with respect to orbital
 * rotation parameters using numerical differentiation methods.
 *
 * @param _norb Number of molecular orbitals
 * @param ninact Number of inactive (doubly occupied) orbitals
 * @param nact Number of active orbitals in the active space
 * @param T One-electron integrals matrix
 * @param LDT Leading dimension of the T matrix
 * @param V Two-electron integrals in MO basis
 * @param LDV Leading dimension of the V matrix
 * @param A1RDM Active space one-particle reduced density matrix
 * @param LDD1 Leading dimension of the A1RDM matrix
 * @param A2RDM Active space two-particle reduced density matrix
 * @param LDD2 Leading dimension of the A2RDM matrix
 * @param[out] OH Output orbital Hessian matrix
 * @param LDOH Leading dimension of the orbital Hessian matrix
 *
 * @note The orbital Hessian is used for Newton-Raphson optimization
 *       of the molecular orbitals in MCSCF procedures
 */
void numerical_orbital_hessian(NumOrbital _norb, NumInactive ninact,
                               NumActive nact, const double* T, size_t LDT,
                               const double* V, size_t LDV, const double* A1RDM,
                               size_t LDD1, const double* A2RDM, size_t LDD2,
                               double* OH, size_t LDOH);

}  // namespace macis
