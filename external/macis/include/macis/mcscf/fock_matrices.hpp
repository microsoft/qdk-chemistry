/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <stddef.h>

#include <macis/types.hpp>

namespace macis {

/** @brief Compute the inactive fock matrix.
 *
 *  Computes the inactive fock matrix from MO integrals.
 *
 *  Fi(p,q) = T(p,q) + \sum_i 2*V(p,q,i,i) - V(p,i,i,q)
 *
 *  The order of these integrals must contain
 *  the inactive orbitals as the leading blocks (i.e. the
 *  first `ninact` indices correspond to inactive orbitals).
 *
 *  @param[in]  norb   Number of total orbitals
 *  @param[in]  ninact Number of inactive orbitals
 *  @param[in]  T     The MO 1-body hamiltonian
 *  @param[in]  LDT   The leading dimension of `T`
 *  @param[in]  V     The MO 2-body hamiltonian
 *  @param[in]  LDV   The (single index) leading dimension of `V`
 *  @param[out] Fi    The inactive fock matrix.
 *  @param[in]  LDFi  The leading dimension of `Fi`
 */
void inactive_fock_matrix(NumOrbital norb, NumInactive ninact, const double* T,
                          size_t LDT, const double* V, size_t LDV, double* Fi,
                          size_t LDF);

/** @brief Compute the inactive energy.
 *
 *  Computes the inactive energy from the 1-body hamiltonian
 *  and the inactive Fock matrix.
 *
 *  E_I = \sum_i T(i,i) + Fi(i,i)
 *
 *  @param[in] ninact Number of inactive orbitals.
 *  @param[in] T      The MO 1-body hamiltonian.
 *  @param[in] LDT    The leading dimension of `T`.
 *  @param[in] Fi     The inactive Fock matrix.
 *  @param[in] LDF    The leading dimension of `Fi`.
 *
 *  @returns The inactive energy.
 */
double inactive_energy(NumInactive ninact, const double* T, size_t LDT,
                       const double* Fi, size_t LDF);

/** @brief Extact the active-active subblock of a structured matrix.
 *
 *  Extract the active-active block of a operator matrix in the MO
 *  basis. Input matrix is assumed to contain inactive orbitals
 *  as the leading block.
 *
 *  A_sub(0:na, 0:na) = A(ni:ni+na, ni:ni+na)
 *
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  A_full Full dimensional structured matrix
 *  @param[in]  LDAF   Leading dimension of `A_full`
 *  @param[out] A_sub  Extracted submatrix
 *  @param[in]  LDAS   Leading dimension of `A_sub`
 */
void active_submatrix_1body(NumActive nact, NumInactive ninact,
                            const double* A_full, size_t LDAF, double* A_sub,
                            size_t LDAS);

/** @brief Extact the all-active subblock of a structured tensor.
 *
 *  Extract the all-active block of a two-body operator tensor in the MO
 *  basis. Input tensor is assumed to contain inactive orbitals
 *  as the leading block.
 *
 *  A_sub(0:na, 0:na, 0:na, 0:na) =
 *    A(ni:ni+na, ni:ni+na, ni:ni+na, ni:ni+na)
 *
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  A_full Full dimensional structured tensor
 *  @param[in]  LDAF   Single index leading dimension of `A_full`
 *  @param[out] A_sub  Extracted submatrix
 *  @param[in]  LDAS   Single index leading dimension of `A_sub`
 */
void active_subtensor_2body(NumActive nact, NumInactive ninact,
                            const double* A_full, size_t LDAF, double* A_sub,
                            size_t LDAS);

/** @brief Compute the active-space hamiltonian.
 *
 *  Computes the active-only 1- and 2-body hamiltonian contributions for
 *  active space calculations from full dimensional hamiltonian contributions.
 *  Input tensors are assumed to be structured to have inactive orbtitals
 *  as leading indices.
 *
 *  This function computes the full dimensional inactive Fock matrix (`Fi`)
 *  as a by-product.
 *
 *  Let a_range = ni:ni+na
 *  T_active(0:na,0:na) = Fi(a_range, a_range)
 *  V_active(0:na,0:na,0:na,0:na) = V(a_range,a_range,a_range,a_range)
 *
 *  @param[in]  norb   Number of total orbitals
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  T_full The full MO 1-body hamiltonian
 *  @param[in]  LDTF   The leading dimension of `T_full`
 *  @param[in]  V_full The full MO 2-body hamiltonian
 *  @param[in]  LDVF   The (single index) leading dimension of `V_full`
 *  @param[out] Fi     The full MO inactive fock matrix.
 *  @param[in]  LDFi   The leading dimension of `Fi`
 *  @param[out] T_act  The act MO 1-body hamiltonian
 *  @param[in]  LDTF   The leading dimension of `T_act`
 *  @param[out] V_act  The act MO 2-body hamiltonian
 *  @param[in]  LDVF   The (single index) leading dimension of `V_act`
 */
void active_hamiltonian(NumOrbital norb, NumActive nact, NumInactive ninact,
                        const double* T_full, size_t LDTF, const double* V_full,
                        size_t LDVF, double* Fi, size_t LDFi, double* T_act,
                        size_t LDTA, double* V_act, size_t LDVA);

/** Compute the active Fock matrix.
 *
 *  Computes the active fock matrix from full dimensional MO
 *  integrals and the active-1RDM. Input MO integrals are assumed
 *  to be structured to contain inactive orbitals as leading indices.
 *
 *  Fa(p,q) = \sum_{wv} \gamma^A(v,w) * (V(p,q,v,w) - 0.5*V(p,v,w,q))
 *
 *  @param[in]  norb   Number of total orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  V      The MO 2-body hamiltonian
 *  @param[in]  LDV    The (single index) leading dimension of `V`
 *  @param[in]  A1RDM  Active 1-RDM
 *  @param[in]  LDD    Leading dimention of `A1RDM`
 *  @param[out] Fa     The active fock matrix.
 *  @param[in]  LDFa   The leading dimension of `Fa`
 */
void active_fock_matrix(NumOrbital norb, NumInactive ninact, NumActive nact,
                        const double* V, size_t LDV, const double* A1RDM,
                        size_t LDD, double* Fa, size_t LDF);

/** @brief Compute the auxillary Q matrix
 *
 *  Computes the auxillary Q matrix contribution to the
 *  generalized Fock matrix. Takes full dimensional MO
 *  integrals and the all-active 2RDM.
 *
 *  Q(v,p) = \sum_{wxy} \Gamma^A(v,w,x,y) * V(p,w,x,y)
 *
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  norb   Number of total orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  V      The MO 2-body hamiltonian
 *  @param[in]  LDV    The (single index) leading dimension of `V`
 *  @param[in]  A2RDM  Active 2-RDM
 *  @param[in]  LDD    Leading dimention of `A2RDM`
 *  @param[out] Q      The Q matrix.
 *  @param[in]  LDQ    The leading dimension of `Q`
 */
void aux_q_matrix(NumActive nact, NumOrbital norb, NumInactive ninact,
                  const double* V, size_t LDV, const double* A2RDM, size_t LDD,
                  double* Q, size_t LDQ);

/** @brief Compute the generalized Fock given pre-computed contributions.
 *
 *  Compute the generalized Fock matrix given all pre-computed Fock
 *  contributions: inactive Fock (`Fi`), active Fock (`Fa`),
 *  auxillary Q (`Q`). Input matrices are assumed to be structured
 *  as [inactive, active, virtual] along each index
 *
 *  Inactive - General: F(i,p) = 2 * (Fi(p,i) + Fa(p,i))
 *  Active   - General: F(v,p) = Q(v,p) + \sum_w \gamma^A(v,w) * Fi(p,w)
 *  Virtual  - General: F(a,p) = 0
 *
 *  @param[in]  norb   Number of total orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  Fi     The inactive fock matrix.
 *  @param[in]  LDFi   The leading dimension of `Fi`
 *  @param[in]  Fa     The active fock matrix.
 *  @param[in]  LDFa   The leading dimension of `Fa`
 *  @param[in]  Q      The Q matrix.
 *  @param[in]  LDQ    The leading dimension of `Q`
 *  @param[out] F      The generalized fock matrix.
 *  @param[in]  LDF    The leading dimension of `F`
 */
void generalized_fock_matrix(NumOrbital norb, NumInactive ninact,
                             NumActive nact, const double* Fi, size_t LDFi,
                             const double* Fa, size_t LDFa, const double* A1RDM,
                             size_t LDD, const double* Q, size_t LDQ, double* F,
                             size_t LDF);

/** @brief Compute the generalized Fock matrix from non-active intermediates.
 *
 *  Compute the generalied Fock matrix (see `generalized_fock_matrix`)
 *  given only the inactive Fock matrix and active RDMs. This function
 *  will compute the active Fock and auxillary Q internally and discard.
 *  The function requires full dimenional MO integrals must be structured
 *  as [inactive, active, virtual].
 *
 *  @param[in]  norb   Number of total orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  Fi     The inactive fock matrix.
 *  @param[in]  LDFi   The leading dimension of `Fi`
 *  @param[in]  A1RDM  Active 1-RDM
 *  @param[in]  LDD1   Leading dimention of `A1RDM`
 *  @param[in]  A2RDM  Active 2-RDM
 *  @param[in]  LDD2   Leading dimention of `A2RDM`
 *  @param[out] F      The generalized fock matrix.
 *  @param[in]  LDF    The leading dimension of `F`
 */
void generalized_fock_matrix_comp_mat1(NumOrbital norb, NumInactive ninact,
                                       NumActive nact, const double* Fi,
                                       size_t LDFi, const double* V_full,
                                       size_t LDV, const double* A1RDM,
                                       size_t LDD1, const double* A2RDM,
                                       size_t LDD2, double* F, size_t LDF);

/** @brief Compute the generalized Fock matrix given no intermediates.
 *
 *  Compute the generalized Fock matrix only from full MO integrals and
 *  active RDMs. All intermediates are computed internally and discarded.
 *  All full MO input tensors must be structured as [inactive, active, virtual].
 *
 *  @param[in]  norb   Number of total orbitals
 *  @param[in]  ninact Number if inactive orbitals
 *  @param[in]  nact   Number of active orbitals
 *  @param[in]  T     The MO 1-body hamiltonian
 *  @param[in]  LDT   The leading dimension of `T`
 *  @param[in]  V     The MO 2-body hamiltonian
 *  @param[in]  LDV   The (single index) leading dimension of `V`
 *  @param[in]  A1RDM  Active 1-RDM
 *  @param[in]  LDD1   Leading dimention of `A1RDM`
 *  @param[in]  A2RDM  Active 2-RDM
 *  @param[in]  LDD2   Leading dimention of `A2RDM`
 *  @param[out] F      The generalized fock matrix.
 *  @param[in]  LDF    The leading dimension of `F`
 */
void generalized_fock_matrix_comp_mat2(NumOrbital norb, NumInactive ninact,
                                       NumActive nact, const double* T,
                                       size_t LDT, const double* V, size_t LDV,
                                       const double* A1RDM, size_t LDD1,
                                       const double* A2RDM, size_t LDD2,
                                       double* F, size_t LDF);

/** @brief Compute the CI energy from the generalized Fock matrix.
 *
 *  Compute the CI energy according to
 *  E = 0.5 * \sum_{pq} \gamma(p,q) * T(p,q) + \delta(i,j) * F(p,q)
 *
 *  This function only requires the active 1RDM as the inactive/virtual
 *  RDMs are known implicitly. The full MO 1-body hamiltonian must be
 *  structured as [inactive, active, virtual]
 *
 *  @param[in] ninact Number if inactive orbitals
 *  @param[in] nact   Number of active orbitals
 *  @param[in] T     The MO 1-body hamiltonian
 *  @param[in] LDT   The leading dimension of `T`
 *  @param[in] A1RDM  Active 1-RDM
 *  @param[in] LDD    Leading dimention of `A1RDM`
 *  @param[in] F      The generalized fock matrix.
 *  @param[in] LDF    The leading dimension of `F`
 *
 *  @returns The CI energy
 */
double energy_from_generalized_fock(NumInactive ninact, NumActive nact,
                                    const double* T, size_t LDT,
                                    const double* A1RDM, size_t LDD,
                                    const double* F, size_t LDF);

}  // namespace macis
