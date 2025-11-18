/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/mcscf/fock_matrices.hpp>
#include <macis/mcscf/orbital_rotation_utilities.hpp>
#include <macis/types.hpp>

namespace macis {

/**
 * @brief Compute the approximate diagonal Hessian for orbital rotations
 *
 * Computes the approximate diagonal elements of the orbital-orbital Hessian
 * matrix for use in optimization algorithms. The Hessian is split into
 * virtual-inactive, virtual-active, and active-inactive orbital rotation
 * blocks.
 *
 * @param _ni Number of inactive orbitals
 * @param _na Number of active orbitals
 * @param _nv Number of virtual orbitals
 * @param Fi Inactive Fock matrix (ni x ni)
 * @param LDFi Leading dimension of Fi matrix
 * @param Fa Active Fock matrix (na x na)
 * @param LDFa Leading dimension of Fa matrix
 * @param A1RDM Active space one-particle reduced density matrix
 * @param LDD Leading dimension of A1RDM matrix
 * @param F Generalized Fock matrix
 * @param LDF Leading dimension of F matrix
 * @param H_vi Output array for virtual-inactive Hessian diagonal elements
 * @param H_va Output array for virtual-active Hessian diagonal elements
 * @param H_ai Output array for active-inactive Hessian diagonal elements
 */
void approx_diag_hessian(NumInactive _ni, NumActive _na, NumVirtual _nv,
                         const double* Fi, size_t LDFi, const double* Fa,
                         size_t LDFa, const double* A1RDM, size_t LDD,
                         const double* F, size_t LDF, double* H_vi,
                         double* H_va, double* H_ai);

/**
 * @brief Compute the approximate diagonal Hessian with linearized output
 *
 * Convenience overload that computes the approximate diagonal Hessian and
 * stores the result in a single linearized array. The output array is
 * automatically split into the appropriate orbital rotation blocks.
 *
 * @param ni Number of inactive orbitals
 * @param na Number of active orbitals
 * @param nv Number of virtual orbitals
 * @param Fi Inactive Fock matrix (ni x ni)
 * @param LDFi Leading dimension of Fi matrix
 * @param Fa Active Fock matrix (na x na)
 * @param LDFa Leading dimension of Fa matrix
 * @param A1RDM Active space one-particle reduced density matrix
 * @param LDD Leading dimension of A1RDM matrix
 * @param F Generalized Fock matrix
 * @param LDF Leading dimension of F matrix
 * @param H_lin Output linearized array for all Hessian diagonal elements
 */
inline void approx_diag_hessian(NumInactive ni, NumActive na, NumVirtual nv,
                                const double* Fi, size_t LDFi, const double* Fa,
                                size_t LDFa, const double* A1RDM, size_t LDD,
                                const double* F, size_t LDF, double* H_lin) {
  auto [H_vi, H_va, H_ai] = split_linear_orb_rot(ni, na, nv, H_lin);
  approx_diag_hessian(ni, na, nv, Fi, LDFi, Fa, LDFa, A1RDM, LDD, F, LDF, H_vi,
                      H_va, H_ai);
}

/**
 * @brief Compute the approximate diagonal Hessian from basic integrals
 *
 * High-level interface that computes the approximate diagonal Hessian starting
 * from the one- and two-electron integrals. This function constructs the
 * necessary Fock matrices and auxiliary matrices internally before calling the
 * lower-level Hessian computation routine.
 *
 * @tparam Args Variadic template arguments passed to the underlying Hessian
 * routine
 * @param norb Total number of orbitals
 * @param ninact Number of inactive orbitals
 * @param nact Number of active orbitals
 * @param nvirt Number of virtual orbitals
 * @param T One-electron kinetic energy integral matrix
 * @param LDT Leading dimension of T matrix
 * @param V Two-electron repulsion integral matrix
 * @param LDV Leading dimension of V matrix
 * @param A1RDM Active space one-particle reduced density matrix
 * @param LDD1 Leading dimension of A1RDM matrix
 * @param A2RDM Active space two-particle reduced density matrix
 * @param LDD2 Leading dimension of A2RDM matrix
 * @param args Additional arguments forwarded to the underlying Hessian routine
 */
template <typename... Args>
void approx_diag_hessian(NumOrbital norb, NumInactive ninact, NumActive nact,
                         NumVirtual num_virtual_orbitals, const double* T,
                         size_t LDT, const double* V, size_t LDV,
                         const double* A1RDM, size_t LDD1, const double* A2RDM,
                         size_t LDD2, Args&&... args) {
  const size_t no = norb.get();
  const size_t ni = ninact.get();
  const size_t na = nact.get();
  const size_t nv = num_virtual_orbitals.get();

  // Compute inactive Fock
  std::vector<double> Fi(no * no);
  inactive_fock_matrix(norb, ninact, T, LDT, V, LDV, Fi.data(), no);

  // Compute active fock
  std::vector<double> Fa(no * no);
  active_fock_matrix(norb, ninact, nact, V, LDV, A1RDM, LDD1, Fa.data(), no);

  // Compute Q matrix
  std::vector<double> Q(na * no);
  aux_q_matrix(nact, norb, ninact, V, LDV, A2RDM, LDD2, Q.data(), na);

  // Compute generalized Fock
  std::vector<double> F(no * no);
  generalized_fock_matrix(norb, ninact, nact, Fi.data(), no, Fa.data(), no,
                          A1RDM, LDD1, Q.data(), na, F.data(), no);

  // Compute approximate diagonal hessian
  approx_diag_hessian(ninact, nact, num_virtual_orbitals, Fi.data(), no,
                      Fa.data(), no, A1RDM, LDD1, F.data(), no,
                      std::forward<Args>(args)...);
}

/**
 * @brief Contract the orbital-orbital Hessian with a vector
 *
 * Computes the matrix-vector product of the full orbital-orbital Hessian matrix
 * with an input vector K_lin. This is used in iterative optimization methods
 * that require Hessian-vector products without explicitly forming the full
 * Hessian matrix.
 *
 * @param norb Total number of orbitals
 * @param ninact Number of inactive orbitals
 * @param nact Number of active orbitals
 * @param nvirt Number of virtual orbitals
 * @param T One-electron kinetic energy integral matrix
 * @param LDT Leading dimension of T matrix
 * @param V Two-electron repulsion integral matrix
 * @param LDV Leading dimension of V matrix
 * @param A1RDM Active space one-particle reduced density matrix
 * @param LDD1 Leading dimension of A1RDM matrix
 * @param A2RDM Active space two-particle reduced density matrix
 * @param LDD2 Leading dimension of A2RDM matrix
 * @param OG Orbital gradient vector
 * @param K_lin Input vector for Hessian contraction
 * @param HK_lin Output vector containing the Hessian-vector product
 */
void orb_orb_hessian_contract(NumOrbital norb, NumInactive ninact,
                              NumActive nact, NumVirtual num_virtual_orbitals,
                              const double* T, size_t LDT, const double* V,
                              size_t LDV, const double* A1RDM, size_t LDD1,
                              const double* A2RDM, size_t LDD2,
                              const double* OG, const double* K_lin,
                              double* HK_lin);

}  // namespace macis
