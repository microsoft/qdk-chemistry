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
#include <tuple>

namespace macis {

/**
 *  @brief Splits a linear offsettable (pointer, index, etc) input into
 *  virtual-inactive, virtual-active, and active-inactive pieces
 *
 *  @tparam T Type of offsettable input
 *
 *  @param[in] ni Number of inactive orbitals
 *  @param[in] na Numner of active orbitals
 *  @param[in] nv Number of virtual orbitals
 *
 *  @returns a tuple containing the virtual-inactive, virtual-active,
 *  and active-inactive blocks corresponding to the the linear input.
 */
template <typename T>
auto split_linear_orb_rot(NumInactive ni, NumActive na, NumVirtual nv,
                          T&& V_lin) {
  auto V_vi = V_lin;
  auto V_va = V_vi + nv.get() * ni.get();
  auto V_ai = V_va + nv.get() * na.get();
  return std::make_tuple(V_vi, V_va, V_ai);
}

/**
 *  @brief Convert a set of linear orbital rotation vectors
 *  into a full anti-symmetric matrix.
 *
 *  @param[in] ni    Number of inactive orbitals
 *  @param[in] na    Numner of active orbitals
 *  @param[in] nv    Number of virtual orbitals
 *  @param[in] K_vi  Virtual-inactive block of the orbital rotation vector
 *  @param[in] K_va  Virtual-active block of the orbital rotation vector
 *  @param[in] K_ai  Active-inactive block of the orbital rotation vector
 *  @param[out] K    Full antisymmetric matrix corresponding to linear input
 *  @param[in]  LDK  Leading dimenion of K
 */
void linear_orb_rot_to_matrix(NumInactive ni, NumActive na, NumVirtual nv,
                              const double* K_vi, const double* K_va,
                              const double* K_ai, double* K, size_t LDK);

/**
 *  @brief Convert a packed linear orbital rotation vector
 *  into a full anti-symmetric matrix.
 *
 *  @param[in] ni     Number of inactive orbitals
 *  @param[in] na     Numner of active orbitals
 *  @param[in] nv     Number of virtual orbitals
 *  @param[in] K_lin  Packed orbital rotation vector
 *  @param[out] K     Full antisymmetric matrix corresponding to linear input
 *  @param[in]  LDK   Leading dimenion of K
 */
inline void linear_orb_rot_to_matrix(NumInactive ni, NumActive na,
                                     NumVirtual nv, const double* K_lin,
                                     double* K, size_t LDK) {
  auto [K_vi, K_va, K_ai] = split_linear_orb_rot(ni, na, nv, K_lin);
  linear_orb_rot_to_matrix(ni, na, nv, K_vi, K_va, K_ai, K, LDK);
}

/**
 *  @brief Convert a full anti-symmetric matrix into separate linear
 *  orbital rotation vectors.
 *
 *  @param[in] _ni   Number of inactive orbitals
 *  @param[in] _na   Number of active orbitals
 *  @param[in] _nv   Number of virtual orbitals
 *  @param[in] F     Full antisymmetric matrix to convert
 *  @param[in] LDF   Leading dimension of F
 *  @param[out] G_vi Virtual-inactive block of the orbital rotation vector
 *  @param[out] G_va Virtual-active block of the orbital rotation vector
 *  @param[out] G_ai Active-inactive block of the orbital rotation vector
 */
void matrix_to_linear_orb_rot(NumInactive _ni, NumActive _na, NumVirtual _nv,
                              const double* F, size_t LDF, double* G_vi,
                              double* G_va, double* G_ai);

/**
 *  @brief Convert a full anti-symmetric matrix into a packed linear
 *  orbital rotation vector.
 *
 *  @param[in] ni    Number of inactive orbitals
 *  @param[in] na    Number of active orbitals
 *  @param[in] nv    Number of virtual orbitals
 *  @param[in] F     Full antisymmetric matrix to convert
 *  @param[in] LDF   Leading dimension of F
 *  @param[out] G_lin Packed orbital rotation vector output
 */
inline void matrix_to_linear_orb_rot(NumInactive ni, NumActive na,
                                     NumVirtual nv, const double* F, size_t LDF,
                                     double* G_lin) {
  auto [G_vi, G_va, G_ai] = split_linear_orb_rot(ni, na, nv, G_lin);
  matrix_to_linear_orb_rot(ni, na, nv, F, LDF, G_vi, G_va, G_ai);
}

/**
 *  @brief Convert a Fock matrix into separate linear orbital gradient vectors.
 *
 *  @param[in] _ni   Number of inactive orbitals
 *  @param[in] _na   Number of active orbitals
 *  @param[in] _nv   Number of virtual orbitals
 *  @param[in] F     Fock matrix input
 *  @param[in] LDF   Leading dimension of F
 *  @param[out] G_vi Virtual-inactive block of the orbital gradient vector
 *  @param[out] G_va Virtual-active block of the orbital gradient vector
 *  @param[out] G_ai Active-inactive block of the orbital gradient vector
 */
void fock_to_linear_orb_grad(NumInactive _ni, NumActive _na, NumVirtual _nv,
                             const double* F, size_t LDF, double* G_vi,
                             double* G_va, double* G_ai);

/**
 *  @brief Convert a Fock matrix into a packed linear orbital gradient vector.
 *
 *  @param[in] ni    Number of inactive orbitals
 *  @param[in] na    Number of active orbitals
 *  @param[in] nv    Number of virtual orbitals
 *  @param[in] F     Fock matrix input
 *  @param[in] LDF   Leading dimension of F
 *  @param[out] G_lin Packed orbital gradient vector output
 */
inline void fock_to_linear_orb_grad(NumInactive ni, NumActive na, NumVirtual nv,
                                    const double* F, size_t LDF,
                                    double* G_lin) {
  auto [G_vi, G_va, G_ai] = split_linear_orb_rot(ni, na, nv, G_lin);
  fock_to_linear_orb_grad(ni, na, nv, F, LDF, G_vi, G_va, G_ai);
}

}  // namespace macis
