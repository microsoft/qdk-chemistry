/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <macis/mcscf/orbital_rotation_utilities.hpp>

namespace macis {

void linear_orb_rot_to_matrix(NumInactive _ni, NumActive _na, NumVirtual _nv,
                              const double* K_vi, const double* K_va,
                              const double* K_ai, double* K, size_t LDK) {
  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

  // Virtual - Inactive Block
  for (size_t i = 0; i < ni; ++i)
    for (size_t a = 0; a < nv; ++a) {
      const auto a_off = a + ni + na;
      const auto k = K_vi[a + i * nv];
      K[a_off + i * LDK] = k;
      K[i + a_off * LDK] = -k;
    }

  // Virtual - Active Block
  for (size_t i = 0; i < na; ++i)
    for (size_t a = 0; a < nv; ++a) {
      const auto i_off = i + ni;
      const auto a_off = a + ni + na;
      const auto k = K_va[a + i * nv];
      K[a_off + i_off * LDK] = k;
      K[i_off + a_off * LDK] = -k;
    }

  // Active - Inactive Block
  for (size_t i = 0; i < ni; ++i)
    for (size_t a = 0; a < na; ++a) {
      const auto a_off = a + ni;
      const auto k = K_ai[a + i * na];
      K[a_off + i * LDK] = k;
      K[i + a_off * LDK] = -k;
    }
}

void matrix_to_linear_orb_rot(NumInactive _ni, NumActive _na, NumVirtual _nv,
                              const double* F, size_t LDF, double* G_vi,
                              double* G_va, double* G_ai) {
  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

#define FOCK(i, j) F[i + j * LDF]

  // Virtual - Inactive Block
  for (size_t i = 0; i < ni; ++i)
    for (size_t a = 0; a < nv; ++a) {
      const auto a_off = a + ni + na;
      G_vi[a + i * nv] = FOCK(a_off, i);
    }

  // Virtual - Active Block
  for (size_t i = 0; i < na; ++i)
    for (size_t a = 0; a < nv; ++a) {
      const auto i_off = i + ni;
      const auto a_off = a + ni + na;
      G_va[a + i * nv] = FOCK(a_off, i_off);
    }

  // Active - Inactive Block
  for (size_t i = 0; i < ni; ++i)
    for (size_t a = 0; a < na; ++a) {
      const auto a_off = a + ni;
      G_ai[a + i * na] = FOCK(a_off, i);
    }

#undef FOCK
}

void fock_to_linear_orb_grad(NumInactive _ni, NumActive _na, NumVirtual _nv,
                             const double* F, size_t LDF, double* G_vi,
                             double* G_va, double* G_ai) {
  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

#define FOCK(i, j) F[i + j * LDF]

  // Virtual - Inactive Block
  for (size_t i = 0; i < ni; ++i)
    for (size_t a = 0; a < nv; ++a) {
      const auto a_off = a + ni + na;
      G_vi[a + i * nv] = 2 * (FOCK(a_off, i) - FOCK(i, a_off));
    }

  // Virtual - Active Block
  for (size_t i = 0; i < na; ++i)
    for (size_t a = 0; a < nv; ++a) {
      const auto i_off = i + ni;
      const auto a_off = a + ni + na;
      G_va[a + i * nv] = 2 * (FOCK(a_off, i_off) - FOCK(i_off, a_off));
    }

  // Active - Inactive Block
  for (size_t i = 0; i < ni; ++i)
    for (size_t a = 0; a < na; ++a) {
      const auto a_off = a + ni;
      G_ai[a + i * na] = 2 * (FOCK(a_off, i) - FOCK(i, a_off));
    }

#undef FOCK
}

}  // namespace macis
