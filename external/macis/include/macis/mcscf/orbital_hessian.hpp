/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/mcscf/fock_matrices.hpp>
#include <macis/mcscf/orbital_rotation_utilities.hpp>
#include <macis/types.hpp>

namespace macis {

void approx_diag_hessian(NumInactive _ni, NumActive _na, NumVirtual _nv,
                         const double* Fi, size_t LDFi, const double* Fa,
                         size_t LDFa, const double* A1RDM, size_t LDD,
                         const double* F, size_t LDF, double* H_vi,
                         double* H_va, double* H_ai);

inline void approx_diag_hessian(NumInactive ni, NumActive na, NumVirtual nv,
                                const double* Fi, size_t LDFi, const double* Fa,
                                size_t LDFa, const double* A1RDM, size_t LDD,
                                const double* F, size_t LDF, double* H_lin) {
  auto [H_vi, H_va, H_ai] = split_linear_orb_rot(ni, na, nv, H_lin);
  approx_diag_hessian(ni, na, nv, Fi, LDFi, Fa, LDFa, A1RDM, LDD, F, LDF, H_vi,
                      H_va, H_ai);
}

template <typename... Args>
void approx_diag_hessian(NumOrbital norb, NumInactive ninact, NumActive nact,
                         NumVirtual nvirt, const double* T, size_t LDT,
                         const double* V, size_t LDV, const double* A1RDM,
                         size_t LDD1, const double* A2RDM, size_t LDD2,
                         Args&&... args) {
  const size_t no = norb.get();
  const size_t ni = ninact.get();
  const size_t na = nact.get();
  const size_t nv = nvirt.get();

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
  approx_diag_hessian(ninact, nact, nvirt, Fi.data(), no, Fa.data(), no, A1RDM,
                      LDD1, F.data(), no, std::forward<Args>(args)...);
}

void orb_orb_hessian_contract(NumOrbital norb, NumInactive ninact,
                              NumActive nact, NumVirtual nvirt, const double* T,
                              size_t LDT, const double* V, size_t LDV,
                              const double* A1RDM, size_t LDD1,
                              const double* A2RDM, size_t LDD2,
                              const double* OG, const double* K_lin,
                              double* HK_lin);

}  // namespace macis
