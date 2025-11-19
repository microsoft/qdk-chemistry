/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <macis/mcscf/orbital_hessian.hpp>
#include <macis/mcscf/orbital_steps.hpp>

namespace macis {

void precond_cg_orbital_step(NumOrbital norb, NumInactive ninact,
                             NumActive nact, NumVirtual num_virtual_orbitals,
                             const double* Fi, size_t LDFi, const double* Fa,
                             size_t LDFa, const double* F, size_t LDF,
                             const double* A1RDM, size_t LDD, const double* OG,
                             double* K_lin) {
  const size_t no = norb.get(), ni = ninact.get(), na = nact.get(),
               nv = num_virtual_orbitals.get(),
               orb_rot_sz = nv * (na + ni) + na * ni;
  std::vector<double> DH(orb_rot_sz);

  // Compute approximate diagonal hessian
  approx_diag_hessian(ninact, nact, num_virtual_orbitals, Fi, LDFi, Fa, LDFa,
                      A1RDM, LDD, F, LDF, DH.data());

  // Precondition the gradient
  for (size_t p = 0; p < orb_rot_sz; ++p) {
    K_lin[p] = -OG[p] / DH[p];
  }
}

}  // namespace macis
