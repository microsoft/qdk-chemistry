/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <macis/mcscf/orbital_energies.hpp>

namespace macis {

void canonical_orbital_energies(NumOrbital norb, NumInactive ninact,
                                const double* T, size_t LDT, const double* V,
                                size_t LDV, double* eps) {
  const auto no = norb.get();
  const auto ni = ninact.get();

  const auto LDV2 = LDV * LDV;
  const auto LDV3 = LDV2 * LDV;

  for (size_t p = 0; p < no; ++p) {
    double tmp = 0.0;
    for (size_t i = 0; i < ni; ++i) {
      tmp += 2. * V[p * (LDV + 1) + i * (LDV2 + LDV3)] -
             V[p * (1 + LDV3) + i * (LDV + LDV2)];
    }
    eps[p] = T[p * (LDT + 1)] + tmp;
  }
}

}  // namespace macis
