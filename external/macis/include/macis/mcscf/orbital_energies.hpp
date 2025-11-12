/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>

namespace macis {

void canonical_orbital_energies(NumOrbital norb, NumInactive ninact,
                                const double* T, size_t LDT, const double* V,
                                size_t LDV, double* eps);

}
