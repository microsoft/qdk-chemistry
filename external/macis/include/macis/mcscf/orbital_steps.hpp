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

void precond_cg_orbital_step(NumOrbital norb, NumInactive ninact,
                             NumActive nact, NumVirtual nvirt, const double* Fi,
                             size_t LDFi, const double* Fa, size_t LDFa,
                             const double* F, size_t LDF, const double* A1RDM,
                             size_t LDD, const double* OG, double* K_lin);

}
