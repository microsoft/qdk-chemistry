/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstddef>

namespace macis {

// Y(p,q) = C(i,p) * X(i,j) * C(j,q)
// X <- [norb_old, norb_old]
// Y <- [norb_new, norb_new]
// C <- [norb_old, norb_new]
void two_index_transform(size_t norb_old, size_t norb_new, const double* X,
                         size_t LDX, const double* C, size_t LDC, double* Y,
                         size_t LDY);

// Y(p,q,r,s) = X(i,j,k,l) * C(i,p) * C(j,q) * C(k,r) * C(l,s)
// X <- [norb_old, norb_old, norb_old, norb_old]
// Y <- [norb_new, norb_new, norb_new, norb_new]
// C <- [norb_old, norb_new]
void four_index_transform(size_t norb_old, size_t norb_new, const double* X,
                          size_t LDX, const double* C, size_t LDC, double* Y,
                          size_t LDY);

}  // namespace macis
