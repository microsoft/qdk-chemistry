/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstddef>

namespace macis {

/**
 * @brief Transforms a two-index matrix using orbital transformation
 * coefficients
 *
 * Performs the transformation: Y(p,q) = C(i,p) * X(i,j) * C(j,q)
 *
 * Matrix dimensions:
 * - X: [norb_old, norb_old] (input matrix)
 * - Y: [norb_new, norb_new] (output matrix)
 * - C: [norb_old, norb_new] (transformation coefficients)
 *
 * @param norb_old Number of orbitals in the original basis
 * @param norb_new Number of orbitals in the new basis
 * @param X Pointer to input matrix in column-major format
 * @param LDX Leading dimension of matrix X
 * @param C Pointer to transformation coefficient matrix in column-major format
 * @param LDC Leading dimension of matrix C
 * @param Y Pointer to output matrix in column-major format
 * @param LDY Leading dimension of matrix Y
 */
void two_index_transform(size_t norb_old, size_t norb_new, const double* X,
                         size_t LDX, const double* C, size_t LDC, double* Y,
                         size_t LDY);

/**
 * @brief Transforms a four-index tensor using orbital transformation
 * coefficients
 *
 * Performs the transformation:
 *
 * Y(p,q,r,s) = X(i,j,k,l) * C(i,p) * C(j,q) * C(k,r) * C(l,s)
 *
 * Tensor dimensions:
 * - X: [norb_old, norb_old, norb_old, norb_old] (input tensor)
 * - Y: [norb_new, norb_new, norb_new, norb_new] (output tensor)
 * - C: [norb_old, norb_new] (transformation coefficients)
 *
 * @param norb_old Number of orbitals in the original basis
 * @param norb_new Number of orbitals in the new basis
 * @param X Pointer to input tensor in column-major format
 * @param LDX Leading dimension of tensor X
 * @param C Pointer to transformation coefficient matrix in column-major format
 * @param LDC Leading dimension of matrix C
 * @param Y Pointer to output tensor in column-major format
 * @param LDY Leading dimension of tensor Y
 */
void four_index_transform(size_t norb_old, size_t norb_new, const double* X,
                          size_t LDX, const double* C, size_t LDC, double* Y,
                          size_t LDY);

}  // namespace macis
