// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace qdk::chemistry::scf::blas {

/**
 * @brief Matrix-vector multiplication: y = alpha*op(A)*x + beta*y
 * @see https://netlib.org/lapack/explore-html/dd/d09/group__gemv.html for
 * details
 */
void gemv(const char* TRANS, int M, int N, double ALPHA, const double* A,
          int LDA, const double* X, int INCX, double BETA, double* Y, int INCY);

/**
 * @brief Matrix-matrix multiplication: C = alpha*op(A)*op(B) + beta*C
 * @see https://netlib.org/lapack/explore-html/dd/d09/group__gemm.html for
 * details
 */
void gemm(const char* TRANSA, const char* TRANSB, int M, int N, int K,
          double ALPHA, const double* A, int LDA, const double* B, int LDB,
          double BETA, double* C, int LDC);

}  // namespace qdk::chemistry::scf::blas
