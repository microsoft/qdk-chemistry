// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <blas.hh>
#include <cstdint>
#include <vector>

namespace qdk::chemistry::scf {

/**
 * @brief Compute C = ALPHA * A**H * B * A + BETA * C using BLAS GEMM
 *
 * Performs a similarity (congruence) transformation of B by A, with optional
 * scaling and accumulation into C.
 *
 * Matrix shapes:
 * A: K x N
 * B: K x K  (square)
 * C: N x N  (square)
 *
 * If workspace is nullptr the function allocates its own internal buffer;
 * otherwise the caller-supplied buffer is used (and resized if too small).
 * Passing a pre-allocated workspace avoids repeated heap allocations across
 * consecutive calls.
 *
 * @param[in]     layout    BLAS matrix storage layout (RowMajor or ColMajor).
 * @param[in]     N         Column count of A and dimension of square C.
 * @param[in]     K         Row count of A and dimension of square B.
 * @param[in]     ALPHA     Scalar multiplier for A**H * B * A.
 * @param[in]     A         Pointer to the A matrix (K x N).
 * @param[in]     LDA       Leading dimension of A.
 * @param[in]     B         Pointer to the B matrix (K x K).
 * @param[in]     LDB       Leading dimension of B.
 * @param[in]     BETA      Scalar multiplier for the input C matrix.
 * @param[in,out] C         Pointer to the C matrix (N x N); overwritten on
 *                          output.
 * @param[in]     LDC       Leading dimension of C.
 * @param[in,out] workspace Optional temporary buffer of at least K*N doubles.
 *                          Resized automatically if smaller than required.
 *                          Pass nullptr to let the function allocate
 *                          internally.
 */
void similarity_transform(blas::Layout layout, int64_t N, int64_t K,
                          double ALPHA, const double* A, int64_t LDA,
                          const double* B, int64_t LDB, double BETA, double* C,
                          int64_t LDC,
                          std::vector<double>* workspace = nullptr);

}  // namespace qdk::chemistry::scf
