// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "../linalg.h"
#include "../type_traits.h"

namespace qdk::chemistry::scf {
/**
 *  @brief Perform a single step in the Arnoldi iteration given the result
 *         of the application of the system matrix onto the previous Krylov
 *         vector
 *
 *  @tparam T Floating point type
 *
 *  @param[in]     N   Problem Dimension
 *  @param[in]     K   Current Dimension of the Krylov subspace in Q
 *  @param[in]     Q   Current Krylov subspace
 *  @param[in]     LDQ Leading dimension of Q
 *  @param[in/out] Qn  The new Krylov vector
 *                     On input:  A * Q(:,K-1)
 *                     On output: Normalized krylov vector
 *  @param[out]    H   New column of the Hessenberg matrix
 */
template <typename T>
void arnoldi_iter(int32_t N, int32_t K, const T* Q, int32_t LDQ, T* Qn, T* H) {
  // Modified GS
  for (int32_t k = 0; k < K; ++k) {
    const auto* Qk = Q + k * LDQ;
    auto h = inner(N, Qk, 1, Qn, 1);
    laxpby(N, 1, -h, Qk, N, T(1.), Qn, N);
    H[k] = h;

    // Reorthogonalization
    h = inner(N, Qk, 1, Qn, 1);
    laxpby(N, 1, -h, Qk, N, T(1.), Qn, N);
    H[k] += h;
  }

  // Normalize
  H[K] = two_norm(N, Qn, 1);
  lascal(N, 1, 1. / H[K], Qn, N);
}

/**
 *  @brief Perform a single step in the Arnoldi iteration
 *
 *  @tparam T Floating point type
 *
 *  @param[in]  N   Problem Dimension
 *  @param[in]  K   Current Dimension of the Krylov subspace in Q
 *  @param[in]  A   Functor for the application of the problem matrix
 *  @param[in]  Q   Current Krylov subspace
 *  @param[in]  LDQ Leading dimension of Q
 *  @param[out] Qn  The new Krylov vector
 *  @param[out] H   New column of the Hessenberg matrix
 */
template <typename T>
void arnoldi(int32_t N, int32_t K, const matrix_op_t<T>& A, const T* Q,
             int32_t LDQ, T* Qn, T* H) {
  // Qn = A * Q(:, K-1)
  A(N, 1, T(1.), Q + (K - 1) * LDQ, LDQ, T(0.), Qn, N);

  // Perform the rest of the iteration
  arnoldi_iter(N, K, Q, LDQ, Qn, H);
}
}  // namespace qdk::chemistry::scf
