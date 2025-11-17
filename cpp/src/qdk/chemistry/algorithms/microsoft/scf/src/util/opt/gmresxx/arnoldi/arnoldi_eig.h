// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "arnoldi.h"

namespace qdk::chemistry::scf {
/**
 * @brief Perform a fixed-dimension Arnoldi iteration for the eigenvalue problem
 *
 * @param[in] N Problem dimension
 * @param[in] MAX_M Arnoldi dimension
 * @param[in] A Operator action instance
 * @param[in] V0 Initial vector for the Arnoldi iteration
 * @param[out] Q The Krylov subspace
 * @param[in]  LDQ Leading dimension of Q
 * @param[out] H A projected into the Krylov subspace
 * @param[in]  LDH Leading dimension of H
 */
template <typename T>
void arnoldi_eig(int_t N, int_t MAX_M, const matrix_op_t<T>& A, const T* V0,
                 T* Q, int_t LDQ, T* H, int_t LDH) {
  assert(LDQ >= N);
  assert(LDH >= (MAX_M + 1));

  A(N, 1, T(1.), V0, N, T(0.), Q, LDQ);
  auto rnorm = two_norm(N, Q, 1);
  lascal(N, 1, 1. / rnorm, Q, LDQ);

  for (int_t m = 0; m < MAX_M; ++m) {
    auto* Qmp1 = Q + (m + 1) * LDQ;
    auto* Hm = H + m * LDH;

    arnoldi(N, m + 1, A, Q, LDQ, Qmp1, Hm);
  }
}
}  // namespace qdk::chemistry::scf
