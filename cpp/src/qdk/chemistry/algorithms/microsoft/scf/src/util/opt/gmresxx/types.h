// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <functional>

namespace qdk::chemistry::scf {

/**
 * @brief Matrix-vector operation function type
 *
 * Computes Y = BETA * Y + ALPHA * A * X
 * Signature: (N, K, ALPHA, X, LDX, BETA, Y, LDY)
 */
template <typename T>
using matrix_op_t =
    std::function<void(int32_t, int32_t, T, const T*, int32_t, T, T*, int32_t)>;

/**
 * @brief Shifted matrix-vector operation function type
 *
 * Computes Z = BETA*Y + (B - S*I) * X for shifted linear systems
 */
template <typename T>
using shift_op_t = std::function<void(int32_t, int32_t, T, const T*, int32_t, T,
                                      const T*, int32_t, T*, int32_t)>;

/**
 * @brief Preconditioner operation function type
 *
 * Solves M*Y = X, stores result Y in X
 * Signature: (N, K, X, LDX)
 */
template <typename T>
using precond_op_t = std::function<void(int32_t, int32_t, T*, int32_t)>;

/**
 * @brief Shifted preconditioner operation function type
 *
 * Solves M(S)*Y(S) = X(S), stores result Y in X
 * Signature: (N, K, S, X, LDX)
 */
template <typename T>
using shifted_precond_op_t =
    std::function<void(int32_t, int32_t, T, T*, int32_t)>;
}  // namespace qdk::chemistry::scf
