// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <complex>
#include <type_traits>

namespace qdk::chemistry::utils {

/**
 * @brief Type trait: true for @c std::complex<T> specializations.
 *
 * Default specialization yields @c false.
 *
 * @tparam S Scalar type under test.
 */
template <class S>
struct is_complex_scalar : std::false_type {};

/**
 * @brief Specialization that yields @c true for @c std::complex<S>.
 *
 * @tparam S Underlying real type of the complex specialization.
 */
template <class S>
struct is_complex_scalar<std::complex<S>> : std::true_type {};

/**
 * @brief Convenience value template: @c true iff @p S is a
 * @c std::complex<T> specialization.
 *
 * @tparam S Scalar type under test (typically @c double or
 *         @c std::complex<double>).
 */
template <class S>
inline constexpr bool is_complex_scalar_v = is_complex_scalar<S>::value;

}  // namespace qdk::chemistry::utils
