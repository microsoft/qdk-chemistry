// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <complex>
#include <type_traits>

namespace qdk::chemistry::utils {

/**
 * @brief Type trait: true for @c std::complex<T> specializations.
 */
template <class S>
struct is_complex_scalar : std::false_type {};
template <class S>
struct is_complex_scalar<std::complex<S>> : std::true_type {};

template <class S>
inline constexpr bool is_complex_scalar_v = is_complex_scalar<S>::value;

}  // namespace qdk::chemistry::utils
