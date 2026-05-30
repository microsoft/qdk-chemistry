// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <complex>
#include <nlohmann/json.hpp>
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

/**
 * @brief Serialize a scalar (real or complex) to JSON.
 *
 * Real scalars are stored as a JSON number. Complex scalars are stored as a
 * two-element JSON array @c [real, imag].
 */
template <class S>
nlohmann::json scalar_to_json(const S& value) {
  if constexpr (is_complex_scalar_v<S>) {
    return nlohmann::json::array({value.real(), value.imag()});
  } else {
    return value;
  }
}

/**
 * @brief Deserialize a scalar (real or complex) from JSON.
 */
template <class S>
S scalar_from_json(const nlohmann::json& j) {
  if constexpr (is_complex_scalar_v<S>) {
    using Real = typename S::value_type;
    return S(j.at(0).get<Real>(), j.at(1).get<Real>());
  } else {
    return j.get<S>();
  }
}

}  // namespace qdk::chemistry::utils
