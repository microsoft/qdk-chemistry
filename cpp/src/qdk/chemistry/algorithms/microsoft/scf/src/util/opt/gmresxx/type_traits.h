// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <complex>
#include <type_traits>

namespace qdk::chemistry::scf::detail {
/**
 * @brief Type trait to extract the real component type from complex or real
 * types
 *
 * Primary template that maps a type to itself. This handles the case where
 * the input type is already real (e.g., float, double).
 *
 * @tparam T The input type (real or complex)
 */
template <typename T>
struct make_real {
  using type = T;
};

/**
 * @brief Specialization for complex types to extract the underlying real type
 *
 * Template specialization that extracts the underlying real type from
 * std::complex<T>. For example, std::complex<double> -> double.
 *
 * @tparam T The underlying real type of the complex number
 */
template <typename T>
struct make_real<std::complex<T> > {
  using type = T;
};

/**
 * @brief Alias template for convenient access to the real component type
 *
 * Provides a convenient shorthand for accessing the real type associated
 * with either real or complex input types.
 *
 * @tparam T Input type (real or complex)
 */
template <typename T>
using real_t = typename make_real<T>::type;

/**
 * @brief Type trait to check if a type is real (not complex)
 *
 * Compile-time boolean that is true if T is a real type and false if T
 * is a complex type. Used for template specialization and SFINAE.
 *
 * @tparam T Type to check
 */
template <typename T>
struct is_real {
  static constexpr bool value = std::is_same<T, real_t<T> >::value;
};

/**
 * @brief Returns the input unchanged for real types (conjugate is identity)
 *
 * For real-valued types, the complex conjugate operation is the identity
 * function. This overload is selected when T is a real type.
 *
 * @tparam T Real numeric type
 * @param x Input value
 * @return The input value unchanged
 */
template <typename T>
std::enable_if_t<is_real<T>::value, T> smart_conj(T x) {
  return x;
}

/**
 * @brief Returns the complex conjugate for complex types
 *
 * For complex-valued types, returns the complex conjugate using std::conj.
 * This overload is selected when T is a complex type.
 *
 * @tparam T Complex numeric type
 * @param x Input complex value
 * @return Complex conjugate of the input
 */
template <typename T>
std::enable_if_t<!is_real<T>::value, T> smart_conj(T x) {
  return std::conj(x);
}
}  // namespace qdk::chemistry::scf::detail
