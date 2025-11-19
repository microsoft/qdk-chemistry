#pragma once
#include <complex>

namespace lobpcgxx {

namespace detail {

/**
 * @brief Type trait to extract the underlying real type from numeric types.
 *
 * For real types (float, double), this trait simply returns the same type.
 * This is the primary template that handles non-complex types.
 *
 * @tparam T Input numeric type
 */
template <typename T>
struct real {
  using type = T;  ///< For real types, the real type is the type itself
};

/**
 * @brief Specialization of real type trait for complex numbers.
 *
 * For complex types (std::complex<float>, std::complex<double>), this trait
 * extracts the underlying real component type (float or double respectively).
 * This specialization is essential for LOBPCG operations that need to work
 * with real eigenvalues even when the matrix elements are complex.
 *
 * @tparam T The real component type of the complex number
 */
template <typename T>
struct real<std::complex<T>> {
  using type = T;  ///< The underlying real type of the complex number
};

/**
 * @brief Convenience alias template for extracting real types.
 *
 * This alias template provides a shorter syntax for accessing the real type
 * from the real type trait. It automatically deduces whether the input type
 * is real or complex and returns the appropriate real component type.
 *
 * Usage examples:
 * - real_t<double> = double
 * - real_t<float> = float
 * - real_t<std::complex<double>> = double
 * - real_t<std::complex<float>> = float
 *
 * @tparam T Input numeric type (real or complex)
 */
template <typename T>
using real_t = typename real<T>::type;

}  // namespace detail

}  // namespace lobpcgxx
