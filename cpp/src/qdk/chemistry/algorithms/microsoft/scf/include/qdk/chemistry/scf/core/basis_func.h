// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstddef>
#include <cstdint>

namespace qdk::chemistry::scf {
/**
 * @brief Maximum number of primitive Gaussians in a contracted basis function
 */
inline static constexpr size_t MAX_CONTRACTION = 64;

/**
 * @brief Represents a single contracted Gaussian basis function
 *
 * A basis function is a linear combination of primitive Gaussian functions
 * (Gaussian-Type Orbitals, GTOs) centered at a specific atomic position.
 * The functional form is:
 *
 *   φ(r) = Σ_i c_i * exp(-α_i * |r - r0|²) * x^l * y^m * z^n
 *
 * where (l,m,n) are angular momentum quantum numbers, α_i are exponents,
 * c_i are contraction coefficients, and r0 is the center position.
 *
 * Angular momentum type is determined by L = l + m + n
 */
struct BasisFunc {
  uint64_t l;  ///< Angular momentum quantum number in x-direction (x^l)
  uint64_t m;  ///< Angular momentum quantum number in y-direction (y^m)
  uint64_t n;  ///< Angular momentum quantum number in z-direction (z^n)

  uint64_t n_exponents;  ///< Number of primitive Gaussians in contraction
  double exponents[MAX_CONTRACTION];     ///< Gaussian exponents
  double coefficients[MAX_CONTRACTION];  ///< Contraction coefficients

  double x0;  ///< x-coordinate of basis function center
  double y0;  ///< y-coordinate of basis function center
  double z0;  ///< z-coordinate of basis function center
};
}  // namespace qdk::chemistry::scf
