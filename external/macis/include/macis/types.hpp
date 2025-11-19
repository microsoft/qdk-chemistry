/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <strings.h>

#include <bit>
#include <bitset>
#include <mdspan/mdspan.hpp>
#include <vector>

namespace macis {

/// @brief Namespace alias for Kokkos extensions
namespace KokkosEx = Kokkos;

/**
 * @brief Column-major multidimensional span type.
 * @tparam T Element type
 * @tparam rank Number of dimensions
 */
template <typename T, size_t rank>
using col_major_span =
    Kokkos::mdspan<T, Kokkos::dextents<size_t, rank>, Kokkos::layout_left>;

/**
 * @brief Two-dimensional matrix span with column-major layout.
 * @tparam T Element type
 */
template <typename T>
using matrix_span = col_major_span<T, 2>;

/**
 * @brief Three-dimensional span with column-major layout.
 * @tparam T Element type
 */
template <typename T>
using rank3_span = col_major_span<T, 3>;

/**
 * @brief Four-dimensional span with column-major layout.
 * @tparam T Element type
 */
template <typename T>
using rank4_span = col_major_span<T, 4>;

/**
 * @brief Bitset-based wavefunction type.
 * @tparam N Orbital capacity for the wavefunction
 */
template <size_t N>
using wfn_t = std::bitset<N>;

/**
 * @brief Iterator type for wavefunction vectors.
 * @tparam Orbital capacity for the wavefunction
 */
template <size_t N>
using wavefunction_iterator_t = typename std::vector<std::bitset<N> >::iterator;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
/// @brief 128-bit unsigned integer type
using uint128_t = __uint128_t;
#pragma GCC diagnostic pop

/**
 * @brief A type-safe wrapper that associates a value with a specific parameter
 * type.
 * @tparam T The underlying value type
 * @tparam ParameterType A tag type to distinguish different named types
 */
template <typename T, typename ParameterType>
class NamedType {
 public:
  /**
   * @brief Default constructor.
   */
  constexpr explicit NamedType() : value_() {}

  /**
   * @brief Constructor from const reference.
   * @param[in] value The value to wrap
   */
  constexpr explicit NamedType(T const& value) : value_(value) {}

  /**
   * @brief Constructor from rvalue reference.
   * @param[in] value The value to wrap (moved)
   */
  constexpr explicit NamedType(T&& value) : value_(std::move(value)) {}

  /**
   * @brief Copy constructor.
   * @param[in] other The NamedType to copy from
   */
  constexpr NamedType(const NamedType& other) : value_(other.get()) {}

  /**
   * @brief Move constructor.
   * @param[in] other The NamedType to move from
   */
  constexpr NamedType(NamedType&& other) noexcept
      : value_(std::move(other.get())) {};

  /**
   * @brief Copy assignment operator.
   * @param[in] other The NamedType to copy from
   * @return Reference to this object
   */
  constexpr NamedType& operator=(const NamedType& other) {
    value_ = other.get();
    return *this;
  }

  /**
   * @brief Move assignment operator.
   * @param[in] other The NamedType to move from
   * @return Reference to this object
   */
  constexpr NamedType& operator=(NamedType&& other) noexcept {
    value_ = std::move(other.get());
    return *this;
  }

  /**
   * @brief Get mutable reference to the wrapped value.
   * @return Reference to the wrapped value
   */
  constexpr T& get() { return value_; }

  /**
   * @brief Get const reference to the wrapped value.
   * @return Const reference to the wrapped value
   */
  constexpr T const& get() const { return value_; }

 private:
  T value_;  ///< The wrapped value
};

/// @brief Type-safe wrapper for number of electrons
using NumElectron = NamedType<size_t, struct nelec_type>;

/// @brief Type-safe wrapper for number of orbitals
using NumOrbital = NamedType<size_t, struct norb_type>;

/// @brief Type-safe wrapper for number of active orbitals
using NumActive = NamedType<size_t, struct nactive_type>;

/// @brief Type-safe wrapper for number of inactive orbitals
using NumInactive = NamedType<size_t, struct ninactive_type>;

/// @brief Type-safe wrapper for number of virtual orbitals
using NumVirtual = NamedType<size_t, struct nvirtual_type>;

}  // namespace macis
