// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <functional>
#include <utility>

namespace qdk::chemistry::utils {

/**
 * @brief Combines a hash value with the hash of another value.
 *
 * This function implements a hash combination algorithm that takes an existing
 * hash seed and combines it with the hash of a new value. It uses a modified
 * version of the boost::hash_combine algorithm with the golden ratio constant
 * for better hash distribution.
 *
 * @tparam T The type of value to hash.
 * @tparam Hasher The hash function type (defaults to std::hash<T>).
 * @param seed The existing hash value to combine with.
 * @param v The value to hash and combine.
 * @return The combined hash value.
 *
 * @note The algorithm uses the magic constant 0x9e3779b9 (derived from the
 *       golden ratio) and bit shifting for good hash distribution properties.
 */
template <typename T, typename Hasher = std::hash<T>>
inline std::size_t hash_combine(std::size_t seed, const T& v) {
  Hasher h;
  return seed ^ (h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Variadic overload that combines a hash seed with multiple values.
 *
 * This function recursively combines the hash of multiple values into a single
 * hash value, processing values left-to-right.
 *
 * @tparam T The type of the first value to hash.
 * @tparam Args The types of the remaining values to hash.
 * @param seed The existing hash value to combine with.
 * @param v The first value to hash and combine.
 * @param args The remaining values to hash and combine.
 * @return The combined hash value.
 *
 * Example:
 * @code
 *   std::size_t h = hash_combine(0, x, y, z);  // Combines hashes of x, y, z
 * @endcode
 */
template <typename T, typename... Args>
inline std::size_t hash_combine(std::size_t seed, const T& v, Args&&... args) {
  return hash_combine(hash_combine(seed, v), std::forward<Args>(args)...);
}

}  // namespace qdk::chemistry::utils
