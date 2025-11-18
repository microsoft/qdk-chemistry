/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <functional>

namespace sparsexx::detail {

/**
 * @brief Combines a hash value with the hash of another value
 *
 * This function implements a hash combination algorithm that takes an existing
 * hash seed and combines it with the hash of a new value. It uses a modified
 * version of the boost::hash_combine algorithm with a magic constant for
 * better hash distribution.
 *
 * @tparam T The type of value to hash
 * @tparam Hasher The hash function type (defaults to std::hash<T>)
 * @param seed The existing hash value to combine with
 * @param v The value to hash and combine
 * @return The combined hash value
 *
 * @note The algorithm uses the magic constant 0x9e3779b9 (golden ratio) and
 *       bit shifting for good hash distribution properties
 */
template <typename T, typename Hasher = std::hash<T>>
inline static std::size_t hash_combine(std::size_t seed, const T& v) {
  Hasher h;
  return seed ^ (h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Hash function object for std::pair types
 *
 * This struct provides a hash function for std::pair objects by combining
 * the hashes of both elements using the hash_combine function. It can be
 * used with hash-based containers like std::unordered_map or std::unordered_set
 * that need to store pairs as keys.
 *
 * @note The hash is computed by starting with seed 0 and combining the hashes
 *       of first and second elements in order
 */
struct pair_hasher {
  /**
   * @brief Hash function operator for std::pair
   *
   * Computes a hash value for a std::pair by combining the hashes of both
   * elements using the hash_combine function.
   *
   * @tparam T1 Type of the first element in the pair
   * @tparam T2 Type of the second element in the pair
   * @param p The pair to hash
   * @return The computed hash value for the pair
   */
  template <typename T1, typename T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    std::size_t seed = 0;
    seed = hash_combine(seed, p.first);
    seed = hash_combine(seed, p.second);
    return seed;
  }
};

}  // namespace sparsexx::detail
