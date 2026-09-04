// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <functional>
#include <utility>

namespace qdk::chemistry::utils {

// Lightweight helpers for local hash-code composition. Use hash_combine() for
// std::hash-compatible values used by in-memory lookup structures, especially
// in hot paths. These hashes are process-local hash codes, not content digests.

/**
 * @brief Combine an existing hash seed with the hash of another value.
 *
 * This helper is intended for fast local hash-code composition, not for
 * persistent content hashes.
 *
 * @tparam T Value type to hash.
 * @tparam Hasher Hash function type. Defaults to std::hash<T>.
 * @param seed Existing hash seed.
 * @param value Value to hash and combine into the seed.
 * @return Combined hash value.
 */
template <typename T, typename Hasher = std::hash<T>>
inline std::size_t hash_combine(std::size_t seed, const T& value) {
  Hasher hasher;
  return seed ^ (hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Combine an existing hash seed with multiple values from left to right.
 *
 * This helper is intended for fast local hash-code composition, not for
 * persistent content hashes.
 *
 * @tparam T First value type to hash.
 * @tparam Args Remaining value types to hash.
 * @param seed Existing hash seed.
 * @param value First value to hash and combine into the seed.
 * @param args Remaining values to hash and combine into the seed.
 * @return Combined hash value.
 */
template <typename T, typename... Args>
inline std::size_t hash_combine(std::size_t seed, const T& value,
                                Args&&... args) {
  return hash_combine(hash_combine(seed, value), std::forward<Args>(args)...);
}

}  // namespace qdk::chemistry::utils
