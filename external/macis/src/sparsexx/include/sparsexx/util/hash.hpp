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

template <typename T, typename Hasher = std::hash<T>>
inline static std::size_t hash_combine(std::size_t seed, const T& v) {
  Hasher h;
  return seed ^ (h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

struct pair_hasher {
  template <typename T1, typename T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    std::size_t seed = 0;
    seed = hash_combine(seed, p.first);
    seed = hash_combine(seed, p.second);
    return seed;
  }
};

}  // namespace sparsexx::detail
