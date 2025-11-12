/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <vector>

namespace macis {

template <typename T>
double to_gib(const std::vector<T>& x) {
  return double(x.capacity() * sizeof(T)) / 1024. / 1024. / 1024.;
}

}  // namespace macis
