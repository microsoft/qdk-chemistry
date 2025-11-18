/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <vector>

namespace macis {

/**
 * @brief Converts the memory capacity of a vector to gibibytes (GiB).
 *
 * Calculates the total memory capacity allocated by a vector in gibibytes.
 * The capacity (not size) is used to account for all allocated memory,
 * including unused reserve space.
 *
 * @tparam T The type of elements stored in the vector.
 * @param x The vector whose memory capacity is to be calculated.
 * @return The memory capacity in gibibytes (GiB).
 */
template <typename T>
double to_gib(const std::vector<T>& x) {
  return double(x.capacity() * sizeof(T)) / 1024. / 1024. / 1024.;
}

}  // namespace macis
