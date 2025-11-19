/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sparsexx {

template <typename T, typename index_t = int64_t,
          typename Alloc = std::allocator<T> >
class csr_matrix;

template <typename T, typename index_t = int64_t,
          typename Alloc = std::allocator<T> >
class csc_matrix;

template <typename T, typename index_t = int64_t,
          typename Alloc = std::allocator<T> >
class coo_matrix;

}  // namespace sparsexx
