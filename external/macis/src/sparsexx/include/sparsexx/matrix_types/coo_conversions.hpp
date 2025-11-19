/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cassert>

#include "coo_matrix.hpp"
#include "csr_matrix.hpp"

namespace sparsexx {

/**
 * @brief Constructor to create a COO matrix from a CSR matrix.
 *
 * This constructor converts a CSR (Compressed Sparse Row) matrix to COO
 * (Coordinate) format. The conversion expands the compressed row representation
 * into explicit row indices, creating triplet format (row, column, value)
 * suitable for COO storage.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 * @tparam Alloc The allocator type for memory management
 *
 * @param other The CSR matrix to convert from
 *
 * @note The resulting COO matrix will have the same indexing base as the input
 * CSR matrix
 * @note Matrix dimensions and non-zero count are preserved exactly
 * @note The COO matrix will maintain the same element ordering as the CSR
 * matrix
 *
 * @see csr_matrix for details on CSR format
 */
template <typename T, typename index_t, typename Alloc>
coo_matrix<T, index_t, Alloc>::coo_matrix(
    const csr_matrix<T, index_t, Alloc>& other)
    : coo_matrix(other.m(), other.n(), other.nnz(), other.indexing()) {
  auto rowind_it = rowind_.begin();
  for (size_t i = 0; i < m_; ++i) {
    const auto row_count = other.rowptr()[i + 1] - other.rowptr()[i];
    rowind_it = std::fill_n(rowind_it, row_count, i + indexing_);
  }
  assert(rowind_it == rowind_.end());

  std::copy(other.colind().begin(), other.colind().end(), colind_.begin());
  std::copy(other.nzval().begin(), other.nzval().end(), nzval_.begin());
}

}  // namespace sparsexx
