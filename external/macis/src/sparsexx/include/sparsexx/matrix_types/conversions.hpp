/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <stdexcept>

#include "coo_matrix.hpp"
#include "csc_matrix.hpp"
#include "csr_matrix.hpp"

namespace sparsexx {

namespace detail {

/**
 * @brief Converts CSR (Compressed Sparse Row) format to CSC (Compressed Sparse
 * Column) format.
 *
 * This function performs an in-place conversion from CSR format to CSC format
 * using an algorithm adapted from SciPy's csr_tocsc implementation. The
 * conversion handles arbitrary indexing bases (0-based or 1-based) and
 * efficiently reorders the sparse matrix data.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 *
 * @param M Number of rows in the input CSR matrix
 * @param N Number of columns in the input CSR matrix
 * @param Ap Input CSR row pointer array (size M+1)
 * @param Ai Input CSR column indices array (size nnz)
 * @param Az Input CSR non-zero values array (size nnz)
 * @param Bp Output CSC column pointer array (size N+1)
 * @param Bi Output CSC row indices array (size nnz)
 * @param Bz Output CSC non-zero values array (size nnz)
 * @param indexing The indexing base (0 for 0-based, 1 for 1-based indexing)
 *
 * @note The function assumes all input arrays are pre-allocated with correct
 * sizes
 * @note Output arrays Bp, Bi, Bz will be overwritten completely
 *
 * @warning Input and output arrays must not overlap (undefined behavior)
 * @warning No bounds checking is performed on input parameters
 */
template <typename T, typename index_t>
void csr_to_csc(size_t M, size_t N, const index_t* Ap, const index_t* Ai,
                const T* Az, index_t* Bp, index_t* Bi, T* Bz, size_t indexing) {
  // Adapted from SciPy csr_tocsc - generalized to take arbitrary indexing

  const auto nnz = Ap[M] - indexing;
  std::fill_n(Bp, N, 0);

  // Compute col counts
  for (index_t i = 0; i < nnz; ++i) Bp[Ai[i] - indexing]++;

  // Cumulative sum to get Bp
  for (index_t j = 0, csum = 0; j < N; ++j) {
    auto tmp = Bp[j];
    Bp[j] = csum;
    csum += tmp;
  }
  Bp[N] = nnz;

  // Reorder data
  for (index_t i = 0; i < M; ++i)
    for (index_t j = Ap[i] - indexing; j < Ap[i + 1] - indexing; ++j) {
      index_t col_idx = Ai[j] - indexing;
      index_t dest = Bp[col_idx];

      Bi[dest] = i;
      Bz[dest] = Az[j];
      Bp[col_idx]++;
    }

  for (index_t j = 0, last = 0; j < N; ++j) {
    std::swap(Bp[j], last);
  }

  // Fix indexing
  if (indexing) {
    for (index_t j = 0; j < (N + 1); ++j) Bp[j] += indexing;
    for (index_t i = 0; i < nnz; ++i) Bi[i] += indexing;
  }
}

}  // namespace detail

/**
 * @brief Constructor to create a CSR matrix from a COO matrix.
 *
 * This constructor converts a COO (Coordinate) sparse matrix to CSR (Compressed
 * Sparse Row) format. The conversion requires the input COO matrix to be sorted
 * by row indices for efficient processing.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 * @tparam Alloc The allocator type for memory management
 *
 * @param other The COO matrix to convert from
 *
 * @throws std::runtime_error if the COO matrix is not sorted by row indices
 * @throws std::runtime_error if non-zero indexing is used (temporary
 * limitation)
 *
 * @note The resulting CSR matrix will have the same indexing base as the input
 * COO matrix
 * @note Matrix dimensions and non-zero count are preserved exactly
 *
 * @warning Current implementation has a limitation with non-zero indexing and
 * will throw
 */
template <typename T, typename index_t, typename Alloc>
csr_matrix<T, index_t, Alloc>::csr_matrix(
    const coo_matrix<T, index_t, Alloc>& other)
    : csr_matrix(other.m(), other.n(), other.nnz(), other.indexing()) {
  if (not other.is_sorted_by_row_index()) {
    throw std::runtime_error(
        "COO -> CSR Conversion Requires COO To Be Row Sorted");
  }

  const auto& rowind_coo = other.rowind();
  const auto& colind_coo = other.colind();
  const auto& nzval_coo = other.nzval();

  // Compute rowptr

  if (indexing_) throw std::runtime_error("NONZERO INDEXING");
  for (size_type i = 0; i < nnz_; ++i) {
    rowptr_[rowind_coo[i] - indexing_ + 1]++;
  }
  for (size_type i = 0; i < m_; ++i) {
    rowptr_[i + 1] += rowptr_[i];
  }
  if (indexing_)
    for (size_type i = 0; i < m_ + 1; ++i) {
      rowptr_[i] += indexing_;
    }

  std::copy(colind_coo.begin(), colind_coo.end(), colind_.begin());
  std::copy(nzval_coo.begin(), nzval_coo.end(), nzval_.begin());
}

/**
 * @brief Constructor to create a CSC matrix from a COO matrix.
 *
 * This constructor converts a COO (Coordinate) sparse matrix to CSC (Compressed
 * Sparse Column) format. The conversion requires the input COO matrix to be
 * sorted by column indices for efficient processing.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 * @tparam Alloc The allocator type for memory management
 *
 * @param other The COO matrix to convert from
 *
 * @throws std::runtime_error if the COO matrix is not sorted by column indices
 *
 * @note The resulting CSC matrix will have the same indexing base as the input
 * COO matrix
 * @note Matrix dimensions and non-zero count are preserved exactly
 * @note Column pointer computation handles sparse columns (columns with no
 * non-zeros)
 */
template <typename T, typename index_t, typename Alloc>
csc_matrix<T, index_t, Alloc>::csc_matrix(
    const coo_matrix<T, index_t, Alloc>& other)
    : csc_matrix(other.m(), other.n(), other.nnz(), other.indexing()) {
  if (not other.is_sorted_by_col_index()) {
    throw std::runtime_error(
        "COO -> CSC Conversion Requires COO To Be Column Sorted");
  }

  const auto& rowind_coo = other.rowind();
  const auto& colind_coo = other.colind();
  const auto& nzval_coo = other.nzval();

  // Compute colptr
  colptr_.at(0) = other.indexing();
  auto cur_col = 0;
  for (size_t i = 0; i < nnz_; ++i)
    while (colind_coo[i] != (cur_col + indexing_)) {
      cur_col++;
      colptr_.at(cur_col) = i + indexing_;
    }
  colptr_.at(m_) = nnz_ + indexing_;

  std::copy(rowind_coo.begin(), rowind_coo.end(), rowind_.begin());
  std::copy(nzval_coo.begin(), nzval_coo.end(), nzval_.begin());
}

/**
 * @brief Constructor to create a CSR matrix from a CSC matrix.
 *
 * This constructor converts a CSC (Compressed Sparse Column) matrix to CSR
 * (Compressed Sparse Row) format using the transpose conversion algorithm. The
 * conversion efficiently reorders the sparse matrix data from column-major to
 * row-major storage.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 * @tparam Alloc The allocator type for memory management
 *
 * @param other The CSC matrix to convert from
 *
 * @note The resulting CSR matrix will have the same indexing base as the input
 * CSC matrix
 * @note Matrix dimensions and non-zero count are preserved exactly
 * @note Time complexity is O(nnz) for the conversion
 *
 * @see detail::csr_to_csc for the underlying conversion algorithm
 */
template <typename T, typename index_t, typename Alloc>
csr_matrix<T, index_t, Alloc>::csr_matrix(
    const csc_matrix<T, index_t, Alloc>& other)
    : csr_matrix(other.m(), other.n(), other.nnz(), other.indexing()) {
  detail::csr_to_csc(n_, m_, other.colptr().data(), other.rowind().data(),
                     other.nzval().data(), rowptr_.data(), colind_.data(),
                     nzval_.data(), indexing_);
}
}  // namespace sparsexx
