/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <iostream>
#include <sparsexx/sparsexx_config.hpp>

#include "coo_matrix.hpp"

#ifdef SPARSEXX_ENABLE_RANGES_V3
#include <range/v3/all.hpp>
#endif /* SPARSEXX_ENABLE_RANGES_V3 */
#include <algorithm>
#include <numeric>

namespace sparsexx {

/**
 * @brief Sorts the COO matrix entries by row index, with column index as
 * secondary sort key.
 *
 * This function reorders the COO matrix triplets (row, column, value) such that
 * they are sorted first by row index, then by column index within each row.
 * This ordering is particularly useful for efficient conversion to CSR format.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 * @tparam Alloc The allocator type for memory management
 *
 * @note Sorting algorithm:
 *       - If SPARSEXX_ENABLE_RANGES_V3: Uses ranges::sort on zipped triplets
 *       - Otherwise: Uses indirect sorting with index permutation for cache
 * efficiency
 *
 * @note Sort order:
 *       - Primary key: row index (ascending)
 *       - Secondary key: column index (ascending)
 *       - Lexicographic ordering: (i1, j1) < (i2, j2) if i1 < i2 or (i1 == i2
 * and j1 < j2)
 *
 * @note Performance:
 *       - Time complexity: O(nnz * log(nnz))
 *       - Space complexity: O(nnz) for indirect sort, O(1) for ranges version
 *
 * @see sort_by_col_index() for column-major ordering
 */
template <typename T, typename index_t, typename Alloc>
void coo_matrix<T, index_t, Alloc>::sort_by_row_index() {
#ifdef SPARSEXX_ENABLE_RANGES_V3
  auto coo_zip = ranges::views::zip(rowind_, colind_, nzval_);

  // Sort lex by row index
  using coo_el = std::tuple<index_type, index_type, value_type>;
  ranges::sort(coo_zip, [](const coo_el& el1, const coo_el& el2) {
    const auto i1 = std::get<0>(el1);
    const auto i2 = std::get<0>(el2);
    const auto j1 = std::get<1>(el1);
    const auto j2 = std::get<1>(el2);

    if (i1 < i2)
      return true;
    else if (i1 > i2)
      return false;
    else
      return j1 < j2;
  });
#else

  std::vector<index_t> indx(nnz_);
  std::iota(indx.begin(), indx.end(), 0);

  std::sort(indx.begin(), indx.end(), [&](auto i, auto j) {
    if (rowind_[i] < rowind_[j])
      return true;
    else if (rowind_[j] < rowind_[i])
      return false;
    else
      return colind_[i] < colind_[j];
  });

  std::vector<index_t> new_rowind_(nnz_), new_colind_(nnz_);
  std::vector<T> new_nzval_(nnz_);

  for (int64_t i = 0; i < nnz_; ++i) {
    new_rowind_[i] = rowind_[indx[i]];
    new_colind_[i] = colind_[indx[i]];
    new_nzval_[i] = nzval_[indx[i]];
  }

  rowind_ = std::move(new_rowind_);
  colind_ = std::move(new_colind_);
  nzval_ = std::move(new_nzval_);

#endif /* SPARSEXX_ENABLE_RANGES_V3 */
}

/**
 * @brief Expands a triangular matrix to its full symmetric form.
 *
 * This function takes a matrix stored in triangular form (either upper or lower
 * triangle) and expands it to the full symmetric matrix by mirroring the
 * off-diagonal elements. Diagonal elements are preserved as-is.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 * @tparam Alloc The allocator type for memory management
 *
 * @note Matrix type detection:
 *       - Lower triangle: All entries satisfy row_index <= column_index
 *       - Upper triangle: All entries satisfy row_index >= column_index
 *       - Diagonal: Both conditions true (only diagonal entries)
 *       - Full matrix: Neither condition true (already expanded)
 *
 * @note Expansion process:
 *       1. Detects if matrix is lower triangular, upper triangular, or already
 * full
 *       2. If diagonal-only or already full, returns without modification
 *       3. For triangular matrices, adds mirrored off-diagonal entries
 *       4. Diagonal entries are not duplicated
 *
 * @note Memory requirements:
 *       - New nnz = 2 * old_nnz - n (for triangular input)
 *       - Arrays are resized to accommodate new entries
 *
 * @warning Function includes debug output that may affect performance
 * @warning No validation of matrix symmetry is performed
 */
template <typename T, typename index_t, typename Alloc>
void coo_matrix<T, index_t, Alloc>::expand_from_triangle() {
#ifdef SPARSEXX_ENABLE_RANGES_V3

  auto idx_zip = ranges::views::zip(rowind_, colind_);

  auto lt_check = [](const std::tuple<index_type, index_type>& p) {
    return std::get<0>(p) <= std::get<1>(p);
  };
  auto ut_check = [](const std::tuple<index_type, index_type>& p) {
    return std::get<0>(p) >= std::get<1>(p);
  };

  bool lower_triangle = ranges::all_of(idx_zip, lt_check);
  bool upper_triangle = ranges::all_of(idx_zip, ut_check);

#else

  bool upper_triangle, lower_triangle;
  {
    std::vector<index_t> indx(nnz_);
    std::iota(indx.begin(), indx.end(), 0);
    auto lt_check = [&](auto i) { return rowind_[i] <= colind_[i]; };
    auto ut_check = [&](auto i) { return rowind_[i] >= colind_[i]; };

    lower_triangle = std::all_of(indx.begin(), indx.end(), lt_check);
    upper_triangle = std::all_of(indx.begin(), indx.end(), ut_check);
  }

#endif /* SPARSEXX_ENABLE_RANGES_V3 */
  bool diagonal = lower_triangle and upper_triangle;
  bool full_matrix = (not lower_triangle) and (not upper_triangle);
  if (diagonal or full_matrix) return;

  // std::cout << "Performing Expansion..." << std::endl;
  size_t new_nnz = 2 * nnz_ - n_;
  rowind_.reserve(new_nnz);
  colind_.reserve(new_nnz);
  nzval_.reserve(new_nnz);

  for (size_t i = 0; i < nnz_; ++i)
    if (rowind_[i] != colind_[i]) {
      rowind_.emplace_back(colind_[i]);
      colind_.emplace_back(rowind_[i]);
      nzval_.emplace_back(nzval_[i]);
    }

  assert(rowind_.size() == new_nnz);
  assert(colind_.size() == new_nnz);
  assert(nzval_.size() == new_nnz);

  nnz_ = new_nnz;
}

/**
 * @brief Sorts the COO matrix entries by column index, with row index as
 * secondary sort key.
 *
 * This function reorders the COO matrix triplets (row, column, value) such that
 * they are sorted first by column index, then by row index within each column.
 * This ordering is particularly useful for efficient conversion to CSC format.
 *
 * @tparam T The value type for matrix elements
 * @tparam index_t The index type for matrix indices
 * @tparam Alloc The allocator type for memory management
 *
 * @note Sorting algorithm:
 *       - If SPARSEXX_ENABLE_RANGES_V3: Uses ranges::sort on zipped triplets
 *       - Otherwise: Uses indirect sorting with index permutation for cache
 * efficiency
 *
 * @note Sort order:
 *       - Primary key: column index (ascending)
 *       - Secondary key: row index (ascending)
 *       - Lexicographic ordering: (i1, j1) < (i2, j2) if j1 < j2 or (j1 == j2
 * and i1 < i2)
 *
 * @note Performance:
 *       - Time complexity: O(nnz * log(nnz))
 *       - Space complexity: O(nnz) for indirect sort, O(1) for ranges version
 *
 * @see sort_by_row_index() for row-major ordering
 */
template <typename T, typename index_t, typename Alloc>
void coo_matrix<T, index_t, Alloc>::sort_by_col_index() {
#ifdef SPARSEXX_ENABLE_RANGES_V3
  auto coo_zip = ranges::views::zip(rowind_, colind_, nzval_);

  // Sort lex by row index
  using coo_el = std::tuple<index_type, index_type, value_type>;
  ranges::sort(coo_zip, [](const coo_el& el1, const coo_el& el2) {
    const auto i1 = std::get<0>(el1);
    const auto i2 = std::get<0>(el2);
    const auto j1 = std::get<1>(el1);
    const auto j2 = std::get<1>(el2);

    if (j1 < j2)
      return true;
    else if (j1 > j2)
      return false;
    else
      return i1 < i2;
  });
#else

  std::vector<index_t> indx(nnz_);
  std::iota(indx.begin(), indx.end(), 0);

  std::sort(indx.begin(), indx.end(), [&](auto i, auto j) {
    if (colind_[i] < colind_[j])
      return true;
    else if (colind_[j] < colind_[i])
      return false;
    else
      return rowind_[i] < rowind_[j];
  });

  std::vector<index_t> new_rowind_(nnz_), new_colind_(nnz_);
  std::vector<T> new_nzval_(nnz_);

  for (int64_t i = 0; i < nnz_; ++i) {
    new_rowind_[i] = rowind_[indx[i]];
    new_colind_[i] = colind_[indx[i]];
    new_nzval_[i] = nzval_[indx[i]];
  }

  rowind_ = std::move(new_rowind_);
  colind_ = std::move(new_colind_);
  nzval_ = std::move(new_nzval_);

#endif /* SPARSEXX_ENABLE_RANGES_V3 */
}

}  // namespace sparsexx
