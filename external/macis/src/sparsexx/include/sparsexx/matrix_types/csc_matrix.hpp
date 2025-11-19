/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "type_fwd.hpp"

namespace sparsexx {

/**
 *  @brief A class to manipulate sparse matrices stored in CSC format
 *
 *  @tparam T       Field over which the elements of the sparse matrix are
 * defined
 *  @tparam index_t Integer type for the sparse indices
 *  @tparam Alloc   Allocator type for internal storage
 */
template <typename T, typename index_t, typename Alloc>
class csc_matrix {
 public:
  using value_type = T;  ///< Field over which the matrix elements are defined
  using index_type = index_t;    ///< Sparse index type
  using size_type = int64_t;     ///< Size type
  using allocator_type = Alloc;  ///< Allocator type

 protected:
  using alloc_traits = typename std::allocator_traits<Alloc>;

  template <typename U>
  using rebind_alloc = typename alloc_traits::template rebind_alloc<U>;

  template <typename U>
  using internal_storage = typename std::vector<U, rebind_alloc<U> >;

  size_type m_;         ///< Number of rows in the sparse matrix
  size_type n_;         ///< Number of cols in the sparse matrix
  size_type nnz_;       ///< Number of non-zeros in the sparse matrix
  size_type indexing_;  ///< Indexing base (0 or 1)

  internal_storage<T> nzval_;         ///< Storage of the non-zero values
  internal_storage<index_t> colptr_;  ///< Storage of the starting indices for
                                      ///< each col of the sparse matrix
  internal_storage<index_t> rowind_;  ///< Storage of the row indices

 public:
  csc_matrix() = default;

  /**
   *  @brief Construct a CSC matrix.
   *
   *  @param[in] m    Number of rows in the sparse matrix
   *  @param[in] n    Number of columns in the sparse matrix
   *  @param[in] nnz  Number of non-zeros in the sparse matrix
   *  @param[in] indexing Indexing base (default 1)
   */
  csc_matrix(size_type m, size_type n, size_type nnz, size_type indexing = 1)
      : m_(m),
        n_(n),
        nnz_(nnz),
        indexing_(indexing),
        nzval_(nnz),
        rowind_(nnz),
        colptr_(n + 1) {}

  csc_matrix(const csc_matrix& other) = default;
  csc_matrix(csc_matrix&& other) noexcept = default;

  csc_matrix& operator=(const csc_matrix&) = default;
  csc_matrix& operator=(csc_matrix&&) noexcept = default;

  // Convert between sparse formats
  csc_matrix(const coo_matrix<T, index_t, Alloc>& other);

  /**
   *  @brief Get the number of rows in the sparse matrix
   *
   *  @returns Number of rows in the sparse matrix
   */
  size_type m() const { return m_; };

  /**
   *  @brief Get the number of columns in the sparse matrix
   *
   *  @returns Number of columns in the sparse matrix
   */
  size_type n() const { return n_; };

  /**
   *  @brief Get the number of non-zeros in the sparse matrix
   *
   *  @returns Number of non-zeros in the sparse matrix
   */
  size_type nnz() const { return nnz_; };

  /**
   *  @brief Get the indexing base for the sparse matrix
   *
   *  @returns The indexing base for the sparse matrix
   */
  size_type indexing() const { return indexing_; }

  /**
   *  @brief Access the non-zero values of the sparse matrix in
   *  CSC format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in CSC format
   */
  auto& nzval() { return nzval_; };

  /**
   *  @brief Access the row indices of the sparse matrix in
   *  CSC format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  row indices of the sparse matrix in CSC format
   */
  auto& rowind() { return rowind_; };

  /**
   *  @brief Access the column pointer indirection array of the sparse matrix in
   *  CSC format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  column pointer indirection array of the sparse matrix in CSC format
   */
  auto& colptr() { return colptr_; };

  /**
   *  @brief Access the non-zero values of the sparse matrix in
   *  CSC format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in CSC format
   */
  const auto& nzval() const { return nzval_; };

  /**
   *  @brief Access the row indices of the sparse matrix in
   *  CSC format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  row indices of the sparse matrix in CSC format
   */

  const auto& rowind() const { return rowind_; };
  /**
   *  @brief Access the column pointer indirection array of the sparse matrix in
   *  CSC format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  column pointer indirection array of the sparse matrix in CSC format
   */
  const auto& colptr() const { return colptr_; };

};  // class csc_matrix

}  // namespace sparsexx

#include "conversions.hpp"
