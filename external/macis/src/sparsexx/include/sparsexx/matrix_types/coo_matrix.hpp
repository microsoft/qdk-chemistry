/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <algorithm>
#include <stdexcept>

#include "type_fwd.hpp"

namespace sparsexx {

/**
 *  @brief A class to manipulate sparse matrices stored in coordiate (COO)
 * format
 *
 *  @tparam T       Field over which the elements of the sparse matrix are
 * defined
 *  @tparam index_t Integer type for the sparse indices
 *  @tparam Alloc   Allocator type for internal storage
 */
template <typename T, typename index_t, typename Alloc>
class coo_matrix {
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
  internal_storage<index_t> colind_;  ///< Storage of the column indices
  internal_storage<index_t> rowind_;  ///< Storage of the row indices

 public:
  coo_matrix() = default;

  /**
   *  @brief Construct a COO matrix.
   *
   *  @param[in] m    Number of rows in the sparse matrix
   *  @param[in] n    Number of columns in the sparse matrix
   *  @param[in] nnz  Number of non-zeros in the sparse matrix
   *  @param[in] indexing Indexing base (default 1)
   */
  coo_matrix(size_type m, size_type n, size_type nnz, size_type indexing = 1)
      : m_(m),
        n_(n),
        nnz_(nnz),
        indexing_(indexing),
        nzval_(nnz),
        colind_(nnz),
        rowind_(nnz) {}

  /**
   * @brief Construct a COO matrix from existing vectors.
   *
   * This constructor creates a COO matrix by moving existing vectors of row
   * indices, column indices, and non-zero values. This allows efficient
   * construction without copying large data arrays.
   *
   * @param m Number of rows in the sparse matrix
   * @param n Number of columns in the sparse matrix
   * @param colind Vector of column indices (will be moved)
   * @param rowind Vector of row indices (will be moved)
   * @param nzval Vector of non-zero values (will be moved)
   * @param indexing Indexing base (default: 1)
   *
   * @throws std::runtime_error if the vectors have incompatible sizes
   *
   * @note The nnz is automatically determined from the vector sizes
   * @note All input vectors must have the same size
   * @note The input vectors will be moved (emptied) after construction
   */
  coo_matrix(size_type m, size_type n, std::vector<index_type>&& colind,
             std::vector<index_type>&& rowind, std::vector<value_type>&& nzval,
             size_type indexing = 1)
      : m_(m),
        n_(n),
        nnz_(0),
        indexing_(indexing),
        nzval_(std::move(nzval)),
        colind_(std::move(colind)),
        rowind_(std::move(rowind)) {
    if (colind_.size() != rowind_.size())
      throw std::runtime_error("Incompatible Row/Col indices for COO");
    if (nzval_.size() != rowind_.size())
      throw std::runtime_error("Incompatible NZVAL for COO");
    nnz_ = nzval_.size();
  }

  coo_matrix(const coo_matrix& other) = default;
  coo_matrix(coo_matrix&& other) noexcept = default;

  coo_matrix& operator=(const coo_matrix&) = default;
  coo_matrix& operator=(coo_matrix&&) noexcept = default;

  coo_matrix(const csr_matrix<T, index_t, Alloc>& other);

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
   *  COO format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in COO format
   */
  auto& nzval() { return nzval_; };

  /**
   *  @brief Access the column indices of the sparse matrix in
   *  COO format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  column indices of the sparse matrix in COO format
   */
  auto& colind() { return colind_; };

  /**
   *  @brief Access the row indices of the sparse matrix in
   *  COO format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  row indices of the sparse matrix in COO format
   */
  auto& rowind() { return rowind_; };

  /**
   *  @brief Access the non-zero values of the sparse matrix in
   *  COO format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in COO format
   */
  const auto& nzval() const { return nzval_; };

  /**
   *  @brief Access the column indices of the sparse matrix in
   *  COO format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  column indices of the sparse matrix in COO format
   */
  const auto& colind() const { return colind_; };

  /**
   *  @brief Access the row indices of the sparse matrix in
   *  COO format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  row indices of the sparse matrix in COO format
   */
  const auto& rowind() const { return rowind_; };

  /**
   * @brief Automatically determines the indexing base from the matrix data.
   *
   * This function analyzes the row and column indices to determine whether the
   * matrix uses 0-based or 1-based indexing. It sets the indexing_ member
   * variable accordingly.
   *
   * @note Detection logic:
   *       - If any index is 0, assumes 0-based indexing
   *       - Otherwise, assumes 1-based indexing
   *       - Checks both row and column indices
   *
   * @warning This function modifies the indexing_ member variable
   * @warning Should only be called when the indexing base is unknown
   */
  void determine_indexing_from_adj() {
    auto eq_zero = [](const auto x) { return x == 0; };
    bool zero_based = std::any_of(rowind_.begin(), rowind_.end(), eq_zero) or
                      std::any_of(colind_.begin(), colind_.end(), eq_zero);
    indexing_ = !zero_based;
  }

  void sort_by_row_index();
  void sort_by_col_index();

  /**
   * @brief Checks if the matrix is sorted by row indices.
   *
   * This function determines whether the COO matrix entries are currently
   * sorted by row index in ascending order. This is useful for determining if
   * the matrix is suitable for efficient CSR conversion.
   *
   * @return true if entries are sorted by row index, false otherwise
   *
   * @note Only checks row index ordering, not secondary column index ordering
   * @note This is a simple check using std::is_sorted on the row index array
   *
   * @see sort_by_row_index() to sort the matrix by row indices
   */
  bool is_sorted_by_row_index() const {
    return std::is_sorted(rowind_.begin(), rowind_.end());
  }

  /**
   * @brief Checks if the matrix is sorted by column indices.
   *
   * This function determines whether the COO matrix entries are currently
   * sorted by column index in ascending order. This is useful for determining
   * if the matrix is suitable for efficient CSC conversion.
   *
   * @return true if entries are sorted by column index, false otherwise
   *
   * @note Only checks column index ordering, not secondary row index ordering
   * @note This is a simple check using std::is_sorted on the column index array
   *
   * @see sort_by_col_index() to sort the matrix by column indices
   */
  bool is_sorted_by_col_index() const {
    return std::is_sorted(colind_.begin(), colind_.end());
  }

  void expand_from_triangle();

  /**
   * @brief Insert a new triplet (row, column, value) into the COO matrix.
   *
   * This function adds a new non-zero element to the COO matrix by appending
   * the triplet to the internal storage arrays. The matrix size (nnz) is
   * automatically incremented.
   *
   * @tparam Check Template parameter for bounds checking (currently not
   * implemented)
   *
   * @param i Row index of the element to insert
   * @param j Column index of the element to insert
   * @param v Value of the element to insert
   *
   * @note The function is marked noexcept and performs no bounds checking
   * @note The Check template parameter is not yet implemented (will trigger
   * static_assert)
   * @note Elements are appended in insertion order, no sorting is maintained
   * @note nnz_ counter is automatically incremented
   *
   * @warning No duplicate checking is performed
   * @warning No bounds validation against matrix dimensions is performed
   * @warning Check template parameter must be false (implementation limitation)
   */
  template <bool Check>
  inline void insert(index_type i, index_type j, value_type v) noexcept {
    static_assert(not Check, "insert check NYI");
    rowind_.emplace_back(i);
    colind_.emplace_back(j);
    nzval_.emplace_back(v);
    nnz_++;
  }

};  // coo_matrix

}  // namespace sparsexx

#include "coo_conversions.hpp"
#include "coo_matrix_ops.hpp"
