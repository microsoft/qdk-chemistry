/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cmath>

#include "type_fwd.hpp"

namespace sparsexx {

/**
 *  @brief A class to manipulate sparse matrices stored in CSR format
 *
 *  @tparam T       Field over which the elements of the sparse matrix are
 * defined
 *  @tparam index_t Integer type for the sparse indices
 *  @tparam Alloc   Allocator type for internal storage
 */
template <typename T, typename index_t, typename Alloc>
class csr_matrix {
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
  internal_storage<index_t> rowptr_;
  ///< Storage of the starting indices for each row of the sparse matrix

 public:
  csr_matrix() = default;

  /**
   *  @brief Construct a CSR matrix.
   *
   *  @param[in] m    Number of rows in the sparse matrix
   *  @param[in] n    Number of columns in the sparse matrix
   *  @param[in] nnz  Number of non-zeros in the sparse matrix
   *  @param[in] indexing Indexing base (default 1)
   */
  csr_matrix(size_type m, size_type n, size_type nnz, size_type indexing = 1)
      : m_(m),
        n_(n),
        nnz_(nnz),
        indexing_(indexing),
        nzval_(nnz),
        colind_(nnz),
        rowptr_(m + 1) {}

  /**
   * @brief Construct a CSR matrix from existing vectors.
   *
   * This constructor creates a CSR matrix by moving existing vectors of row
   * pointers, column indices, and non-zero values. This allows efficient
   * construction without copying large data arrays.
   *
   * @param m Number of rows in the sparse matrix
   * @param n Number of columns in the sparse matrix
   * @param rowptr Vector of row pointers (will be moved)
   * @param colind Vector of column indices (will be moved)
   * @param nzval Vector of non-zero values (will be moved)
   *
   * @note The nnz is automatically determined from the nzval vector size
   * @note The indexing base is determined from rowptr[0]
   * @note The input vectors will be moved (emptied) after construction
   * @note This constructor assumes double for value type
   */
  csr_matrix(size_type m, size_type n, std::vector<index_t>&& rowptr,
             std::vector<index_t>&& colind, std::vector<double>&& nzval)
      : m_(m),
        n_(n),
        nnz_(nzval.size()),
        indexing_(rowptr[0]),
        nzval_(std::move(nzval)),
        colind_(std::move(colind)),
        rowptr_(std::move(rowptr)) {}

  csr_matrix(const csr_matrix& other) = default;
  csr_matrix(csr_matrix&& other) noexcept = default;

  csr_matrix& operator=(const csr_matrix&) = default;
  csr_matrix& operator=(csr_matrix&&) noexcept = default;

  // Convert between sparse formats
  csr_matrix(const coo_matrix<T, index_t, Alloc>& other);
  csr_matrix(const csc_matrix<T, index_t, Alloc>& other);
  // csr_matrix& operator=( const coo_matrix<T, index_t, Alloc>& other );

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
   *  CSR format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in CSR format
   */
  auto& nzval() { return nzval_; };

  /**
   *  @brief Access the column indices of the sparse matrix in
   *  CSR format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  column indices of the sparse matrix in CSR format
   */
  auto& colind() { return colind_; };

  /**
   *  @brief Access the row pointer indirection array of the sparse matrix in
   *  CSR format
   *
   *  Non-const variant
   *
   *  @returns A non-const reference to the internal storage of the
   *  row pointer indirection array of the sparse matrix in CSR format
   */
  auto& rowptr() { return rowptr_; };

  /**
   *  @brief Access the non-zero values of the sparse matrix in
   *  CSR format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  non-zero elements of the sparse matrix in CSR format
   */
  const auto& nzval() const { return nzval_; };

  /**
   *  @brief Access the column indices of the sparse matrix in
   *  CSR format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  column indices of the sparse matrix in CSR format
   */

  const auto& colind() const { return colind_; };
  /**
   *  @brief Access the row pointer indirection array of the sparse matrix in
   *  CSR format
   *
   *  Const variant
   *
   *  @returns A const reference to the internal storage of the
   *  row pointer indirection array of the sparse matrix in CSR format
   */
  const auto& rowptr() const { return rowptr_; };

  /**
   * @brief Changes the indexing base of the sparse matrix.
   *
   * This function converts the matrix between 0-based and 1-based indexing
   * by adjusting all indices in the column indices and row pointer arrays.
   * If the target indexing is the same as current, no operation is performed.
   *
   * @param idx The new indexing base (0 for 0-based, 1 for 1-based)
   *
   * @note Conversion process:
   *       - Adjusts all column indices by (indexing_ - idx)
   *       - Adjusts all row pointer values by (indexing_ - idx)
   *       - Updates the indexing_ member variable
   *
   * @note This is an in-place operation that modifies the matrix data
   * @note No validation is performed on the input indexing value
   *
   * @warning Calling this function  with invalid values may corrupt the matrix
   */
  inline void set_indexing(index_type idx) {
    if (idx == indexing_) return;
    for (auto& i : colind_) i -= (indexing_ - idx);
    for (auto& i : rowptr_) i -= (indexing_ - idx);
    indexing_ = idx;
  }

  /**
   * @brief Equality comparison operator for CSR matrices.
   *
   * This function compares two CSR matrices for exact equality by checking
   * all matrix properties and data arrays.
   *
   * @param other The CSR matrix to compare against
   *
   * @return true if matrices are identical, false otherwise
   *
   * @note This is an exact comparison; no tolerance is used for floating-point
   * values
   */
  bool operator==(const csr_matrix& other) const noexcept {
    return m_ == other.m_ and n_ == other.n_ and
           indexing_ == other.indexing_ and colind_ == other.colind_ and
           rowptr_ == other.rowptr_ and nzval_ == other.nzval_;
  }

  /**
   * @brief Inequality comparison operator for CSR matrices.
   *
   * This function compares two CSR matrices for inequality by negating
   * the equality comparison.
   *
   * @param other The CSR matrix to compare against
   *
   * @return true if matrices are different, false if identical
   *
   * @note This function delegates to operator== and negates the result
   */
  bool operator!=(const csr_matrix& other) const noexcept {
    return not((*this) == other);
  }

  /**
   * @brief Calculates the memory footprint of the CSR matrix in bytes.
   *
   * This function computes the total memory usage of the matrix by summing
   * the capacity of all internal storage vectors multiplied by their element
   * sizes.
   *
   * @return Total memory footprint in bytes
   *
   * @note Uses capacity() rather than size() to account for potential
   * over-allocation
   * @note Does not include the size of the matrix object itself, only data
   * arrays
   */
  size_type mem_footprint() const noexcept {
    return nzval_.capacity() * sizeof(T) +
           colind_.capacity() * sizeof(index_t) +
           rowptr_.capacity() * sizeof(index_t);
  }

  /**
   * @brief Shrinks the storage of all internal vectors to fit their actual
   * size.
   *
   * This function calls shrink_to_fit() on all internal storage vectors to
   * release any excess allocated memory and minimize memory usage.
   *
   * @note This operation may trigger memory reallocation and copying
   * @note Useful for reducing memory footprint after matrix
   * construction/modification
   * @note No guarantee that memory will actually be released (implementation
   * dependent)
   */
  void shrink_storage_to_fit() {
    nzval_.shrink_to_fit();
    colind_.shrink_to_fit();
    rowptr_.shrink_to_fit();
  }

  /**
   * @brief Threshold the sparse matrix elements in-place, dropping all
   * values below threshold
   *
   * This method operates in three phases:
   * 1. Count the surviving elements per row in parallel
   * 2. Compute new row pointers (sequential prefix sum)
   * 3. Compact elements in-place by moving surviving elements forward
   *
   * @param threshold The absolute threshold value below which elements are
   * removed
   */
  void threshold_parallel(double threshold) {
    int num_rows = rowptr_.size() - 1;
    std::vector<index_t> new_row_counts(num_rows);

    // Phase 1: Count surviving elements per row in parallel
#pragma omp parallel for
    for (int row = 0; row < num_rows; row++) {
      auto count = 0;
      auto row_start = rowptr_[row];
      auto row_end = rowptr_[row + 1];

      for (auto pos = row_start; pos < row_end; pos++) {
        if (std::abs(nzval_[pos]) > threshold) {
          count++;
        }
      }
      new_row_counts[row] = count;
    }

    // Phase 2: Compute new row pointers (sequential prefix sum)
    std::vector<index_t> new_row_ptr(num_rows + 1, 0);
    for (int row = 0; row < num_rows; row++) {
      new_row_ptr[row + 1] = new_row_ptr[row] + new_row_counts[row];
    }

    // Phase 3: Compact elements in-place
    // Process rows sequentially to avoid conflicts when moving elements
    index_t write_pos = 0;
    for (int row = 0; row < num_rows; row++) {
      auto old_start = rowptr_[row];
      auto old_end = rowptr_[row + 1];

      for (auto read_pos = old_start; read_pos < old_end; read_pos++) {
        if (std::abs(nzval_[read_pos]) > threshold) {
          if (write_pos != read_pos) {
            nzval_[write_pos] = nzval_[read_pos];
            colind_[write_pos] = colind_[read_pos];
          }
          write_pos++;
        }
      }
    }

    // Update row pointers and resize arrays
    rowptr_ = std::move(new_row_ptr);
    auto new_nnz = rowptr_[num_rows];

    // Resize vectors to new size
    nzval_.resize(new_nnz);
    colind_.resize(new_nnz);

    // Update the number of non-zeros
    nnz_ = new_nnz;
  }
};  // class csr_matrix

}  // namespace sparsexx

#include "conversions.hpp"
