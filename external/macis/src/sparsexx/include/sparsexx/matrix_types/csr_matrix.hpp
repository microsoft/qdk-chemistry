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

#ifdef SPARSEXX_ENABLE_CEREAL
#include <cereal/types/vector.hpp>
#endif /* SPARSEXX_ENABLE_CEREAL */

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

  inline void set_indexing(index_type idx) {
    if (idx == indexing_) return;
    for (auto& i : colind_) i -= (indexing_ - idx);
    for (auto& i : rowptr_) i -= (indexing_ - idx);
    indexing_ = idx;
  }

#ifdef SPARSEXX_ENABLE_CEREAL
  template <class Archive>
  void serialize(Archive& ar) {
    ar(m_, n_, nnz_, indexing_, rowptr_, colind_, nzval_);
  }
#endif /* SPARSEXX_ENABLE_CEREAL */

  bool operator==(const csr_matrix& other) const noexcept {
    return m_ == other.m_ and n_ == other.n_ and
           indexing_ == other.indexing_ and colind_ == other.colind_ and
           rowptr_ == other.rowptr_ and nzval_ == other.nzval_;
  }

  bool operator!=(const csr_matrix& other) const noexcept {
    return not((*this) == other);
  }

  size_type mem_footprint() const noexcept {
    return nzval_.capacity() * sizeof(T) +
           colind_.capacity() * sizeof(index_t) +
           rowptr_.capacity() * sizeof(index_t);
  }

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
