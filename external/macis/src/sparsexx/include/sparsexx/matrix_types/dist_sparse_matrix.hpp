/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/mpi.hpp>
#include <sparsexx/util/submatrix.hpp>
#include <sstream>

namespace sparsexx {

/**
 * @brief A distributed sparse matrix class that partitions matrices across MPI
 * processes.
 *
 * This class represents a sparse matrix distributed across multiple MPI
 * processes using row-based partitioning. Each process owns a contiguous block
 * of rows, with the matrix data split into diagonal and off-diagonal tiles for
 * efficient parallel operations.
 *
 * @tparam SpMatType The underlying sparse matrix type (e.g., csr_matrix,
 * csc_matrix)
 *
 * @note Matrix structure:
 *       - Diagonal tile: Contains elements where row and column belong to local
 * partition
 *       - Off-diagonal tile: Contains elements where row is local but column is
 * remote
 *       - Row partitioning: Each process owns a contiguous range of rows
 */
template <typename SpMatType>
class dist_sparse_matrix {
 public:
  using value_type = detail::value_type_t<SpMatType>;
  using index_type = detail::index_type_t<SpMatType>;
  using size_type = detail::size_type_t<SpMatType>;
  using tile_type = SpMatType;
  using extent_type = std::pair<index_type, index_type>;

 protected:
  MPI_Comm comm_;
  int comm_size_;
  int comm_rank_;

  size_type global_m_;
  size_type global_n_;

  std::shared_ptr<tile_type> diagonal_tile_ = nullptr;
  std::shared_ptr<tile_type> off_diagonal_tile_ = nullptr;

  std::vector<extent_type> dist_row_extents_;

 public:
  constexpr dist_sparse_matrix() noexcept = default;
  dist_sparse_matrix(dist_sparse_matrix&&) noexcept = default;

  /**
   * @brief Constructs a distributed sparse matrix with default row
   * partitioning.
   *
   * This constructor creates a distributed sparse matrix by evenly dividing the
   * rows among all MPI processes. The last process receives any remainder rows.
   *
   * @param c MPI communicator for the distributed matrix
   * @param M Number of rows in the global matrix
   * @param N Number of columns in the global matrix
   *
   * @note Row distribution:
   *       - Rows are divided as evenly as possible among processes
   *       - Process i gets rows [i*nrow_per_rank, (i+1)*nrow_per_rank)
   *       - Last process gets any remainder rows: M % comm_size
   */
  // Ctor with default row partitioning
  dist_sparse_matrix(MPI_Comm c, size_type M, size_type N)
      : comm_(c), global_m_(M), global_n_(N) {
    comm_size_ = detail::get_mpi_size(comm_);
    comm_rank_ = detail::get_mpi_rank(comm_);
    const auto nrow_per_rank = M / comm_size_;

    dist_row_extents_.resize(comm_size_);
    dist_row_extents_[0] = {0, nrow_per_rank};
    for (int i = 1; i < comm_size_; ++i) {
      dist_row_extents_[i] = {dist_row_extents_[i - 1].second,
                              dist_row_extents_[i - 1].second + nrow_per_rank};
    }
    dist_row_extents_.back().second +=
        M % comm_size_;  // Last rank gets carry-over
  }

  /**
   * @brief Constructs a distributed sparse matrix with custom row partitioning.
   *
   * This constructor allows for custom row distribution across MPI processes,
   * enabling load balancing based on matrix sparsity patterns or other
   * criteria.
   *
   * @param c MPI communicator for the distributed matrix
   * @param M Number of rows in the global matrix
   * @param N Number of columns in the global matrix
   * @param row_tiles Vector of row extent pairs defining the partitioning
   *
   * @throws std::runtime_error if row_tiles.size() != comm_size
   * @throws std::runtime_error if row tiles don't cover the full matrix range
   * [0, M)
   * @throws std::runtime_error if row tiles are not sorted or contiguous
   *
   * @note Row tile requirements:
   *       - Must have exactly comm_size entries
   *       - First tile must start at 0, last tile must end at M
   *       - Tiles must be contiguous and non-overlapping
   *       - Each tile defines a half-open interval [start, end)
   */
  // Ctor with custom row partitioning
  dist_sparse_matrix(MPI_Comm c, size_t M, size_t N,
                     const std::vector<extent_type>& row_tiles)
      : comm_(c), global_m_(M), global_n_(N), dist_row_extents_(row_tiles) {
    comm_size_ = detail::get_mpi_size(comm_);
    comm_rank_ = detail::get_mpi_rank(comm_);

    if (dist_row_extents_.size() != comm_size_)
      throw std::runtime_error("Incorrect Row Tile Size");

    if (dist_row_extents_[0].first != 0 or dist_row_extents_.back().second != M)
      throw std::runtime_error("Invalid Row Tile Bounds");

    for (auto [i, j] : dist_row_extents_) {
      if (i > j) throw std::runtime_error("Row Tiles Must Be Sorted");
    }
    for (auto i = 0; i < comm_size_ - 1; ++i) {
      if (dist_row_extents_[i].second != dist_row_extents_[i + 1].first)
        throw std::runtime_error("Row Tiles Must Be Contiguous");
    }
  }

  /**
   * @brief Copy constructor for distributed sparse matrix.
   *
   * Creates a copy of the distributed matrix including all tile data.
   * Both diagonal and off-diagonal tiles are copied if they exist.
   *
   * @param other The distributed matrix to copy from
   */
  dist_sparse_matrix(const dist_sparse_matrix& other)
      : dist_sparse_matrix(other.comm_, other.global_m_, other.global_n_) {
    if (other.diagonal_tile_) set_diagonal_tile(other.diagonal_tile());
    if (other.off_diagonal_tile_)
      set_off_diagonal_tile(other.off_diagonal_tile());
  }

  /**
   * @brief Constructs a distributed matrix from a global sparse matrix with
   * default partitioning.
   *
   * This constructor takes a global sparse matrix and distributes it across MPI
   * processes using the default row partitioning. The matrix is split into
   * diagonal and off-diagonal tiles based on the row ownership.
   *
   * @param c MPI communicator for the distributed matrix
   * @param A Global sparse matrix to distribute
   *
   * @note Matrix extraction:
   *       - Diagonal tile: Contains A[local_rows, local_rows]
   *       - Off-diagonal tile: Contains A[local_rows, remote_columns]
   *       - Both tiles are converted to 0-based indexing
   */
  dist_sparse_matrix(MPI_Comm c, const SpMatType& A)
      : dist_sparse_matrix(c, A.m(), A.n()) {
    auto [local_row_st, local_row_en] = dist_row_extents_[comm_rank_];
    auto local_lo =
        std::make_pair<int64_t, int64_t>(local_row_st, local_row_st);
    auto local_up =
        std::make_pair<int64_t, int64_t>(local_row_en, local_row_en);
    diagonal_tile_ =
        std::make_shared<tile_type>(extract_submatrix(A, local_lo, local_up));
    off_diagonal_tile_ = std::make_shared<tile_type>(
        extract_submatrix_inclrow_exclcol(A, local_lo, local_up));
    diagonal_tile_->set_indexing(0);
    off_diagonal_tile_->set_indexing(0);
  }

  dist_sparse_matrix(MPI_Comm c, const SpMatType& A,
                     const std::vector<extent_type>& row_tiles)
      : dist_sparse_matrix(c, A.m(), A.n(), row_tiles) {
    auto [local_row_st, local_row_en] = dist_row_extents_[comm_rank_];
    auto local_lo =
        std::make_pair<int64_t, int64_t>(local_row_st, local_row_st);
    auto local_up =
        std::make_pair<int64_t, int64_t>(local_row_en, local_row_en);
    diagonal_tile_ =
        std::make_shared<tile_type>(extract_submatrix(A, local_lo, local_up));
    off_diagonal_tile_ = std::make_shared<tile_type>(
        extract_submatrix_inclrow_exclcol(A, local_lo, local_up));
    diagonal_tile_->set_indexing(0);
    off_diagonal_tile_->set_indexing(0);
  }

  /**
   * @brief Gets the global number of rows.
   * @return Global number of rows in the distributed matrix
   */
  inline auto m() const { return global_m_; }

  /**
   * @brief Gets the global number of columns.
   * @return Global number of columns in the distributed matrix
   */
  inline auto n() const { return global_n_; }

  /**
   * @brief Gets the MPI communicator.
   * @return The MPI communicator used by this distributed matrix
   */
  inline MPI_Comm comm() const { return comm_; }

  /**
   * @brief Gets the row bounds for a specific MPI rank.
   * @param rank The MPI rank to query
   * @return Pair (start, end) defining the half-open row interval for the rank
   */
  inline auto row_bounds(int rank) const { return dist_row_extents_[rank]; }

  /**
   * @brief Gets the number of rows owned by a specific rank.
   * @param rank The MPI rank to query
   * @return Number of rows owned by the specified rank
   */
  inline size_type row_extent(int rank) const {
    return dist_row_extents_[rank].second - dist_row_extents_[rank].first;
  }

  /**
   * @brief Gets the number of rows owned by the local process.
   * @return Number of rows owned by the current MPI process
   */
  inline size_type local_row_extent() const { return row_extent(comm_rank_); }

  /**
   * @brief Gets the starting row index for the local process.
   * @return First row index owned by the current MPI process
   */
  inline size_type local_row_start() const {
    return dist_row_extents_[comm_rank_].first;
  }

  /**
   * @brief Gets the total number of non-zero elements in local tiles.
   *
   * This function returns the sum of non-zero elements in both the diagonal
   * and off-diagonal tiles owned by the current process.
   *
   * @return Total number of local non-zero elements
   *
   * @note This is a local operation that does not communicate with other
   * processes
   * @note To get the global nnz, a reduction across all processes would be
   * needed
   */
  inline size_type nnz() const noexcept {
    size_t _nnz = 0;
    if (diagonal_tile_) _nnz += diagonal_tile_->nnz();
    if (off_diagonal_tile_) _nnz += off_diagonal_tile_->nnz();
    return _nnz;
  }

  /**
   * @brief Gets the memory footprint of local tiles in bytes.
   *
   * This function calculates the total memory usage of both diagonal and
   * off-diagonal tiles owned by the current process.
   *
   * @return Total memory footprint in bytes for local tiles
   *
   * @note This is a local operation that does not include memory from other
   * processes
   * @note Memory calculation is delegated to the underlying tile types
   */
  inline size_type mem_footprint() const noexcept {
    size_type _mf = 0;
    if (diagonal_tile_) _mf += diagonal_tile_->mem_footprint();
    if (off_diagonal_tile_) _mf += off_diagonal_tile_->mem_footprint();
    return _mf;
  }

  /**
   * @brief Gets a shared pointer to the diagonal tile.
   * @return Shared pointer to the diagonal tile (may be nullptr)
   */
  auto diagonal_tile_ptr() { return diagonal_tile_; }

  /**
   * @brief Gets a const shared pointer to the diagonal tile.
   * @return Const shared pointer to the diagonal tile (may be nullptr)
   */
  const auto diagonal_tile_ptr() const { return diagonal_tile_; }

  /**
   * @brief Gets a shared pointer to the off-diagonal tile.
   * @return Shared pointer to the off-diagonal tile (may be nullptr)
   */
  auto off_diagonal_tile_ptr() { return off_diagonal_tile_; }

  /**
   * @brief Gets a const shared pointer to the off-diagonal tile.
   * @return Const shared pointer to the off-diagonal tile (may be nullptr)
   */
  const auto off_diagonal_tile_ptr() const { return off_diagonal_tile_; }

  /**
   * @brief Gets a const reference to the diagonal tile.
   * @return Const reference to the diagonal tile
   * @warning Undefined behavior if diagonal_tile_ is nullptr
   */
  const auto& diagonal_tile() const { return *diagonal_tile_; }

  /**
   * @brief Gets a const reference to the off-diagonal tile.
   * @return Const reference to the off-diagonal tile
   * @warning Undefined behavior if off_diagonal_tile_ is nullptr
   */
  const auto& off_diagonal_tile() const { return *off_diagonal_tile_; }

  /**
   * @brief Sets the diagonal tile by copying the provided matrix.
   * @param A The sparse matrix to copy as the diagonal tile
   */
  void set_diagonal_tile(const SpMatType& A) {
    diagonal_tile_ = std::make_shared<tile_type>(A);
  }

  /**
   * @brief Sets the off-diagonal tile by copying the provided matrix.
   * @param A The sparse matrix to copy as the off-diagonal tile
   */
  void set_off_diagonal_tile(const SpMatType& A) {
    off_diagonal_tile_ = std::make_shared<tile_type>(A);
  }

  /**
   * @brief Sets the diagonal tile by moving the provided matrix.
   * @param A The sparse matrix to move as the diagonal tile
   */
  void set_diagonal_tile(SpMatType&& A) {
    diagonal_tile_ = std::make_shared<tile_type>(std::move(A));
  }

  /**
   * @brief Sets the off-diagonal tile by moving the provided matrix.
   * @param A The sparse matrix to move as the off-diagonal tile
   */
  void set_off_diagonal_tile(SpMatType&& A) {
    off_diagonal_tile_ = std::make_shared<tile_type>(std::move(A));
  }
};  // class dist_sparse_matrix

/**
 * @brief Type trait to detect distributed sparse matrix types.
 *
 * This struct provides a compile-time mechanism to determine if a type
 * is a distributed sparse matrix. The default case is false.
 *
 * @tparam SpMatType The type to test
 */
template <typename SpMatType>
struct is_dist_sparse_matrix : public std::false_type {};

/**
 * @brief Specialization of is_dist_sparse_matrix for dist_sparse_matrix types.
 *
 * This specialization returns true for any instantiation of dist_sparse_matrix.
 *
 * @tparam SpMatType The underlying sparse matrix type
 */
template <typename SpMatType>
struct is_dist_sparse_matrix<dist_sparse_matrix<SpMatType>>
    : public std::true_type {};

/**
 * @brief Convenience variable template for is_dist_sparse_matrix.
 *
 * This provides a more convenient way to check if a type is a distributed
 * sparse matrix using the _v suffix convention.
 *
 * @tparam SpMatType The type to test
 */
template <typename SpMatType>
inline static constexpr bool is_dist_sparse_matrix_v =
    is_dist_sparse_matrix<SpMatType>::value;

}  // namespace sparsexx
