/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>

namespace sparsexx {

namespace detail {

/**
 * @brief Counts non-zero values per row that satisfy a column filter
 *
 * This function counts the number of non-zero elements in each row that
 * pass the provided column index filter function.
 *
 * @tparam IndexType The integer type for indices
 * @tparam Functor The type of the filter function
 * @param nrow Number of rows to process
 * @param rowptr Row pointer array in CSR format
 * @param colind Column indices array in CSR format
 * @param indexing The indexing scheme (0 or 1-based)
 * @param rowcounts Output array to store counts per row
 * @param colidx_filter Function that returns true for column indices to include
 */
template <typename IndexType, typename Functor>
void count_row_nzval_submat(size_t nrow, const IndexType* rowptr,
                            const IndexType* colind, size_t indexing,
                            size_t* rowcounts, const Functor& colidx_filter) {
  for (size_t i = 0; i < nrow; ++i) {
    const auto j_st = rowptr[i] - indexing;
    const auto j_en = rowptr[i + 1] - indexing;

    const auto colind_st = colind + j_st;
    const auto colind_en = colind + j_en;

    auto nnz_row = std::count_if(colind_st, colind_en, colidx_filter);
    rowcounts[i] = nnz_row;
  }
}

/**
 * @brief Constructs row pointer array from row counts
 *
 * Given an array of row counts, computes the corresponding row pointer
 * array for CSR format. The row pointer at index i contains the starting
 * position of row i in the column indices array.
 *
 * @tparam IndexType The integer type for indices
 * @param nrow Number of rows
 * @param rowcnts Array of non-zero counts per row
 * @param rowptr Output row pointer array (size nrow+1)
 * @param indexing The indexing scheme (0 or 1-based)
 */
template <typename IndexType>
void rowptr_from_rowcnts(size_t nrow, const size_t* rowcnts, IndexType* rowptr,
                         size_t indexing) {
  rowptr[0] = indexing;
  for (size_t i = 0; i < nrow; ++i) rowptr[i + 1] = rowptr[i] + rowcnts[i];
}

}  // namespace detail

/**
 * @brief Extracts a rectangular submatrix from a sparse matrix
 *
 * Creates a new sparse matrix containing only the elements within the
 * specified row and column ranges. The column indices in the result are
 * offset to start from 0.
 *
 * @tparam SpMatType The sparse matrix type (must be CSR matrix)
 * @param A The input sparse matrix
 * @param lo Lower bounds as (row_start, col_start)
 * @param up Upper bounds as (row_end, col_end) - exclusive
 * @return New sparse matrix containing the specified submatrix
 *
 * @note The resulting submatrix has dimensions (row_end-row_start) x
 * (col_end-col_start)
 * @note Column indices in the result are re-indexed to start from 0
 */
template <typename SpMatType,
          typename = detail::enable_if_csr_matrix_t<SpMatType> >
SpMatType extract_submatrix(const SpMatType& A, std::pair<int64_t, int64_t> lo,
                            std::pair<int64_t, int64_t> up) {
  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

  const auto row_lo = lo.first;
  const auto row_up = up.first;
  const auto col_lo = lo.second;
  const auto col_up = up.second;
  const auto M_sub = row_up - row_lo;
  const auto N_sub = col_up - col_lo;

  // Determine row NNZ in submatrix
  std::vector<size_t> row_counts(M_sub);
  detail::count_row_nzval_submat(M_sub, Arp + row_lo, Aci, indexing,
                                 row_counts.data(), [=](auto j) {
                                   auto j_cor = j - indexing;
                                   return j_cor >= col_lo and j_cor < col_up;
                                 });

  // Calculate total NNZ
  const auto nnz_sub =
      std::accumulate(row_counts.begin(), row_counts.end(), 0ul);

  // Allocate submatrix
  SpMatType sub(M_sub, N_sub, nnz_sub, indexing);

  // Calculate rowtr
  auto* sub_rp = sub.rowptr().data();
  detail::rowptr_from_rowcnts(M_sub, row_counts.data(), sub_rp, indexing);

  // Extract submatrix
  auto* sub_nz = sub.nzval().data();
  auto* sub_ci = sub.colind().data();
  for (int64_t i = 0; i < M_sub; ++i) {
    const auto Aj_st = Arp[row_lo + i] - indexing;
    const auto Aj_en = Arp[row_lo + i + 1] - indexing;
    const auto sub_j_st = sub_rp[i] - indexing;

    const auto* Aci_st = Aci + Aj_st;
    const auto* Aci_en = Aci + Aj_en;
    auto* sub_ci_st = sub_ci + sub_j_st;

    const auto* Anz_st = Anz + Aj_st;
    // const auto* Anz_en    = Anz    + Aj_en;
    auto* sub_nz_st = sub_nz + sub_j_st;

    // Find the first j >= col_lo
    auto Aci_sub_st = std::partition_point(
        Aci_st, Aci_en, [=](auto j) { return (j - indexing) < col_lo; });

    // Find the first j >= col_up
    auto Aci_sub_en = std::partition_point(
        Aci_st, Aci_en, [=](auto j) { return (j - indexing) < col_up; });

    const auto ioff_st = std::distance(Aci_st, Aci_sub_st);
    const auto ioff_en = std::distance(Aci_st, Aci_sub_en);

    assert(ioff_en >= ioff_st);
    const auto* Anz_sub_st = Anz_st + ioff_st;
    const auto* Anz_sub_en = Anz_st + ioff_en;

    std::copy(Aci_sub_st, Aci_sub_en, sub_ci_st);
    std::copy(Anz_sub_st, Anz_sub_en, sub_nz_st);
  }

  for (auto& i : sub.colind()) i -= col_lo;  // Offset the columns bounds

  return sub;
}

/**
 * @brief Extracts a submatrix including specified rows and excluding specified
 * columns
 *
 * Creates a new sparse matrix containing the specified rows but excluding
 * the elements in the specified column range. This effectively removes
 * a vertical strip of columns from the matrix while keeping all rows.
 *
 * @tparam SpMatType The sparse matrix type (must be CSR matrix)
 * @param A The input sparse matrix
 * @param lo Lower bounds as (row_start, col_start)
 * @param up Upper bounds as (row_end, col_end) - exclusive
 * @return New sparse matrix with specified rows and excluded columns
 *
 * @note The resulting matrix has the same number of columns as the original
 * @note Column indices are not re-indexed (gaps remain where columns were
 * excluded)
 */
template <typename SpMatType,
          typename = detail::enable_if_csr_matrix_t<SpMatType> >
SpMatType extract_submatrix_inclrow_exclcol(const SpMatType& A,
                                            std::pair<int64_t, int64_t> lo,
                                            std::pair<int64_t, int64_t> up) {
  // const auto M = A.m();
  const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

  const auto row_lo = lo.first;
  const auto row_up = up.first;
  const auto col_lo = lo.second;
  const auto col_up = up.second;
  const auto M_sub = row_up - row_lo;
  const auto N_sub = N;

  // Determine row NNZ in submatrix
  std::vector<size_t> row_counts(M_sub);
  detail::count_row_nzval_submat(M_sub, Arp + row_lo, Aci, indexing,
                                 row_counts.data(), [=](auto j) {
                                   auto j_cor = j - indexing;
                                   return j_cor < col_lo or j_cor >= col_up;
                                 });

  // Calculate total NNZ
  const auto nnz_sub =
      std::accumulate(row_counts.begin(), row_counts.end(), 0ul);

  // Allocate submatrix
  SpMatType sub(M_sub, N_sub, nnz_sub, indexing);

  // Calculate rowtr
  auto* sub_rp = sub.rowptr().data();
  detail::rowptr_from_rowcnts(M_sub, row_counts.data(), sub_rp, indexing);

  // Extract submatrix
  auto* sub_nz = sub.nzval().data();
  auto* sub_ci = sub.colind().data();
  for (int64_t i = 0; i < M_sub; ++i) {
    const auto Aj_st = Arp[row_lo + i] - indexing;
    const auto Aj_en = Arp[row_lo + i + 1] - indexing;
    const auto sub_j_st = sub_rp[i] - indexing;

    const auto* Aci_st = Aci + Aj_st;
    const auto* Aci_en = Aci + Aj_en;
    auto* sub_ci_st = sub_ci + sub_j_st;

    const auto* Anz_st = Anz + Aj_st;
    const auto* Anz_en = Anz + Aj_en;
    auto* sub_nz_st = sub_nz + sub_j_st;

    // Find the first j >= col_lo
    auto Aci_sub_st = std::partition_point(
        Aci_st, Aci_en, [=](auto j) { return (j - indexing) < col_lo; });

    // Find the first j >= col_up
    auto Aci_sub_en = std::partition_point(
        Aci_st, Aci_en, [=](auto j) { return (j - indexing) < col_up; });

    const auto ioff_st = std::distance(Aci_st, Aci_sub_st);
    const auto ioff_en = std::distance(Aci_st, Aci_sub_en);

    assert(ioff_en >= ioff_st);
    const auto* Anz_sub_st = Anz_st + ioff_st;
    const auto* Anz_sub_en = Anz_st + ioff_en;

    // std::copy( Aci_sub_st, Aci_sub_en, sub_ci_st );
    // std::copy( Anz_sub_st, Anz_sub_en, sub_nz_st );
    sub_ci_st = std::copy(Aci_st, Aci_sub_st, sub_ci_st);
    sub_ci_st = std::copy(Aci_sub_en, Aci_en, sub_ci_st);
    sub_nz_st = std::copy(Anz_st, Anz_sub_st, sub_nz_st);
    sub_nz_st = std::copy(Anz_sub_en, Anz_en, sub_nz_st);
  }

  // for( auto& i : sub.colind() ) i -= col_lo; // Offset the columns bounds

  return sub;
}

/**
 * @brief Extracts the upper triangular part of a sparse matrix
 *
 * Creates a new sparse matrix containing only the upper triangular elements
 * (including the diagonal) of the input matrix. Elements where column index
 * is greater than or equal to row index are included.
 *
 * @tparam SpMatType The sparse matrix type (must be CSR matrix)
 * @param A The input sparse matrix
 * @return New sparse matrix containing only the upper triangular elements
 *
 * @note The resulting matrix has the same dimensions as the input matrix
 * @note Diagonal elements are included in the upper triangle
 */
template <typename SpMatType,
          typename = detail::enable_if_csr_matrix_t<SpMatType> >
SpMatType extract_upper_triangle(const SpMatType& A) {
  const auto M = A.m();
  const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

  // Determine NNZ and row counts in the upper triangle
  size_t nnz_ut = 0;
  std::vector<typename SpMatType::index_type> rowptr_ut(M + 1);
  rowptr_ut[0] = indexing;
  for (int64_t i = 0; i < M; ++i) {
    const auto j_st = Arp[i] - indexing;
    const auto j_en = Arp[i + 1] - indexing;

    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;

    auto nnz_row = std::count_if(
        Aci_st, Aci_en, [&](const auto j) { return (j - indexing) >= i; });

    nnz_ut += nnz_row;
    rowptr_ut[i + 1] = rowptr_ut[i] + nnz_row;
  }

  // Extract the Upper triangle
  SpMatType U(M, N, nnz_ut, indexing);
  // std::copy( rowptr_ut.begin(), rowptr_ut.end(), U.rowptr().begin() );
  U.rowptr() = std::move(rowptr_ut);
  auto* Unz = U.nzval().data();
  auto* Urp = U.rowptr().data();
  auto* Uci = U.colind().data();
  for (int64_t i = 0; i < M; ++i) {
    const auto Aj_st = Arp[i] - indexing;
    const auto Aj_en = Arp[i + 1] - indexing;
    const auto Uj_st = Urp[i] - indexing;

    const auto* Aci_st = Aci + Aj_st;
    const auto* Aci_en = Aci + Aj_en;
    auto* Uci_st = Uci + Uj_st;

    const auto* Anz_st = Anz + Aj_st;
    const auto* Anz_en = Anz + Aj_en;
    auto* Unz_st = Unz + Uj_st;

    const auto* Aci_ut_st = std::find_if(
        Aci_st, Aci_en, [&](const auto x) { return (x - indexing) >= i; });
    const auto ioff = std::distance(Aci_st, Aci_ut_st);
    const auto* Anz_ut_st = Anz_st + ioff;

    std::copy(Aci_ut_st, Aci_en, Uci_st);
    std::copy(Anz_ut_st, Anz_en, Unz_st);
  }

  return U;
}

/**
 * @brief Extracts the diagonal elements from a sparse matrix
 *
 * Returns a vector containing the diagonal elements of the sparse matrix.
 * If a diagonal element is not present in the sparse structure, it is
 * treated as zero.
 *
 * @tparam SpMatType The sparse matrix type (must be CSR matrix)
 * @param A The input sparse matrix
 * @return Vector containing the diagonal elements (size = min(m,n))
 *
 * @note Missing diagonal elements are set to zero
 * @note The function works for both square and rectangular matrices
 */
template <typename SpMatType,
          typename = detail::enable_if_csr_matrix_t<SpMatType> >
std::vector<typename SpMatType::value_type> extract_diagonal_elements(
    const SpMatType& A) {
  const auto M = A.m();
  // const auto N = A.n();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

  std::vector<typename SpMatType::value_type> D(M);

  using index_t = typename SpMatType::index_type;
  for (index_t i = 0; i < M; ++i) {
    const auto j_st = Arp[i] - indexing;
    const auto j_en = Arp[i + 1] - indexing;

    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;
    const auto* Anz_st = Anz + j_st;

    const auto diag_it = std::find(Aci_st, Aci_en, i + indexing);
    if (diag_it != Aci_en) {
      const auto diag_off = std::distance(Aci_st, diag_it);
      D[i] = Anz_st[diag_off];
    } else
      D[i] = 0;
  }

  return D;
}

/**
 * @brief Computes the trace (sum of diagonal elements) of a sparse matrix
 *
 * Calculates the sum of all diagonal elements in the sparse matrix.
 * Diagonal elements that are not explicitly stored in the sparse structure
 * are treated as zero and do not contribute to the sum.
 *
 * @tparam SpMatType The sparse matrix type (must be CSR matrix)
 * @param A The input sparse matrix
 * @return The trace (sum of diagonal elements)
 *
 * @note Only explicitly stored diagonal elements contribute to the trace
 * @note The function works for both square and rectangular matrices
 */
template <typename SpMatType,
          typename = detail::enable_if_csr_matrix_t<SpMatType> >
typename SpMatType::value_type trace(const SpMatType& A) {
  const auto M = A.m();

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

  typename SpMatType::value_type tr = 0.;

  using index_t = typename SpMatType::index_type;
  for (index_t i = 0; i < M; ++i) {
    const auto j_st = Arp[i] - indexing;
    const auto j_en = Arp[i + 1] - indexing;

    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;
    const auto* Anz_st = Anz + j_st;

    const auto diag_it = std::find(Aci_st, Aci_en, i + indexing);
    if (diag_it != Aci_en) {
      const auto diag_off = std::distance(Aci_st, diag_it);
      tr += Anz_st[diag_off];
    }
  }

  return tr;
}

}  // namespace sparsexx
