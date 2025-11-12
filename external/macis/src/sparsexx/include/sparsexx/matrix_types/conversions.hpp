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
#if 0
  rowptr_.at(0) = other.indexing();
  auto cur_row = 0;
  for(size_type i = 0; i < nnz_; ++i)
    while(rowind_coo[i] != (cur_row + indexing_)) {
      cur_row++;
      rowptr_.at(cur_row) = i + indexing_;
    }
  rowptr_.at(m_) = nnz_ + indexing_;
#else
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
#endif /* 0 */

  // for(size_type i = 0; i < m_; ++i) {
  //   auto row_st = rowptr_[i];
  //   auto row_en = rowptr_[i+1];
  //   for(size_type j = row_st; j < row_en; ++j) {
  //     if(rowind_coo[j] != i) throw std::runtime_error("ROWPTR WRONG");
  //   }
  //   if(!std::is_sorted(colind_coo.begin() + row_st, colind_coo.begin() +
  //   row_en))
  //     throw std::runtime_error("COLIND WRONG");
  // }

  std::copy(colind_coo.begin(), colind_coo.end(), colind_.begin());
  std::copy(nzval_coo.begin(), nzval_coo.end(), nzval_.begin());
}

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

template <typename T, typename index_t, typename Alloc>
csr_matrix<T, index_t, Alloc>::csr_matrix(
    const csc_matrix<T, index_t, Alloc>& other)
    : csr_matrix(other.m(), other.n(), other.nnz(), other.indexing()) {
  detail::csr_to_csc(n_, m_, other.colptr().data(), other.rowind().data(),
                     other.nzval().data(), rowptr_.data(), colind_.data(),
                     nzval_.data(), indexing_);
}
}  // namespace sparsexx
