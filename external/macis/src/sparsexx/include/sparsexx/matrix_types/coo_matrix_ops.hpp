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

  std::cout << std::boolalpha;
  std::cout << "LT " << lower_triangle << std::endl;
  std::cout << "UT " << upper_triangle << std::endl;
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
