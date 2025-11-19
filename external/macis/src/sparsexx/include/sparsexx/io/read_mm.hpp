/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <stdexcept>
#include <string>

namespace sparsexx {

namespace detail {
/**
 * @brief Enumeration for COO matrix sorting schemes
 */
enum class coo_sort_scheme { no_sort, sort_by_row, sort_by_col };

/**
 * @brief Template struct to get the default COO sorting scheme for a matrix
 * type
 * @tparam MatType The matrix type
 */
template <typename MatType>
struct get_default_coo_sort_scheme;
/**
 * @brief Specialization for COO matrix - no sorting by default
 */
template <typename... Args>
struct get_default_coo_sort_scheme<coo_matrix<Args...> > {
  static constexpr auto value = coo_sort_scheme::no_sort;
};

/**
 * @brief Specialization for CSR matrix - sort by row by default
 */
template <typename... Args>
struct get_default_coo_sort_scheme<csr_matrix<Args...> > {
  static constexpr auto value = coo_sort_scheme::sort_by_row;
};

/**
 * @brief Specialization for CSC matrix - sort by column by default
 */
template <typename... Args>
struct get_default_coo_sort_scheme<csc_matrix<Args...> > {
  static constexpr auto value = coo_sort_scheme::sort_by_col;
};

/**
 * @brief Variable template for convenient access to default COO sorting scheme
 * @tparam MatType The matrix type
 */
template <typename MatType>
inline constexpr auto default_coo_sort_scheme_v =
    get_default_coo_sort_scheme<MatType>::value;
}  // namespace detail

/**
 * @brief Read a Matrix Market file and return it as a COO matrix
 * @tparam T The value type of the matrix elements
 * @tparam index_t The index type for row and column indices
 * @tparam Alloc The allocator type for memory management
 * @param fname The filename of the Matrix Market file to read
 * @param sort_scheme The sorting scheme to apply to the COO matrix
 * @return A COO matrix containing the data from the Matrix Market file
 * @throws std::runtime_error if the file is not a valid Matrix Market file or
 * has invalid dimensions
 */
template <typename T, typename index_t, typename Alloc>
coo_matrix<T, index_t, Alloc> read_mm_as_coo(
    std::string fname, detail::coo_sort_scheme sort_scheme) {
  std::ifstream f_in(fname);

  std::string line;

  int64_t m, n, nnz_min;
  bool is_sym = false;
  {
    std::getline(f_in, line);
    auto tokens = tokenize(line);

    // Check if this is actually a MM file...

    if (tokens[0].compare("%%MatrixMarket") or tokens.size() != 5)
      throw std::runtime_error(fname + " is not a MM file");

    is_sym = !tokens[4].compare("symmetric");

    while (std::getline(f_in, line)) {
      if (line[0] != '%') break;
    }

    // std::getline( f_in, line );
    tokens = tokenize(line);
    if (tokens.size() != 3)
      throw std::runtime_error(
          fname + " contains an invalid spec for problem dimension");

    m = std::stoll(tokens[0]);
    n = std::stoll(tokens[1]);
    nnz_min = std::stoll(tokens[2]);

    if (is_sym and m != n)
      throw std::runtime_error(fname + " symmetric not compatible with M!=N");

    if (is_sym) nnz_min *= 2;
  }

  // Read matrix entries from file
  std::vector<index_t> rowind, colind;
  std::vector<T> nzval;
  rowind.reserve(nnz_min);
  colind.reserve(nnz_min);
  nzval.reserve(nnz_min);

  size_t nnz_true = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    int64_t i = std::stoll(tokens[0]);
    int64_t j = std::stoll(tokens[1]);
    T v = (tokens.size() == 3) ? std::stod(tokens[2]) : 1.;

    rowind.push_back(i);
    colind.push_back(j);
    nzval.push_back(v);
    nnz_true++;

    // Handle symmetric matrices by adding transpose entries
    if (is_sym and i != j) {
      rowind.push_back(j);
      colind.push_back(i);
      nzval.push_back(v);
      nnz_true++;
    }
  }

  // Construct COO matrix from the read data
  coo_matrix<T, index_t, Alloc> A(m, n, std::move(colind), std::move(rowind),
                                  std::move(nzval));

  A.determine_indexing_from_adj();
  if (sort_scheme == detail::coo_sort_scheme::sort_by_row) {
    A.sort_by_row_index();
    assert(A.is_sorted_by_row_index());
  }
  if (sort_scheme == detail::coo_sort_scheme::sort_by_col) {
    A.sort_by_col_index();
    assert(A.is_sorted_by_col_index());
  }

  return A;
}

/**
 * @brief Read a Matrix Market file and return it as the specified sparse matrix
 * type
 * @tparam SpMatType The sparse matrix type to return (COO, CSR, or CSC)
 * @param fname The filename of the Matrix Market file to read
 * @return A sparse matrix of the specified type containing the data from the
 * Matrix Market file
 * @throws std::runtime_error if the file is not a valid Matrix Market file
 *
 * This function automatically determines the appropriate sorting scheme based
 * on the target matrix type and handles the conversion from COO format if
 * necessary.
 */
template <typename SpMatType>
SpMatType read_mm(std::string fname) {
  using value_t = detail::value_type_t<SpMatType>;
  using index_t = detail::index_type_t<SpMatType>;
  using allocator_t = detail::allocator_type_t<SpMatType>;

  // For COO matrices, read directly without sorting
  if constexpr (detail::is_coo_matrix_v<SpMatType>)
    return read_mm_as_coo<value_t, index_t, allocator_t>(
        fname, detail::coo_sort_scheme::no_sort);
  // For other matrix types, read as COO first then convert with appropriate
  // sorting
  else
    return SpMatType(read_mm_as_coo<value_t, index_t, allocator_t>(
        fname, detail::default_coo_sort_scheme_v<SpMatType>));
}

}  // namespace sparsexx
