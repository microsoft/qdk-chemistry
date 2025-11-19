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
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <string>

namespace sparsexx {

/**
 * @brief Reads a Rutherford-Boeing format file and returns a CSC (Compressed
 * Sparse Column) matrix.
 *
 * This function parses a matrix file in Rutherford-Boeing format and constructs
 * a CSC sparse matrix from the data. The function handles the specific format
 * requirements of RB files including metadata parsing and data extraction.
 *
 * @tparam T The value type for matrix elements (e.g., double, float)
 * @tparam index_t The index type for matrix indices (e.g., int, int64_t)
 * @tparam Alloc The allocator type for memory management
 *
 * @param fname The filename/path to the Rutherford-Boeing format file to read
 *
 * @return A CSC matrix constructed from the file data
 *
 * @throws std::runtime_error if the file cannot be opened or has invalid format
 * @throws std::invalid_argument if the matrix dimensions or data are
 * inconsistent
 *
 * @note The function expects the RB file to have the standard format:
 *       - Line 1: Comments
 *       - Line 2: Strange metadata (skipped)
 *       - Line 3: Type, rows, columns, nnz, format info
 *       - Line 4: Format specification (skipped)
 *       - Following lines: Column pointers, row indices, and non-zero values
 *
 * @warning The function assumes 1-based indexing in the input file and converts
 * to 0-based
 */
template <typename T, typename index_t, typename Alloc>
csc_matrix<T, index_t, Alloc> read_rb_as_csc(std::string fname) {
  std::ifstream f_in(fname);

  std::string line;

  // Skip the first two lines
  std::getline(f_in, line);  // comments
  std::getline(f_in, line);  // some strange metadata

  // Get useful meta data
  int64_t m, n, nnz;
  // bool is_sym;
  {
    std::getline(f_in, line);
    auto tokens = tokenize(line);
    assert(tokens.size() == 5);

    auto type = tokens[0];
    m = std::stoll(tokens[1]);
    n = std::stoll(tokens[2]);
    nnz = std::stoll(tokens[3]);

    // is_sym = type[1] == 's' or type[1] == 'S';
    assert(nnz <= m * n);
  }

  // Skip format line
  std::getline(f_in, line);

  csc_matrix<T, index_t, Alloc> A(m, n, nnz);

  int64_t curcount = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    for (const auto& t : tokens) A.colptr()[curcount++] = std::stoi(t);

    if (curcount == (n + 1)) break;
  }

  curcount = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    for (const auto& t : tokens) A.rowind()[curcount++] = std::stoi(t);

    if (curcount == nnz) break;
  }

  curcount = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    for (const auto& t : tokens) A.nzval()[curcount++] = std::stod(t);

    if (curcount == nnz) break;
  }

  assert(!std::getline(f_in, line));
  return A;
}

/**
 * @brief Generic function to read a Rutherford-Boeing format file into any
 * supported sparse matrix type.
 *
 * This is a generic wrapper around read_rb_as_csc that can return different
 * sparse matrix types based on the template parameter. It automatically deduces
 * the value type, index type, and allocator from the requested matrix type and
 * delegates to the appropriate specialized reader.
 *
 * @tparam SpMatType The target sparse matrix type to construct (e.g.,
 * csc_matrix, csr_matrix)
 *
 * @param fname The filename/path to the Rutherford-Boeing format file to read
 *
 * @return A sparse matrix of type SpMatType constructed from the file data
 *
 * @throws std::runtime_error if the file cannot be opened or has invalid format
 * @throws std::invalid_argument if the matrix dimensions or data are
 * inconsistent
 *
 * @example
 * ```cpp
 * // Read into a CSC matrix
 * auto csc_mat = read_rb<csc_matrix<double, int>>("matrix.rb");
 *
 * // Read into a CSR matrix (will be converted from CSC)
 * auto csr_mat = read_rb<csr_matrix<double, int>>("matrix.rb");
 * ```
 */
template <typename SpMatType>
SpMatType read_rb(std::string fname) {
  using value_t = detail::value_type_t<SpMatType>;
  using index_t = detail::index_type_t<SpMatType>;
  using allocator_t = detail::allocator_type_t<SpMatType>;

  if constexpr (detail::is_csc_matrix_v<SpMatType>)
    return read_rb_as_csc<value_t, index_t, allocator_t>(fname);
  else
    return SpMatType(read_rb_as_csc<value_t, index_t, allocator_t>(fname));
  abort();
}

}  // namespace sparsexx
