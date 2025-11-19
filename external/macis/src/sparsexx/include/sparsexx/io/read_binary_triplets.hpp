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

/**
 * @brief Reads binary triplet data from a file and returns it as a COO
 * (Coordinate) matrix
 *
 * This function reads binary matrix data stored in triplet format from a file
 * and constructs a COO (Coordinate) sparse matrix. The binary file format
 * contains matrix dimensions, number of non-zeros, followed by row indices,
 * column indices, and values.
 *
 * The function performs several post-processing steps:
 * - Determines indexing convention (0-based or 1-based) from adjacency data
 * - Expands matrix from triangular format if applicable
 * - Sorts entries by row index for efficient access
 *
 * @tparam SpMatType The sparse matrix type (must be a COO matrix type)
 *
 * @param fname The filename/path to the binary file containing triplet data
 *
 * @return A COO sparse matrix containing the data from the file
 *
 * @throws std::ifstream::failure if the file cannot be opened or read
 *
 * @note The binary file format is:
 *       - sizeof(index_t) bytes: number of rows (m)
 *       - sizeof(index_t) bytes: number of columns (n)
 *       - sizeof(size_t) bytes: number of non-zeros (nnz)
 *       - nnz * sizeof(index_t) bytes: row indices
 *       - nnz * sizeof(index_t) bytes: column indices
 *       - nnz * sizeof(value_t) bytes: matrix values
 */
template <typename SpMatType>
detail::enable_if_coo_matrix_t<SpMatType, SpMatType> read_binary_triplet_as_coo(
    std::string fname) {
  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ifstream f_in(fname, std::ios::binary);
  size_t nnz;
  index_t m, n;
  f_in.read((char*)&m, sizeof(index_t));
  f_in.read((char*)&n, sizeof(index_t));
  f_in.read((char*)&nnz, sizeof(size_t));

  std::cout << "Reading bin data" << std::endl;
  SpMatType A(m, n, nnz);
  f_in.read((char*)A.rowind().data(), nnz * sizeof(index_t));
  f_in.read((char*)A.colind().data(), nnz * sizeof(index_t));
  f_in.read((char*)A.nzval().data(), nnz * sizeof(value_t));

  A.determine_indexing_from_adj();
  A.expand_from_triangle();
  A.sort_by_row_index();

  assert(A.is_sorted_by_row_index());

  return A;
}

/**
 * @brief Reads binary triplet data from a file and returns it as the specified
 * matrix type
 *
 * This function provides a generic interface for reading binary triplet data
 * from a file and converting it to any supported sparse matrix format. It acts
 * as a dispatcher:
 * - For COO matrix types: directly calls read_binary_triplet_as_coo
 * - For other matrix types: reads as COO first, then converts to the target
 * type
 *
 * This approach leverages the fact that COO format is a natural intermediate
 * representation for triplet data, and most sparse matrix types provide
 * constructors from COO matrices.
 *
 * @tparam SpMatType The desired sparse matrix type (COO, CSR, CSC, etc.)
 *
 * @param fname The filename/path to the binary file containing triplet data
 *
 * @return A sparse matrix of type SpMatType containing the data from the file
 *
 * @throws std::ifstream::failure if the file cannot be opened or read
 * @throws Various constructor exceptions if conversion from COO to SpMatType
 * fails
 *
 * @note The binary file format is the same as described in
 * read_binary_triplet_as_coo
 *
 * @see read_binary_triplet_as_coo for details on the binary file format
 *
 * @example
 * @code
 * // Read as CSR matrix
 * auto csr_matrix = read_binary_triplet<csr_matrix<double>>("matrix.bin");
 *
 * // Read as COO matrix
 * auto coo_matrix = read_binary_triplet<coo_matrix<double>>("matrix.bin");
 * @endcode
 */
template <typename SpMatType>
SpMatType read_binary_triplet(std::string fname) {
  using value_t = detail::value_type_t<SpMatType>;
  using index_t = detail::index_type_t<SpMatType>;
  using allocator_t = detail::allocator_type_t<SpMatType>;

  if constexpr (detail::is_coo_matrix_v<SpMatType>)
    return read_binary_triplet_as_coo<SpMatType>(fname);
  else
    return SpMatType(
        read_binary_triplet_as_coo<coo_matrix<value_t, index_t, allocator_t> >(
            fname));
}

}  // namespace sparsexx
