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
 * @brief Writes a COO (Coordinate) sparse matrix to a binary file in triplet
 * format.
 *
 * This function writes a COO sparse matrix to a binary file using a custom
 * triplet format. The binary format consists of matrix dimensions followed by
 * the triplet data (row indices, column indices, and non-zero values) stored
 * contiguously in binary format for efficient I/O operations.
 *
 * @tparam SpMatType The sparse matrix type (must be a COO matrix type)
 *
 * @param A The COO sparse matrix to write to file
 * @param fname The filename/path where the binary data will be written
 *
 * @throws std::runtime_error if the file cannot be opened for writing
 * @throws std::ios_base::failure if writing to the file fails
 *
 * @note Binary format structure:
 *       - sizeof(index_t) bytes: number of rows (m)
 *       - sizeof(index_t) bytes: number of columns (n)
 *       - sizeof(size_t) bytes: number of non-zero elements (nnz)
 *       - nnz * sizeof(index_t) bytes: row indices array
 *       - nnz * sizeof(index_t) bytes: column indices array
 *       - nnz * sizeof(value_t) bytes: non-zero values array
 */
template <typename SpMatType>
detail::enable_if_coo_matrix_t<SpMatType> write_binary_triplet(
    const SpMatType& A, std::string fname) {
  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ofstream f_out(fname, std::ios::binary);
  size_t nnz = A.nnz();
  ;
  index_t m = A.m(), n = A.n();
  f_out.write((char*)&m, sizeof(index_t));
  f_out.write((char*)&n, sizeof(index_t));
  f_out.write((char*)&nnz, sizeof(size_t));

  f_out.write((char*)A.rowind().data(), nnz * sizeof(index_t));
  f_out.write((char*)A.colind().data(), nnz * sizeof(index_t));
  f_out.write((char*)A.nzval().data(), nnz * sizeof(value_t));
}

/**
 * @brief Writes a CSR (Compressed Sparse Row) sparse matrix to a binary file in
 * triplet format.
 *
 * This function converts a CSR sparse matrix to triplet format and writes it to
 * a binary file. Since CSR format stores data in compressed form with row
 * pointers, this function first reconstructs the explicit row indices from the
 * row pointer array, then writes the triplet data in the same binary format as
 * the COO version.
 *
 * @tparam SpMatType The sparse matrix type (must be a CSR matrix type)
 *
 * @param A The CSR sparse matrix to write to file
 * @param fname The filename/path where the binary data will be written
 *
 * @throws std::runtime_error if the file cannot be opened for writing
 * @throws std::ios_base::failure if writing to the file fails
 * @throws std::bad_alloc if memory allocation for row indices fails
 *
 * @note Binary format structure (identical to COO version):
 *       - sizeof(index_t) bytes: number of rows (m)
 *       - sizeof(index_t) bytes: number of columns (n)
 *       - sizeof(size_t) bytes: number of non-zero elements (nnz)
 *       - nnz * sizeof(index_t) bytes: row indices array (reconstructed from
 * rowptr)
 *       - nnz * sizeof(index_t) bytes: column indices array
 *       - nnz * sizeof(value_t) bytes: non-zero values array
 */
template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType> write_binary_triplet(
    const SpMatType& A, std::string fname) {
  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ofstream f_out(fname, std::ios::binary);
  size_t nnz = A.nnz();
  ;
  index_t m = A.m(), n = A.n();
  f_out.write((char*)&m, sizeof(index_t));
  f_out.write((char*)&n, sizeof(index_t));
  f_out.write((char*)&nnz, sizeof(size_t));

  // Construct rowind
  std::vector<index_t> rowind(nnz);
  auto rowind_it = rowind.begin();
  for (size_t i = 0; i < m; ++i) {
    const auto row_count = A.rowptr()[i + 1] - A.rowptr()[i];
    rowind_it = std::fill_n(rowind_it, row_count, i + A.indexing());
  }
  assert(rowind_it == rowind.end());

  f_out.write((char*)rowind.data(), nnz * sizeof(index_t));
  f_out.write((char*)A.colind().data(), nnz * sizeof(index_t));
  f_out.write((char*)A.nzval().data(), nnz * sizeof(value_t));
}

}  // namespace sparsexx
