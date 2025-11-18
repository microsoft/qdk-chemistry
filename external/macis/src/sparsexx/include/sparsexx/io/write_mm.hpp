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
#include <iomanip>
#include <iostream>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <stdexcept>
#include <string>

namespace sparsexx {

/**
 * @brief Writes the Matrix Market format header to an output stream.
 *
 * This function writes the standard Matrix Market header that describes the
 * matrix format, data type, and structure. The header consists of the banner
 * line and the size/nnz line according to the Matrix Market specification.
 *
 * @param file The output stream to write the header to
 * @param m Number of rows in the matrix
 * @param n Number of columns in the matrix
 * @param nnz Number of non-zero entries in the matrix
 * @param symmetric Whether the matrix is symmetric or general
 *
 * @note Matrix Market header format:
 *       - Banner: "%%MatrixMarket matrix coordinate real [symmetric|general]"
 *       - Dimensions: "m n nnz_adjusted"
 *       - For symmetric matrices, nnz is adjusted to count only upper
 * triangular + diagonal
 *
 * @note The function assumes real-valued matrices and coordinate (triplet)
 * format
 *
 * @see Matrix Market format specification:
 * https://math.nist.gov/MatrixMarket/formats.html
 */
inline void write_mm_header(std::ostream& file, size_t m, size_t n, size_t nnz,
                            bool symmetric) {
  file << "%%MatrixMarket matrix coordinate real ";
  if (symmetric)
    file << "symmetric";
  else
    file << "general";
  file << "\n";
  file << m << " " << n << " ";
  if (symmetric)
    file << ((nnz - n) / 2 + n);
  else
    file << nnz;
  file << std::endl;
}

/**
 * @brief Writes a CSR matrix block to an output stream in Matrix Market triplet
 * format.
 *
 * This function writes the data portion of a CSR matrix to a stream in Matrix
 * Market coordinate format (row column value triplets). The function handles
 * index offsets to allow writing matrix blocks or adjusting indexing base.
 *
 * @tparam Args Template parameter pack for CSR matrix template arguments
 *
 * @param file The output stream to write the matrix data to
 * @param A The CSR matrix to write
 * @param row_off Row index offset to add to each row index
 * @param col_off Column index offset to add to each column index
 *
 * @note The function respects the matrix's internal indexing scheme and adjusts
 * accordingly
 * @note Each line in the output contains: "row_index column_index value"
 * @note Indices in output are adjusted by the specified offsets
 *
 * @warning The function assumes the matrix data is valid and does not perform
 * bounds checking
 *
 * @see write_mm_header for writing the corresponding Matrix Market header
 */
template <typename... Args>
void write_mm_csr_block(std::ostream& file, const csr_matrix<Args...>& A,
                        int row_off, int col_off) {
  const auto m = A.m();
  const auto n = A.n();
  const auto nnz = A.nnz();

  for (auto i = 0; i < m; ++i) {
    const auto inz_st = A.rowptr()[i] - A.indexing();
    const auto inz_en = A.rowptr()[i + 1] - A.indexing();
    for (auto inz = inz_st; inz < inz_en; ++inz) {
      const auto j = A.colind()[inz];
      const auto nzval = A.nzval()[inz];
      file << i + row_off << " " << j + col_off << " " << nzval << "\n";
    }
  }
}

/**
 * @brief Generic function declaration for writing sparse matrices to Matrix
 * Market format.
 *
 * This is a forward declaration for the generic write_mm function that can
 * handle different sparse matrix types. Specific implementations are provided
 * for supported matrix formats.
 *
 * @tparam SpMatType The sparse matrix type to write
 *
 * @param fname The filename/path where the Matrix Market file will be written
 * @param A The sparse matrix to write to file
 * @param symm Whether to treat the matrix as symmetric
 * @param forced_index Optional index base override (default: -1 for automatic
 * detection)
 *
 * @see write_mm(std::string, const csr_matrix<Args...>&, bool, int) for CSR
 * implementation
 */
template <typename SpMatType>
void write_mm(std::string fname, const SpMatType& A, bool symm,
              int forced_index);

/**
 * @brief Writes a CSR sparse matrix to a file in Matrix Market format.
 *
 * This function writes a CSR (Compressed Sparse Row) matrix to a file using the
 * Matrix Market coordinate format. The function handles index base conversion
 * and outputs the matrix in standard triplet format (row, column, value).
 *
 * @tparam Args Template parameter pack for CSR matrix template arguments
 *
 * @param fname The filename/path where the Matrix Market file will be written
 * @param A The CSR matrix to write to file
 * @param symm Whether to treat the matrix as symmetric (currently not
 * implemented)
 * @param forced_index Optional index base override (default: -1 for automatic
 * detection)
 *                     - If >= 0, forces the specified indexing base (0 or 1) in
 * output
 *                     - If -1, uses the matrix's natural indexing scheme
 *
 * @throws std::runtime_error if symm is true (symmetric output not yet
 * implemented)
 * @throws std::ios_base::failure if file creation or writing fails
 *
 * @note Output format:
 *       - Matrix Market header with dimensions and nnz count
 *       - Triplet data: one "row column value" per line
 *       - Floating-point precision set to 17 digits
 *       - File is flushed after writing to ensure data persistence
 *
 * @note Index handling:
 *       - If forced_index >= 0, output indices start from forced_index
 *       - Row indices are adjusted by forced_index directly
 *       - Column indices are adjusted by (forced_index - matrix_indexing)
 *
 * @warning Symmetric matrix output is not yet implemented and will throw an
 * exception
 *
 * @see write_mm_header for header format details
 * @see write_mm_csr_block for the underlying block writing implementation
 */
template <typename... Args>
void write_mm(std::string fname, const csr_matrix<Args...>& A, bool symm,
              int forced_index = -1) {
  if (symm) throw std::runtime_error("write_mm + symmetric NYI");

  // Get meta data
  const auto m = A.m();
  const auto n = A.n();
  const auto nnz = A.nnz();

  // open file and write header
  std::ofstream file(fname);
  write_mm_header(file, m, n, nnz, symm);

  int col_offset = 0;
  int row_offset = 0;
  if (forced_index >= 0) {
    col_offset = forced_index - A.indexing();
    row_offset = forced_index;
  }

  file << std::setprecision(17);
  write_mm_csr_block(file, A, row_offset, col_offset);
  file << std::flush;
}

}  // namespace sparsexx
