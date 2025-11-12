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

template <typename SpMatType>
void write_mm(std::string fname, const SpMatType& A, bool symm,
              int forced_index);

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
