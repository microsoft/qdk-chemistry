/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <sparsexx/sparsexx_config.hpp>

#ifdef SPARSEXX_ENABLE_MPI
#include <sparsexx/io/write_mm.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>

namespace sparsexx {

template <typename SpMatType>
void write_dist_mm(std::string fname, const dist_sparse_matrix<SpMatType>& A,
                   int forced_index = -1) {
  int mpi_rank;
  MPI_Comm_rank(A.comm(), &mpi_rank);
  int mpi_size;
  MPI_Comm_size(A.comm(), &mpi_size);

  // Get meta data
  const auto m = A.m();
  const auto n = A.n();
  size_t local_nnz = A.nnz();
  size_t total_nnz_root;
  MPI_Reduce(&local_nnz, &total_nnz_root, 1, MPI_UINT64_T, MPI_SUM, 0,
             A.comm());

  // Create the file
  if (!mpi_rank) {
    std::ofstream file(fname);
    write_mm_header(file, m, n, total_nnz_root, false);
  }
  MPI_Barrier(A.comm());

  int col_offset = 0;
  int row_offset = 0;
  if (forced_index >= 0) {
    col_offset = forced_index;  // Dist CSR is always 0-based
    row_offset = forced_index;
  }

  // Ring execute writes
  int token = 0;
  if (mpi_rank) {
    MPI_Recv(&token, 1, MPI_INT, mpi_rank - 1, 0, A.comm(), MPI_STATUS_IGNORE);
  }
  std::cout << "WRITING FROM RANK " << mpi_rank << std::endl;

  // Write Diagonal block
  {
    std::ofstream file(fname, std::ios::app);
    file << std::setprecision(17);
    const auto A_loc = A.diagonal_tile();
    write_mm_csr_block(file, A_loc, A.local_row_start() + row_offset,
                       A.local_row_start() + col_offset);
  }

  // Write Off-diagonal block
  if (A.off_diagonal_tile_ptr()) {
    std::ofstream file(fname, std::ios::app);
    file << std::setprecision(17);
    const auto A_loc = A.off_diagonal_tile();
    write_mm_csr_block(file, A_loc, A.local_row_start() + row_offset,
                       col_offset);
  }

  if (mpi_rank != mpi_size - 1) {
    MPI_Send(&token, 1, MPI_INT, mpi_rank + 1, 0, A.comm());
  }

  MPI_Barrier(A.comm());
}

}  // namespace sparsexx
#endif /* SPARSEXX_ENABLE_MPI */
