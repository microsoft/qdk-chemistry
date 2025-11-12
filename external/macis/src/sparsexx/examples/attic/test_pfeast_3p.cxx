/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <mpi.h>

#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/io/read_rb.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/util/submatrix.hpp>
#include <sparsexx/wrappers/mkl_dss_solver.hpp>
#include <sparsexx/wrappers/mkl_sparse_matrix.hpp>
extern "C" {
#include <pfeast.h>
#include <pfeast_sparse.h>
}

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  {
    assert(argc == 2);

    auto A = sparsexx::read_mm<double, sparsexx::detail::mkl::int_type>(
        std::string(argv[1]));
    MKL_INT N = A.m();

    auto* nzval = A.nzval().data();
    auto* colind = A.colind().data();
    auto* rowptr = A.rowptr().data();

    std::vector<double> slice_bounds = {-1.253, -0.993,
                                        //-0.813,
                                        -0.533, -0.323, 0.057, 0.467, 0.907,
                                        1.407, 1.777};

    size_t nslices = slice_bounds.size() - 1;

    int world_size;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    assert(world_size % 8 == 0);
    std::vector<int> grid_dims = {8, world_size / 8};
    std::vector<int> period = {true, false};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dims.data(), period.data(), false,
                    &cart_comm);

    MPI_Comm row_comm, col_comm;
    {
      std::vector<int> row_rem = {false, true};
      std::vector<int> col_rem = {true, false};
      MPI_Cart_sub(cart_comm, row_rem.data(), &row_comm);
      MPI_Cart_sub(cart_comm, col_rem.data(), &col_comm);
    }

    int row_rank, row_size;
    int col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    MPI_Comm l1_world = col_comm;

    int nL3 = 1;
    std::array<MKL_INT, 64> fpm;

    MPI_Barrier(MPI_COMM_WORLD);
    auto feast_st = std::chrono::high_resolution_clock::now();

    for (auto isl = 0; isl < nslices; ++isl) {
      if (isl % row_size != row_rank) continue;

      pfeastinit(fpm.data(), &l1_world, &nL3);
      fpm[0] = -isl - 1;
      fpm[2] = 8;
      // fpm[9] = 0;
      fpm[41] = 0;

      double epsout;
      MKL_INT loop, m_true, info;

      double emin = slice_bounds[isl];
      double emax = slice_bounds[isl + 1];
      MKL_INT m0 = 400;

      if (col_rank == 0) {
        printf("Process group %d will process slice %d [%.4f, %.4f]\n",
               row_rank, isl, emin, emax);
      }
      std::vector<double> W(m0), X(N * m0), RES(m0);
      char UPLO = 'F';
      pdfeast_scsrev(&UPLO, &N, nzval, rowptr, colind, fpm.data(), &epsout,
                     &loop, &emin, &emax, &m0, W.data(), X.data(), &m_true,
                     RES.data(), &info);

      // if( info ) throw std::runtime_error("FEAST DIED");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto feast_en = std::chrono::high_resolution_clock::now();

    if (not world_rank) {
      std::cout << "PFEAST Duration = "
                << std::chrono::duration<double>(feast_en - feast_st).count()
                << std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}
