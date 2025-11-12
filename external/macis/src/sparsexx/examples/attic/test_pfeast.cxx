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

    MPI_Comm l1_world = MPI_COMM_WORLD;
    // MPI_Fint l1_world_f = MPI_Comm_c2f(l1_world);
    int nL3 = 1;
    std::array<MKL_INT, 64> fpm;
    pfeastinit(fpm.data(), &l1_world, &nL3);
    fpm[0] = 1;
    fpm[2] = 8;
    // fpm[9] = 0;
    fpm[41] = 0;

#if 1
    double epsout;
    MKL_INT loop, m_true, info;

    // double emin = 25.;
    // double emax = 25.1;
    // double emin = 0.9;
    // double emax = 1.4;
    double emin = 1.4;
    double emax = 1.77;
    MKL_INT m0 = 400;

    // double emin = 0.;
    // double emax = 0.5;
    // MKL_INT m0  = 2;

    std::vector<double> W(m0), X(N * m0), RES(m0);
    MPI_Barrier(l1_world);
    auto feast_st = std::chrono::high_resolution_clock::now();
    char UPLO = 'F';
    pdfeast_scsrev(&UPLO, &N, nzval, rowptr, colind, fpm.data(), &epsout, &loop,
                   &emin, &emax, &m0, W.data(), X.data(), &m_true, RES.data(),
                   &info);
    MPI_Barrier(l1_world);
    auto feast_en = std::chrono::high_resolution_clock::now();
    if (info) throw std::runtime_error("FEAST DIED");

    int rank;
    MPI_Comm_rank(l1_world, &rank);
    if (not rank) {
      std::cout << "PFEAST Duration = "
                << std::chrono::duration<double>(feast_en - feast_st).count()
                << std::endl;
      std::cout << "W" << std::scientific << std::setprecision(10) << std::endl;
      std::sort(W.begin(), W.end());
      for (auto w : W) std::cout << w << std::endl;
    }
#endif
  }
  MPI_Finalize();
  return 0;
}
