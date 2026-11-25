/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <macis/util/mpi.hpp>

#if defined(MACIS_ENABLE_MPI) && defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char* argv[]) {
#ifdef MACIS_ENABLE_MPI
  int required = MPI_THREAD_SINGLE;
#ifdef _OPENMP
  // ASCI workers issue MPI RMA calls from the OpenMP parallel region.
  if (omp_get_max_threads() > 1) required = MPI_THREAD_MULTIPLE;
#endif
  int provided;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
    std::cerr << "Threaded MACIS tests require MPI_THREAD_MULTIPLE; "
                 "use OMP_NUM_THREADS=1 for single-threaded tests.\n";
    MPI_Finalize();
    return 1;
  }
#endif
  int result = Catch::Session().run(argc, argv);
#ifdef MACIS_ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return result;
}
