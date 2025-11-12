/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <omp.h>

#include <algorithm>
#include <cereal/archives/binary.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sparsexx/io/read_binary_triplets.hpp>
#include <sparsexx/io/write_binary_triplets.hpp>
#include <sparsexx/matrix_types/coo_matrix.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/util/reorder.hpp>
#include <sparsexx/util/submatrix.hpp>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  auto world_size = sparsexx::detail::get_mpi_size(MPI_COMM_WORLD);
  auto world_rank = sparsexx::detail::get_mpi_rank(MPI_COMM_WORLD);
  {
    assert(argc == 2);
    using spmat_type = sparsexx::csr_matrix<double, int32_t>;

    std::string local_matrix;

    if (world_rank == 0) {
      spmat_type Ap;
      std::vector<int32_t> perm, partptr;
      {
        auto read_st = std::chrono::high_resolution_clock::now();
        auto A =
            sparsexx::read_binary_triplet<spmat_type>(std::string(argv[1]));
        auto read_en = std::chrono::high_resolution_clock::now();

        int64_t nparts = std::max(2l, world_size);

        auto part_st = std::chrono::high_resolution_clock::now();
        auto part = sparsexx::kway_partition(nparts, A);
        auto part_en = std::chrono::high_resolution_clock::now();

        auto fperm_st = std::chrono::high_resolution_clock::now();
        std::tie(perm, partptr) = sparsexx::perm_from_part(nparts part);
        auto fperm_en = std::chrono::high_resolution_clock::now();

        auto perm_st = std::chrono::high_resolution_clock::now();
        Ap = sparsexx::permute_rows_cols(A, perm, perm);
        auto perm_en = std::chrono::high_resolution_clock::now();
      }

      int32_t m = Ap.m(), n = Ap.n();
    }

#if 0
  std::cout << "Serializing" << std::endl;
  std::string A_bin;
  {
    std::ostringstream ss;
    {
    cereal::BinaryOutputArchive ar( ss );
    ar(A);
    }
    A_bin = ss.str();
  }

  std::cout << "Deserializing" << std::endl;
  spmat_type A_copy;
  {
    std::istringstream ss( A_bin );
    cereal::BinaryInputArchive ar( ss );
    ar( A_copy );
  }

  assert( A_copy.rowind() == A.rowind() );
  assert( A_copy.colind() == A.colind() );
  assert( A_copy.nzval() == A.nzval() );
#endif
  }
  MPI_Finalize();
  return 0;
}
