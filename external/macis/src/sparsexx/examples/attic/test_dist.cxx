/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sparsexx/io/read_binary_triplets.hpp>
#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/io/write_binary_triplets.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/util/reorder.hpp>
#include <sparsexx/util/submatrix.hpp>

template <typename Op>
double time_op(const Op& op) {
  MPI_Barrier(MPI_COMM_WORLD);
  auto st = std::chrono::high_resolution_clock::now();

  op();

  MPI_Barrier(MPI_COMM_WORLD);
  auto en = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(en - st).count();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  auto world_size = sparsexx::detail::get_mpi_size(MPI_COMM_WORLD);
  auto world_rank = sparsexx::detail::get_mpi_rank(MPI_COMM_WORLD);
  {
    assert(argc == 2);
    using spmat_type = sparsexx::csr_matrix<double, int32_t>;
    // auto A = sparsexx::read_binary_triplet<spmat_type>( std::string( argv[1]
    // ) );
    auto A = sparsexx::read_mm<spmat_type>(std::string(argv[1]));
    const int N = A.m();

    // Default tiling
    int64_t nrow_per_rank = N / world_size;
    using index_t = sparsexx::detail::index_type_t<decltype(A)>;

    std::vector<index_t> row_tiling(world_size + 1);
    for (auto i = 0; i < world_size; ++i) row_tiling[i] = i * nrow_per_rank;
    row_tiling.back() = N;

    std::vector<index_t> col_tiling = {0, N};

    // Reordering
    if (world_rank == 0) {
      std::stringstream ss;
      int64_t nparts = std::max(2l, world_size);

      std::cout << "Partitioning..." << std::flush;
      auto part_st = std::chrono::high_resolution_clock::now();
      auto part = sparsexx::kway_partition(nparts, A);
      auto part_en = std::chrono::high_resolution_clock::now();
      std::cout << "Done!" << std::endl;

      auto part_dur =
          std::chrono::duration<double, std::milli>(part_en - part_st).count();

      ss << "Partition counts" << std::endl;
      for (auto i = 0; i < nparts; ++i)
        ss << "  Group " << i << " = "
           << std::count(part.begin(), part.end(), i) << std::endl;

      std::cout << "Forming Perm...";
      auto fperm_st = std::chrono::high_resolution_clock::now();
      auto [perm, partptr] = sparsexx::perm_from_part(nparts, part);
      auto fperm_en = std::chrono::high_resolution_clock::now();
      std::cout << "Done!" << std::endl;

      auto fperm_dur =
          std::chrono::duration<double, std::milli>(fperm_en - fperm_st)
              .count();

      for (auto ind : partptr) ss << "  " << ind << std::endl;

      // perm = {1, 3, 4, 2};
      // perm.at(0) = 0;
      // perm.at(1) = 2;
      // perm.at(2) = 3;
      // perm.at(3) = 1;

      std::cout << "Permuting...";
      auto perm_st = std::chrono::high_resolution_clock::now();
      auto Ap = sparsexx::permute_rows(A, perm);
      auto perm_en = std::chrono::high_resolution_clock::now();
      auto perm_dur =
          std::chrono::duration<double, std::milli>(perm_en - perm_st).count();
      std::cout << "Done!" << std::endl;

      ss << std::scientific << std::setprecision(5);
      ss << "METIS DUR     = " << part_dur << " ms" << std::endl;
      ss << "Form PERM DUR = " << fperm_dur << " ms" << std::endl;
      ss << "PERM DUR      = " << perm_dur << " ms" << std::endl;

      std::cout << ss.str();

      // A = std::move(Ap);
      // row_tiling = std::move(partptr);
    }

    if (world_size > 1) {
      MPI_Bcast(A.rowptr().data(), A.rowptr().size(), MPI_INT32_T, 0,
                MPI_COMM_WORLD);
      MPI_Bcast(A.colind().data(), A.colind().size(), MPI_INT32_T, 0,
                MPI_COMM_WORLD);
      MPI_Bcast(A.nzval().data(), A.nzval().size(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
      MPI_Bcast(row_tiling.data(), row_tiling.size(), MPI_INT32_T, 0,
                MPI_COMM_WORLD);
      MPI_Bcast(col_tiling.data(), col_tiling.size(), MPI_INT32_T, 0,
                MPI_COMM_WORLD);
    }

    sparsexx::dist_csr_matrix<double, int32_t> dist_A(MPI_COMM_WORLD, N, N,
                                                      row_tiling, col_tiling);

    sparsexx::dist_coo_matrix<double, int32_t> dist_coo(MPI_COMM_WORLD, N, N,
                                                        row_tiling, col_tiling);

    if (world_rank == 0) std::cout << "GLOBAL NNZ = " << A.nnz() << std::endl;
    for (auto& [tile_index, local_tile] : dist_A) {
      std::stringstream ss;
      ss << world_rank << ": "
         << "(" << tile_index.first << ", " << tile_index.second << ") -> ["
         << local_tile.global_row_extent.first << ", "
         << local_tile.global_row_extent.second << ") ";

      local_tile.local_matrix =
          extract_submatrix(A,
                            {local_tile.global_row_extent.first,
                             local_tile.global_col_extent.first},
                            {local_tile.global_row_extent.second,
                             local_tile.global_col_extent.second});

      ss << "Local NNZ = " << local_tile.local_matrix.nnz() << std::endl;

      // std::vector<double> local_dense( local_tile.local_matrix.m() *
      // local_tile.local_matrix.n() ); sparsexx::convert_to_dense(
      // local_tile.local_matrix, local_dense.data(),
      // local_tile.local_matrix.m() );

      std::cout << ss.str();
    }

    const int64_t K = 2;
    std::vector<double> V(N * K), AV_serial(N * K, 0.), AV_dist(N * K, 0.);

    if (not world_rank) {
      std::vector<std::default_random_engine> gen;
      std::vector<std::normal_distribution<> > dist;
      for (auto it = 0; it < omp_get_max_threads(); ++it) {
        gen.emplace_back(it);
        dist.emplace_back(0., 1.);
      }

      auto rand_gen = [&]() {
        auto tid = omp_get_thread_num();
        auto g = gen.at(tid);
        auto d = dist.at(tid);
        return d(g);
      };

#pragma omp parallel for
      for (auto i = 0ul; i < V.size(); ++i) V[i] = rand_gen();
    }

    MPI_Bcast(V.data(), V.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::stringstream ss;

    // ss << std::endl;
    // ss << "V " << world_rank << std::endl;
    // for( auto i = 0; i < N; ++i ) {
    //   for( auto k = 0; k < K; ++k )
    //     ss << V[i + k*N] << " ";
    //   ss << std::endl;
    // }

    sparsexx::spblas::gespmbv(K, 1., A, V.data(), N, 0., AV_serial.data(), N);

    auto pspbmv_dur = time_op([&]() {
      sparsexx::spblas::pgespmbv_grv(K, 1., dist_A, V.data(), N, 0.,
                                     AV_dist.data(), N);
    });

#if 0
  ss << "SPMBV DIFF " << world_rank << " = "
    << *std::max_element(V.begin(), V.end() ) << " "
    << *std::max_element(AV_serial.begin(), AV_serial.end() );
  for( auto i = 0ul; i < N*K; ++i )
    AV_serial[i] = std::abs( AV_serial[i] - AV_dist[i] );
  ss << " "  <<*std::max_element(AV_serial.begin(), AV_serial.end() ) << std::endl;;
#endif
    std::cout << ss.str();

    if (!world_rank) {
      std::vector<int64_t> dist_nnz(world_size);
      for (int i = 0; i < world_size; ++i)
        dist_nnz[i] = A.rowptr()[row_tiling[i + 1]] - A.rowptr()[row_tiling[i]];

      double max_load = *std::max_element(dist_nnz.begin(), dist_nnz.end());
      double avg_load = A.nnz() / double(world_size);
      std::cout << "PSPMBV MAX = " << max_load << std::endl;
      std::cout << "PSPMBV AVG = " << avg_load << std::endl;
      std::cout << "PSPMBV IMB = " << (avg_load / max_load) << std::endl;
      std::cout << "PSPMBV DUR = " << pspbmv_dur << " ms" << std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}
