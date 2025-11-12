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
#include <sparsexx/matrix_types/coo_matrix.hpp>
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
  using test2 = sparsexx::coo_matrix<double, int32_t>;
  MPI_Init(&argc, &argv);
  auto world_size = sparsexx::detail::get_mpi_size(MPI_COMM_WORLD);
  auto world_rank = sparsexx::detail::get_mpi_rank(MPI_COMM_WORLD);
  {
    assert(argc == 2);
    using spmat_type = sparsexx::csr_matrix<double, int32_t>;
    //  auto A = sparsexx::read_binary_triplet<spmat_type>( std::string( argv[1]
    //  ) ); auto A = sparsexx::read_mm<spmat_type>( std::string( argv[1] ) );

    int32_t nn = 1e6;
    int32_t nnz = 1000000;
    const int N = nn;

    //  const int N = A.m();

    // Default tiling
    int64_t nrow_per_rank = N / world_size;
    // using index_t = sparsexx::detail::index_type_t<decltype(A)>;
    //  using index_t = sparsexx::detail::index_type_t<test2>;
    using index_t = int32_t;

    std::vector<index_t> row_tiling(world_size + 1);
    for (auto i = 0; i < world_size; ++i) row_tiling[i] = i * nrow_per_rank;
    row_tiling.back() = N;

    std::vector<index_t> col_tiling = {0, N};

    std::cout << "HERE" << std::endl;

    // DBWY: FYI, you don't need this if you don't edit them per rank, what you
    // created above will avoid that, it'll only be needed if you e.g. graph
    // partition on the root rank to change the tiling
    if (world_size > 1) {
      MPI_Bcast(row_tiling.data(), row_tiling.size(), MPI_INT32_T, 0,
                MPI_COMM_WORLD);
      MPI_Bcast(col_tiling.data(), col_tiling.size(), MPI_INT32_T, 0,
                MPI_COMM_WORLD);
    }

    //  sparsexx::dist_sparse_matrix<decltype(A)> dist_A( MPI_COMM_WORLD, N, N,
    //    row_tiling, col_tiling );

    ///////////////////////////////////////
    //
    if (world_rank == 0)
      std::cout << " starting random matrix section " << std::endl;
    // int32_t nn = 1e6;
    // int32_t nnz = 1000000;
    int nnz_local = nnz / world_size;

    // sparsexx::dist_sparse_matrix<sparsexx::coo_matrix<double,int32_t>>
    // dist_B(MPI_COMM_WORLD, nn, nn, row_tiling, col_tiling);
    //  using test1 = sparsexx::csr_matrix<double, int32_t>;
    sparsexx::dist_sparse_matrix<test2> dist_B(MPI_COMM_WORLD, nn, nn,
                                               row_tiling, col_tiling);
    //  sparsexx::dist_sparse_matrix<test2> dist_C(MPI_COMM_WORLD, nn, nn,
    //  row_tiling, col_tiling);
    for (auto& [tile_index, local_tile] : dist_B) {
      auto m_local = local_tile.global_row_extent.second -
                     local_tile.global_row_extent.first;  // +1?
      auto n_local = local_tile.global_col_extent.second -
                     local_tile.global_col_extent.first;  // +1?
      std::vector<int32_t> is(nnz), js(nnz);
      std::vector<double> nzval(nnz);

      std::default_random_engine gen;
      std::uniform_real_distribution<> dist(0, 10);
      std::uniform_int_distribution<> idist(0, n_local);
      std::uniform_int_distribution<> jdist(0, m_local);

      if (world_rank == 0)
        std::cout << " starting generation of random matrix " << std::endl;

      std::generate(is.begin(), is.end(), [&]() { return idist(gen); });
      std::sort(is.begin(), is.end());

      js[0] = 0;
      for (int tt = 1; tt < nnz; tt++) {
        if (is[tt] == is[tt - 1]) js[tt] = js[tt - 1] + 1;
        if (is[tt] != is[tt - 1]) js[tt] = 0;
      }

      //  std::generate( js.begin(), js.end(), [&](){ return jdist(gen); } );
      std::generate(nzval.begin(), nzval.end(), [&]() { return dist(gen); });

      if (world_rank == 0)
        std::cout << " starting creation of local coo " << std::endl;

      sparsexx::coo_matrix<double, int32_t> BB(m_local, n_local, nnz_local, 0);
      BB.rowind() = std::move(is);
      BB.colind() = std::move(js);
      BB.nzval() = std::move(nzval);

      // populate
      local_tile.local_matrix = std::move(BB);
      std::cout << "Local NNZ = " << local_tile.local_matrix.nnz() << std::endl;
    }

    ///////////////////////////////////////////

    std::cout << "Create Vectors " << std::endl;
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

    //// sparsexx::spblas::gespmbv( K, 1., A, V.data(), N, 0., AV_serial.data(),
    /// N );

    auto pspbmv_dur = time_op([&]() {
      sparsexx::spblas::pgespmbv_grv(K, 1., dist_B, V.data(), N, 0.,
                                     AV_dist.data(), N);
    });

#if 0
  ss << "SPMBV DIFF " << world_rank << " = "
    << *std::max_element(V.begin(), V.end() ) << " "
    << *std::max_element(AV_serial.begin(), AV_serial.end() );
  for( auto i = 0ul; i < N*K; ++i )
    AV_serial[i] = std::abs( AV_serial[i] - AV_dist[i] );
  ss << " "  <<*std::max_element(AV_serial.begin(), AV_serial.end() ) << std::endl;;
#endif /* 0 */
    std::cout << ss.str();

    /*
      if( !world_rank ) {
        std::vector<int64_t> dist_nnz( world_size );
        for( int i = 0; i < world_size; ++i )
          dist_nnz[i] = A.rowptr()[row_tiling[i+1]] - A.rowptr()[row_tiling[i]];

        double max_load = *std::max_element( dist_nnz.begin(), dist_nnz.end() );
        double avg_load = A.nnz() / double(world_size);
        std::cout << "PSPMBV MAX = " << max_load << std::endl;
        std::cout << "PSPMBV AVG = " << avg_load << std::endl;
        std::cout << "PSPMBV IMB = " << (avg_load / max_load) << std::endl;
    ////    std::cout << "PSPMBV DUR = " << pspbmv_dur << " ms" << std::endl;
      }
    */
  }
  MPI_Finalize();
  return 0;
}
