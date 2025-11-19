/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <iomanip>
#include <random>
#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/io/write_dist_mm.hpp>
#include <sparsexx/io/write_mm.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/spblas/pspmbv.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/util/graph.hpp>
#include <sparsexx/util/mpi.hpp>

template <typename T>
auto operator-(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) throw std::runtime_error("");
  std::vector<T> c(a.size());
  for (auto i = 0; i < a.size(); ++i) {
    c[i] = std::abs(a[i] - b[i]);
  }
  return c;
}

#include <chrono>
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::milli>;

#include <functional>

template <typename T>
std::size_t hash_combine(std::size_t seed, const T& x) {
  return seed ^ (std::hash<T>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename T, typename... Args>
std::size_t hash_combine(std::size_t seed, const T& x, Args&&... args) {
  return hash_combine(hash_combine(seed, x), std::forward<Args>(args)...);
}

template <typename T>
struct std::hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T>& v) const noexcept {
    std::size_t seed = v.size();
    for (auto i : v) seed = hash_combine(seed, i);
    return seed;
  }
};

template <typename T, typename I>
struct std::hash<sparsexx::csr_matrix<T, I>> {
  std::size_t operator()(const sparsexx::csr_matrix<T, I>& A) const noexcept {
    return hash_combine(A.m(), A.n(), A.nnz(), A.rowptr(), A.colind(),
                        A.nzval());
  };
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  auto world_size = sparsexx::detail::get_mpi_size(MPI_COMM_WORLD);
  auto world_rank = sparsexx::detail::get_mpi_rank(MPI_COMM_WORLD);
  {
    assert(argc >= 2);
    using spmat_type = sparsexx::csr_matrix<double, int32_t>;
#if 1
    if (world_rank == 0) std::cout << "READING MATRIX" << std::endl;
    auto A = sparsexx::read_mm<spmat_type>(std::string(argv[1]));
    if (world_rank == 0) std::cout << "DONE READING MATRIX" << std::endl;
#else
    // [ x x 0 0 x x 0 0 x ]
    // [ x x 0 0 x x 0 0 x ]
    // [ 0 0 x x 0 0 x x x ]
    // [ 0 0 x x 0 0 x x x ]
    // [ x x 0 0 x x 0 0 0 ]
    // [ x x 0 0 x x 0 0 0 ]
    // [ 0 0 x x 0 0 x x 0 ]
    // [ 0 0 x x 0 0 x x 0 ]
    // [ x x x x 0 0 0 0 x ]
    spmat_type A(9, 9, 41, 0);
    A.rowptr() = {0, 5, 10, 15, 20, 24, 28, 32, 36, 41};
    A.colind() = {0, 1, 4, 5, 8, 0, 1, 4, 5, 8, 2, 3, 6, 7, 8, 2, 3, 6, 7, 8, 0,
                  1, 4, 5, 0, 1, 4, 5, 2, 3, 6, 7, 2, 3, 6, 7, 1, 2, 3, 4, 8};

    // std::iota(A.nzval().begin(), A.nzval().end(), 1. );

    A.nzval() = {10, 11, 14, 15, 18, 20, 21, 24, 25, 28, 32, 33, 36, 37,
                 38, 42, 43, 46, 47, 48, 50, 51, 54, 55, 60, 61, 64, 65,
                 72, 73, 76, 77, 82, 83, 86, 87, 91, 92, 93, 94, 98};

#endif /* 1 */
    // sparsexx::write_mm( "test.mtx", A, false, 1 );
    // auto A_cpy = sparsexx::read_mm<spmat_type>( "test.mtx" );
    // std::cout << (A.rowptr() == A_cpy.rowptr()) << std::endl;
    // std::cout << (A.colind() == A_cpy.colind()) << std::endl;

    const int N = A.m();

    const bool do_reorder = argc >= 3 ? std::stoi(argv[2]) : false;
    std::vector<int32_t> mat_perm;
    std::vector<std::pair<int32_t, int32_t>> row_extents;
    if (do_reorder) {
      int nparts = std::max(2l, world_size);
      // Reorder on root rank
      if (world_rank == 0) {
        // Partition the graph of A (assumes symmetric)
        auto kway_part_begin = clock_type::now();
        auto part = sparsexx::kway_partition(nparts, A);
        auto kway_part_end = clock_type::now();

        // Form permutation from partition
        std::vector<int32_t> partptr;
        std::tie(mat_perm, partptr) = sparsexx::perm_from_part(nparts, part);

        row_extents.resize(nparts);
        for (int i = 0; i < nparts; ++i) {
          row_extents[i] = {partptr[i], partptr[i + 1]};
          std::cout << "PARTITION " << i << " " << partptr[i + 1] - partptr[i]
                    << std::endl;
        }

        // std::cout << "PERM ";
        // for( auto x : mat_perm ) std::cout << x << " ";
        // std::cout << std::endl;

        // Permute rows/cols of A
        // A(I,P(J)) = A(P(I),J)
        auto permute_begin = clock_type::now();
        A = sparsexx::permute_rows_cols(A, mat_perm, mat_perm);
        auto permute_end = clock_type::now();

        duration_type kway_part_dur = kway_part_end - kway_part_begin;
        duration_type permute_dur = permute_end - permute_begin;

        std::cout << "KWAY PART DUR = " << kway_part_dur.count() << std::endl;
        std::cout << "PERMUTE DUR   = " << permute_dur.count() << std::endl;
      } else {
        mat_perm.resize(N);
      }

      // Broadcast reordered data
      if (world_size > 1) {
        sparsexx::detail::mpi_bcast(A.rowptr(), 0, MPI_COMM_WORLD);
        sparsexx::detail::mpi_bcast(A.colind(), 0, MPI_COMM_WORLD);
        sparsexx::detail::mpi_bcast(A.nzval(), 0, MPI_COMM_WORLD);
        sparsexx::detail::mpi_bcast(mat_perm, 0, MPI_COMM_WORLD);
      }
    }

    // sparsexx::detail::mpi_bcast( row_extents, 0, MPI_COMM_WORLD );
    size_t re_size = row_extents.size();
    sparsexx::detail::mpi_bcast(&re_size, 1, 0, MPI_COMM_WORLD);
    if (re_size) {
      row_extents.resize(re_size);  // Safe on root rank
      MPI_Bcast(&row_extents[0], 2 * re_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // if(!world_rank) {
    // std::vector<double> A_dense(A.n() * A.m());
    // sparsexx::convert_to_dense(A, A_dense.data(), A.n());
    // for(int i = 0 ; i < A.n(); ++i) {
    // for(int j = 0 ; j < A.n(); ++j)
    //   std::cout << std::setw(4) << A_dense[i + j*A.n()] << " ";
    // std::cout << std::endl;
    // }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    // Get distributed matrix
    using dist_mat_type = sparsexx::dist_sparse_matrix<spmat_type>;
    auto A_dist = row_extents.size()
                      ? dist_mat_type(MPI_COMM_WORLD, A, row_extents)
                      : dist_mat_type(MPI_COMM_WORLD, A);
    auto spmv_info = sparsexx::spblas::generate_spmv_comm_info(A_dist);

    // write_dist_mm("test_dist.mtx", A_dist, 1);
    // auto A_cpy = sparsexx::read_mm<spmat_type>( "test_dist.mtx" );
    // std::cout << (A.rowptr() == A_cpy.rowptr()) << std::endl;
    // std::cout << (A.colind() == A_cpy.colind()) << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // printf("[rank %2ld] nnz = %lu\n", world_rank, A_dist.nnz());
    size_t comm_vol = spmv_info.communication_volume();
    if (world_rank == 0) std::cout << "COMM VOLUME = " << comm_vol << std::endl;

    // Serial SPMV
    std::vector<double> V(N), AV(N);
    // std::iota( V.begin(), V.end(), 0 );
    // for( auto& x : V ) x *= 0.01;
    std::minstd_rand gen;
    std::uniform_real_distribution<> dist(-1, 1);
    std::generate(V.begin(), V.end(), [&]() { return dist(gen); });
    if (mat_perm.size()) {
      sparsexx::permute_vector(N, V.data(), mat_perm.data(),
                               sparsexx::PermuteDirection::Backward);
    }

    auto spmv_st = clock_type::now();
    if (world_rank == 0)
      sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., AV.data(), N);
    auto spmv_en = clock_type::now();
    if (mat_perm.size()) {
      sparsexx::permute_vector(N, AV.data(), mat_perm.data(),
                               sparsexx::PermuteDirection::Forward);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Parallel SPMV
    std::vector<double> V_dist(A_dist.local_row_extent()),
        AV_dist(A_dist.local_row_extent());
    auto [dist_row_st, dist_row_en] = A_dist.row_bounds(world_rank);
    for (auto i = dist_row_st; i < dist_row_en; ++i) {
      V_dist[i - dist_row_st] = V[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto pspmv_st = clock_type::now();

    sparsexx::spblas::pgespmv(1., A_dist, V_dist.data(), 0., AV_dist.data(),
                              spmv_info);

    MPI_Barrier(MPI_COMM_WORLD);
    auto pspmv_en = clock_type::now();

    if (!world_rank) {
      duration_type spmv_dur = spmv_en - spmv_st;
      duration_type pspmv_dur = pspmv_en - pspmv_st;
      std::cout << "SERIAL   = " << spmv_dur.count() << std::endl;
      std::cout << "PARALLEL = " << pspmv_dur.count() << std::endl;
    }

    // Compare results
    std::vector<double> AV_dist_combine(N);
    std::vector<int> recv_counts(world_size);
    std::vector<int> displs(world_size);
    for (auto i = 0; i < world_size; ++i) {
      auto [st, en] = A_dist.row_bounds(i);
      recv_counts[i] = en - st;
      displs[i] = st;
    }
    MPI_Allgatherv(AV_dist.data(), A_dist.local_row_extent(), MPI_DOUBLE,
                   AV_dist_combine.data(), recv_counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    if (mat_perm.size()) {
      sparsexx::permute_vector(N, AV_dist_combine.data(), mat_perm.data(),
                               sparsexx::PermuteDirection::Forward);
    }

    double max_diff = 0.;
    double nrm = 0.;
    for (auto i = 0; i < N; ++i) {
      auto diff = std::abs(AV_dist_combine[i] - AV[i]);
      max_diff = std::max(max_diff, diff);
      nrm += diff * diff;
    }
    nrm = std::sqrt(nrm);

    if (!world_rank) {
      std::cout << "MAX DIFF = " << max_diff << std::endl;
      std::cout << "NRM DIFF = " << nrm << std::endl;
    }
  }
  MPI_Finalize();
}
