/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <type_traits>

namespace sparsexx::spblas {

template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

template <typename SpMatType, typename ALPHAT, typename BETAT>
// std::enable_if_t< detail::spmbv_uses_generic_csr_v<SpMatType, ALPHAT, BETAT>
// >
void pgespmbv_grv(int64_t K, ALPHAT ALPHA,
                  const dist_sparse_matrix<SpMatType>& A,
                  const typename SpMatType::value_type* V, int64_t LDV,
                  BETAT BETA, typename SpMatType::value_type* AV,
                  int64_t LDAV) {
  if (LDAV != A.global_m())
    throw std::runtime_error("AV cannot be a submatrix");

// Scale AV
#pragma omp parallel for collapse(2)
  for (int64_t k = 0; k < K; ++k)
    for (int64_t i = 0; i < A.global_m(); ++i) AV[i + k * LDAV] *= BETA;

  // Loop over local tiles and compute local pieces of spbmv
  for (const auto& [tile_index, local_tile] : A) {
    const auto& [row_st, row_en] = local_tile.global_row_extent;
    const auto& [col_st, col_en] = local_tile.global_col_extent;

    const auto spmv_m = row_en - row_st;
    const auto spmv_n = K;
    const auto spmv_k = col_en - col_st;

    const auto* V_local = V + col_st;
    auto* AV_local = AV + row_st;

    gespmbv(spmv_n, ALPHA, local_tile.local_matrix, V_local, LDV, 1., AV_local,
            LDAV);
  }

  auto comm = A.comm();
  MPI_Allreduce(
      MPI_IN_PLACE, AV, K * LDAV,
      sparsexx::detail::mpi_data_t<sparsexx::detail::value_type_t<SpMatType>>,
      MPI_SUM, comm);
}

template <typename SpMatType, typename T = typename SpMatType::value_type>
sparsexx::detail::enable_if_csr_matrix_t<SpMatType> pgespmv_rdv(
    type_identity_t<T> ALPHA, const dist_sparse_matrix<SpMatType>& A,
    const T* V, type_identity_t<T> BETA, T* AV) {
  const auto M = A.global_m();
  const auto N = A.global_n();
  if (M != N) throw std::runtime_error("Only works for square matrices");

  auto comm = A.comm();
  auto comm_rank = sparsexx::detail::get_mpi_rank(comm);
  auto comm_size = sparsexx::detail::get_mpi_size(comm);
  auto& row_tiling = A.row_tiling();
  auto& col_tiling = A.col_tiling();

  if (row_tiling.size() != (comm_size + 1) or col_tiling.size() != 2)
    throw std::runtime_error("Only works for block row dist");

  // Get all local row counts
  std::vector<size_t> local_rows(comm_size);
  for (int i = 0; i < comm_size; ++i) {
    local_rows[i] = row_tiling[i + 1] - row_tiling[i];
  }

  // Scale AV
  for (int64_t i = 0; i < local_rows[comm_rank]; ++i) AV[i] *= BETA;

  MPI_Win V_win;
#if 1
  MPI_Win_create((void*)const_cast<T*>(V), local_rows[comm_rank] * sizeof(T),
                 sizeof(T), MPI_INFO_NULL, comm, &V_win);
#else
  T* rma_ptr;
  MPI_Win_allocate(v_buffer_sz * sizeof(T), sizeof(T), MPI_INFO_NULL, comm,
                   &rma_ptr, &V_win),
#endif /* 1 */

  MPI_Win_lock_all(MPI_MODE_NOCHECK, V_win);
  std::vector<T> local_v;

  for (const auto& [tile_index, local_tile] : A) {
    const auto& local_mat = local_tile.local_matrix;
    const auto local_m = local_mat.m();
    const auto local_n = local_mat.n();

    local_v.resize(local_n);  // resize to local dim
    // auto col_idx_start = local_tile.global_col_extent.first;

    std::set<sparsexx::detail::index_type_t<SpMatType>> unique_colind(
        local_mat.colind().cbegin(), local_mat.colind().cend());

#if 0
    // Get pieces that we need

    // Find owner of first element
    int32_t target_rank;
    const auto first_unique = *unique_colind.begin();
    for( auto ir = 0; ir < comm_size; ++ir )
    if( row_tiling[ir] <= first_unique and first_unique < row_tiling[ir+1] ) {
      target_rank = ir; break;
    }



    for( auto j : unique_colind ) {
      // Local buffer offset
      int32_t local_offs = j; // - col_idx_start; // b/c 1d block row

      // Update target rank as needed
      if( j >= row_tiling[target_rank+1] ) target_rank++;
      std::cout << "rank" << comm_rank << ": " << target_rank << ", " << j << std::endl;

      // Displacement of target rank RMA buffer
      int32_t target_disp = j - row_tiling[target_rank];

      // Get remote element
      MPI_Get( local_v.data() + local_offs, sizeof(T), MPI_BYTE, target_rank,
               target_disp, sizeof(T), MPI_BYTE, V_win );
    }
#else

    std::vector<decltype(unique_colind.begin())> rank_bnds;
    rank_bnds.emplace_back(unique_colind.begin());
    for (int ir = 0; ir < comm_size; ++ir) {
      rank_bnds.emplace_back(unique_colind.upper_bound(row_tiling[ir + 1] - 1));
    }

    for (int ir = 0; ir < comm_size; ++ir) {
      if (ir == comm_rank)
        for (auto uit = rank_bnds[ir]; uit != rank_bnds[ir + 1]; ++uit) {
          local_v[*uit] = V[*uit - row_tiling[ir]];
        }

      else
        for (auto uit = rank_bnds[ir]; uit != rank_bnds[ir + 1]; ++uit) {
          MPI_Get(local_v.data() + *uit, sizeof(T), MPI_BYTE, ir,
                  (*uit) - row_tiling[ir], sizeof(T), MPI_BYTE, V_win);
        }
    }

#endif /* 0 */
    MPI_Win_flush_local_all(V_win);

    // Do Matvec
    gespmbv(1, ALPHA, local_mat, local_v.data(), local_n, 1., AV, local_m);
  }
  MPI_Win_unlock_all(V_win);

  MPI_Win_free(&V_win);
}

}  // namespace sparsexx::spblas
