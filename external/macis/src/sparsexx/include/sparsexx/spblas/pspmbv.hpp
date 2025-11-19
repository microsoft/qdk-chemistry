/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <chrono>
#include <numeric>
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/spblas/type_traits.hpp>
#include <sparsexx/util/permute.hpp>

namespace sparsexx::spblas {

namespace detail {
using namespace sparsexx::detail;

/**
 * @brief Creates an uninitialized array of type T with size n.
 *
 * This function allocates memory for an array of type T without initializing
 * the elements, which can be more efficient when the array will be populated
 * immediately after allocation.
 *
 * @tparam T The element type of the array
 * @param n The number of elements to allocate
 * @return std::unique_ptr<T[]> A unique pointer to the allocated array
 */
template <typename T>
auto no_init_array(size_t n) {
  return std::unique_ptr<T[]>(new T[n]);
}
}  // namespace detail

/**
 * @brief Communication information for parallel sparse matrix-vector
 * multiplication.
 *
 * This struct stores all necessary information for MPI communication patterns
 * required in distributed sparse matrix-vector multiplication (SpMV)
 * operations. It contains indices, offsets, and counts for both sending and
 * receiving data between MPI processes.
 *
 * @tparam IndexType The type used for matrix indices (typically int or long)
 */
template <typename IndexType>
struct spmv_info {
  using index_type = IndexType;  ///< Type alias for the index type

  MPI_Comm comm;  ///< MPI communicator for this SpMV operation

  std::vector<index_type>
      send_indices;  ///< Local indices of elements to send to remote processes
  std::vector<index_type> recv_indices;  ///< Remote indices of elements to
                                         ///< receive from remote processes
  std::vector<size_t>
      send_offsets;  ///< Byte offsets for data to send to each process
  std::vector<size_t>
      recv_offsets;  ///< Byte offsets for data to receive from each process
  std::vector<size_t>
      send_counts;  ///< Number of elements to send to each process
  std::vector<size_t>
      recv_counts;  ///< Number of elements to receive from each process

  /**
   * @brief Calculates the total communication volume across all MPI processes.
   *
   * This method computes the global communication volume for the SpMV operation
   * by summing the local communication volumes (send + receive indices) across
   * all processes and dividing by 2 to avoid double counting.
   *
   * @return size_t Total communication volume across all processes
   */
  inline size_t communication_volume() {
    size_t local_comm_vol = (send_indices.size() + recv_indices.size()) / 2;
    return detail::mpi_allreduce(local_comm_vol, MPI_SUM, comm);
  }

  /**
   * @brief Posts asynchronous MPI receives for remote data.
   *
   * This method initiates non-blocking MPI receive operations to receive
   * vector elements from remote processes. The received data will be stored
   * in the provided buffer X at the appropriate offsets.
   *
   * @tparam T The data type of the vector elements
   * @param X Pointer to the buffer where received data will be stored
   * @return std::vector<MPI_Request> Vector of MPI request handles for the
   * receive operations
   * @throws const char* If any receive count exceeds int32_t maximum value
   */
  template <typename T>
  std::vector<MPI_Request> post_remote_recv(T* X) const {
    std::vector<MPI_Request> reqs;
    int comm_size = recv_offsets.size();
    for (int i = 0; i < comm_size; ++i)
      if (recv_counts[i]) {
        if (recv_counts[i] > std::numeric_limits<int32_t>::max())
          throw "DIE IN RECV";
        reqs.emplace_back(
            detail::mpi_irecv(X + recv_offsets[i], recv_counts[i], i, 0, comm));
      }
    return reqs;
  }

  /**
   * @brief Posts asynchronous MPI sends for local data to remote processes.
   *
   * This method initiates non-blocking MPI send operations to send vector
   * elements to remote processes. The data is sent from the provided buffer X
   * starting at the appropriate offsets.
   *
   * @tparam T The data type of the vector elements
   * @param X Pointer to the buffer containing data to be sent
   * @return std::vector<MPI_Request> Vector of MPI request handles for the send
   * operations
   * @throws const char* If any send count exceeds int32_t maximum value
   */
  template <typename T>
  std::vector<MPI_Request> post_remote_send(const T* X) const {
    std::vector<MPI_Request> reqs;
    int comm_size = send_offsets.size();
    for (int i = 0; i < comm_size; ++i)
      if (send_counts[i]) {
        if (send_counts[i] > std::numeric_limits<int32_t>::max())
          throw "DIE IN SEND";
        reqs.emplace_back(
            detail::mpi_isend(X + send_offsets[i], send_counts[i], i, 0, comm));
      }
    return reqs;
  }
};

/**
 * @brief Generates communication information for distributed sparse
 * matrix-vector multiplication.
 *
 * This function analyzes a distributed sparse matrix and generates all
 * necessary communication patterns and data structures required for efficient
 * parallel SpMV operations. It determines which vector elements need to be
 * exchanged between MPI processes and sets up the communication buffers and
 * patterns.
 *
 * The function performs the following steps:
 * 1. Identifies off-diagonal column indices that require remote vector elements
 * 2. Determines which processes own the required vector elements
 * 3. Sets up send/receive patterns and buffers
 * 4. Exchanges index information between processes
 * 5. Computes offsets and counts for efficient data packing
 *
 * @tparam DistSpMatrixType Type of the distributed sparse matrix
 * @param A The distributed sparse matrix to analyze
 * @return spmv_info<index_type> Communication information structure containing
 *         all necessary data for parallel SpMV operations
 */
template <typename DistSpMatrixType>
auto generate_spmv_comm_info(const DistSpMatrixType& A) {
  using index_type = sparsexx::detail::index_type_t<DistSpMatrixType>;
  auto comm = A.comm();
  auto comm_size = sparsexx::detail::get_mpi_size(comm);
  auto comm_rank = sparsexx::detail::get_mpi_rank(comm);

  // Get unique column indices for local rows of A
  // excluding locally owned elements (i.e. off-diagonal col indices)
  auto off_diagonal_tile = A.off_diagonal_tile_ptr();
  std::set<index_type> unique_elements_set;
  if (off_diagonal_tile) {
    assert(off_diagonal_tile->indexing() == 0);
    unique_elements_set.insert(off_diagonal_tile->colind().begin(),
                               off_diagonal_tile->colind().end());
  }

  // Place unique col indices into contiguous memory
  std::vector<index_type> unique_elements(unique_elements_set.begin(),
                                          unique_elements_set.end());

  // Generate a list of elements that need to be sent by remote
  // MPI ranks to the current processs
  std::vector<std::vector<index_type>> recv_indices_by_rank(comm_size);
  auto uniq_it = unique_elements.begin();
  for (int i = 0; i < comm_size; ++i) {
    // Get row extent for rank-i
    auto [row_st, row_en] = A.row_bounds(i);

    // Find upper bound for row end
    auto next_uniq_it =
        std::lower_bound(uniq_it, unique_elements.end(), row_en);

    // Copy indices into local memory
    recv_indices_by_rank[i] = std::vector<index_type>(uniq_it, next_uniq_it);
    uniq_it = next_uniq_it;  // Update iterators
  }

  // Calculate element counts that will be received from remote processes
  std::vector<size_t> recv_counts(comm_size);
  std::transform(recv_indices_by_rank.begin(), recv_indices_by_rank.end(),
                 recv_counts.begin(),
                 [](const auto& idx) { return idx.size(); });

  // Ensure that local recv counts are zero
  assert(recv_counts[comm_rank] == 0);

  // Gather recv counts to remote ranks
  // This tells each rank the number of elements each remote process
  // expects to receive from the local process
  auto recv_counts_gathered = detail::mpi_allgather(recv_counts, comm);

  // Allocate memory to store the remote indices each remote process
  // expects to receive from the current process and post async
  // receives to receive those indices
  std::vector<std::vector<index_type>> send_indices_by_rank(comm_size);
  std::vector<MPI_Request> recv_reqs;
  for (int i = 0; i < comm_size; ++i)
    if (i != comm_rank) {
      auto nremote = recv_counts_gathered[comm_rank + i * comm_size];
      if (nremote) {
        send_indices_by_rank[i].resize(nremote);
        recv_reqs.emplace_back(
            detail::mpi_irecv(send_indices_by_rank[i], i, 0, comm));
      }
    }

  // Send to each remote process the indices it needs to send to the
  // current process
  std::vector<MPI_Request> send_reqs;
  for (auto i = 0; i < comm_size; ++i)
    if (recv_counts[i]) {
      send_reqs.emplace_back(
          detail::mpi_isend(recv_indices_by_rank[i], i, 0, comm));
    }

  // Wait on receives to complete
  detail::mpi_waitall_ignore_status(recv_reqs);

  // Calculate element counts that will be sent to each remote processes
  std::vector<size_t> send_counts(comm_size);
  std::transform(send_indices_by_rank.begin(), send_indices_by_rank.end(),
                 send_counts.begin(),
                 [](const auto& idx) { return idx.size(); });

  // Compute offsets for send/recv indices in contiguous memory
  std::vector<size_t> recv_offsets(comm_size), send_offsets(comm_size);
  std::exclusive_scan(send_counts.begin(), send_counts.end(),
                      send_offsets.begin(), 0);
  std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                      recv_offsets.begin(), 0);

  // Linearize the send/recv data structures
  size_t nrecv_indices =
      std::accumulate(recv_counts.begin(), recv_counts.end(), 0ul);
  size_t nsend_indices =
      std::accumulate(send_counts.begin(), send_counts.end(), 0ul);
  std::vector<index_type> send_indices, recv_indices;
  send_indices.reserve(nsend_indices);
  recv_indices.reserve(nrecv_indices);
  for (auto i = 0; i < comm_size; ++i) {
    send_indices.insert(send_indices.end(), send_indices_by_rank[i].begin(),
                        send_indices_by_rank[i].end());
    recv_indices.insert(recv_indices.end(), recv_indices_by_rank[i].begin(),
                        recv_indices_by_rank[i].end());
  }

  // Correct send indices to be relative to local row start
  const auto lrs = A.local_row_start();
  for (auto& i : send_indices) i -= lrs;

  // Wait for sends to complete to avoid race conditions
  detail::mpi_waitall_ignore_status(send_reqs);

  spmv_info<index_type> info;
  info.comm = comm;
  info.send_indices = std::move(send_indices);
  info.recv_indices = std::move(recv_indices);
  info.recv_offsets = std::move(recv_offsets);
  info.recv_counts = std::move(recv_counts);
  info.send_offsets = std::move(send_offsets);
  info.send_counts = std::move(send_counts);

  return info;
}

/**
 * @brief Performs parallel distributed sparse matrix-vector multiplication.
 *
 * This function implements the operation AV = ALPHA * A * V + BETA * AV for
 * distributed sparse matrices. It uses the provided communication information
 * to efficiently exchange vector elements between MPI processes and performs
 * both diagonal and off-diagonal matrix-vector multiplications.
 *
 * The algorithm follows these steps:
 * 1. Post asynchronous receives for remote vector elements
 * 2. Pack and send local vector elements to remote processes
 * 3. Perform diagonal tile matrix-vector multiplication
 * 4. Wait for remote data and unpack into contiguous buffer
 * 5. Perform off-diagonal tile matrix-vector multiplication
 * 6. Wait for all communications to complete
 *
 * @tparam DistSpMatType Type of the distributed sparse matrix
 * @tparam ScalarType Type of the scalar values (default: inferred from matrix)
 * @tparam IndexType Type of the matrix indices (default: inferred from matrix)
 *
 * @param ALPHA Scalar multiplier for the matrix-vector product A*V
 * @param A The distributed sparse matrix
 * @param V Pointer to the input vector (must be accessible on all processes)
 * @param BETA Scalar multiplier for the existing values in AV
 * @param AV Pointer to the output vector (result of ALPHA*A*V + BETA*AV)
 * @param spmv_info Communication information structure containing send/receive
 * patterns
 */
template <typename DistSpMatType,
          typename ScalarType = detail::value_type_t<DistSpMatType>,
          typename IndexType = detail::index_type_t<DistSpMatType>>
void pgespmv(detail::type_identity_t<ScalarType> ALPHA, const DistSpMatType& A,
             const detail::type_identity_t<ScalarType>* V,
             detail::type_identity_t<ScalarType> BETA,
             detail::type_identity_t<ScalarType>* AV,
             const spmv_info<detail::type_identity_t<IndexType>>& spmv_info) {
  using value_type = ScalarType;
  // using index_type = IndexType;

  // const auto M = A.m();
  const auto N = A.n();

  // auto comm = A.comm();
  // auto comm_size = detail::get_mpi_size(comm);
  // auto comm_rank = detail::get_mpi_rank(comm);

  const auto& recv_indices = spmv_info.recv_indices;
  const auto& send_indices = spmv_info.send_indices;

  /***** Initial Communication Part *****/

  // auto st_alloc = std::chrono::high_resolution_clock::now();
  //  Allocated packed buffers
  size_t nrecv_pack = recv_indices.size();
  size_t nsend_pack = send_indices.size();
  auto V_recv_pack = detail::no_init_array<value_type>(nrecv_pack);
  auto V_send_pack = detail::no_init_array<value_type>(nsend_pack);

  // Buffer for offdiagonal matvec

  // std::vector<value_type> V_remote( N );
  auto V_remote = detail::no_init_array<value_type>(N);
  // auto en_alloc = std::chrono::high_resolution_clock::now();

  // Post async recv's for remote data required for offdiagonal
  // matvec
  auto recv_reqs = spmv_info.post_remote_recv(V_recv_pack.get());

  // Pack data to send to remote processes
  // auto st_permb = std::chrono::high_resolution_clock::now();
  sparsexx::permute_vector(nsend_pack, V, send_indices.data(),
                           V_send_pack.get(),
                           sparsexx::PermuteDirection::Backward);
  // auto en_permb = std::chrono::high_resolution_clock::now();

  // Send data (async) to remote processes
  // auto st_send = std::chrono::high_resolution_clock::now();
  auto send_reqs = spmv_info.post_remote_send(V_send_pack.get());
  // auto en_send = std::chrono::high_resolution_clock::now();

  /***** Diagonal Matvec *****/
  // auto st_loc = std::chrono::high_resolution_clock::now();
  gespmbv(1, ALPHA, A.diagonal_tile(), V, N, BETA, AV, N);
  // auto en_loc = std::chrono::high_resolution_clock::now();

  // Wait for receives to complete
  // auto st_wait1 = std::chrono::high_resolution_clock::now();
  detail::mpi_waitall_ignore_status(recv_reqs);
  // auto en_wait1 = std::chrono::high_resolution_clock::now();

  // Unpack data into contiguous buffer
  // auto st_permf = std::chrono::high_resolution_clock::now();
  sparsexx::permute_vector(nrecv_pack, V_recv_pack.get(), recv_indices.data(),
                           V_remote.get(), sparsexx::PermuteDirection::Forward);
  // auto en_permf = std::chrono::high_resolution_clock::now();

  /***** Off-diagonal Matvec *****/
  // auto st_rem = std::chrono::high_resolution_clock::now();
  if (A.off_diagonal_tile_ptr())
    gespmbv(1, ALPHA, A.off_diagonal_tile(), V_remote.get(), N, 1., AV, N);
  // auto en_rem = std::chrono::high_resolution_clock::now();

  // Wait for all sends to complete to keep packed buffer in scope
  // auto st_wait2 = std::chrono::high_resolution_clock::now();
  detail::mpi_waitall_ignore_status(send_reqs);
  // auto en_wait2 = std::chrono::high_resolution_clock::now();

  // printf("ALLOC = %.2e PRB = %.2e PRF = %.2e SND = %.2e LOC = %.2e REM = %.2e
  // WAIT1 = %.2e WAIT2 = %.2e\n",
  //   std::chrono::duration<double,std::milli>(en_alloc - st_alloc).count(),
  //   std::chrono::duration<double,std::milli>(en_permb - st_permb).count(),
  //   std::chrono::duration<double,std::milli>(en_permf - st_permf).count(),
  //   std::chrono::duration<double,std::milli>(en_send - st_send).count(),
  //   std::chrono::duration<double,std::milli>(en_loc - st_loc).count(),
  //   std::chrono::duration<double,std::milli>(en_rem - st_rem).count(),
  //   std::chrono::duration<double,std::milli>(en_wait1 - st_wait1).count(),
  //   std::chrono::duration<double,std::milli>(en_wait2 - st_wait2).count());
}

}  // namespace sparsexx::spblas
