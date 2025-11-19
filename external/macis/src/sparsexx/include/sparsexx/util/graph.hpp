/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/permute.hpp>
#include <sparsexx/util/submatrix.hpp>

namespace sparsexx {

namespace detail {

/**
 * @brief Performs k-way graph partitioning using METIS algorithm
 *
 * This function partitions a graph into k parts using the METIS library's
 * k-way partitioning algorithm. The graph is represented in compressed
 * sparse row (CSR) format.
 *
 * @tparam IndexType The integer type used for indices (e.g., int32_t, int64_t)
 * @param _nvert Number of vertices in the graph
 * @param _npart Number of partitions to create
 * @param _xadj Array of size _nvert+1 containing adjacency structure row
 * pointers
 * @param _adjncy Array containing column indices of adjacent vertices
 * @param _part Output vector of size _nvert where partition assignments are
 * stored
 */
template <typename IndexType>
void metis_kway_partitioning(int64_t _nvert, int64_t _npart, IndexType* _xadj,
                             IndexType* _adjncy, std::vector<IndexType>& _part);

extern template void metis_kway_partitioning(int64_t _nvert, int64_t _npart,
                                             int32_t* _xadj, int32_t* _adjncy,
                                             std::vector<int32_t>& _part);

extern template void metis_kway_partitioning(int64_t _nvert, int64_t _npart,
                                             int64_t* _xadj, int64_t* _adjncy,
                                             std::vector<int64_t>& _part);

}  // namespace detail

/**
 * @brief Extracts the adjacency graph from a sparse matrix
 *
 * This function converts a sparse matrix in CSR format to an adjacency graph
 * representation by removing diagonal elements.
 *
 * @tparam SpMatType The sparse matrix type (must be CSR matrix)
 * @param A The input sparse matrix in CSR format
 * @param ret_indexing The indexing scheme for the returned arrays (0 or
 * 1-based)
 * @return A tuple containing:
 *         - rowptr: Row pointer array for the adjacency graph
 *         - colind: Column indices array for the adjacency graph
 *
 * @note The function removes diagonal elements from the matrix to create
 *       a proper adjacency representation
 */
template <typename SpMatType>
detail::enable_if_csr_matrix_t<
    SpMatType, std::tuple<std::vector<detail::index_type_t<SpMatType>>,
                          std::vector<detail::index_type_t<SpMatType>>>>
extract_adjacency(const SpMatType& A, int ret_indexing = 0) {
  const auto M = A.m();

  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

  using index_t = detail::index_type_t<SpMatType>;
  std::vector<index_t> rowptr(M + 1), colind(A.nnz());

  rowptr[0] = ret_indexing;
  auto* ci_st = colind.data();

  // Get number of diagonal elements
  int64_t ndiag = 0;
  for (index_t i = 0; i < M; ++i) {
    const auto j_st = Arp[i] - indexing;
    const auto j_en = Arp[i + 1] - indexing;

    const auto* Aci_st = Aci + j_st;
    const auto* Aci_en = Aci + j_en;

    const auto diag_it = std::find(Aci_st, Aci_en, i + indexing);

    // Copy values to left of diagonal
    ci_st = std::copy(Aci_st, diag_it, ci_st);

    auto nrow = std::distance(Aci_st, Aci_en);
    if (diag_it != Aci_en) {
      ndiag++;
      rowptr[i + 1] = rowptr[i] + nrow - 1;

      // Copy values right of diagonal
      ci_st = std::copy(diag_it + 1, Aci_en, ci_st);
    } else
      rowptr[i + 1] = rowptr[i] + nrow;
  }

  colind.resize(A.nnz() - ndiag);
  for (auto& i : colind) i -= (indexing - ret_indexing);

  return std::tuple(rowptr, colind);
}

/**
 * @brief Partitions a sparse matrix using k-way graph partitioning
 *
 * This function performs k-way partitioning of a sparse matrix by first
 * extracting its adjacency graph and then applying the METIS k-way
 * partitioning algorithm. The matrix is treated as the adjacency matrix
 * of a graph.
 *
 * @tparam SpMatType The sparse matrix type (must be CSR matrix)
 * @param npart Number of partitions to create (must be >= 2)
 * @param A The input sparse matrix representing the graph
 * @return A vector of size A.m() containing partition assignments for each
 * vertex
 *
 * @throws std::runtime_error if npart < 2
 *
 * @note The function uses 0-based indexing internally and returns partition
 *       indices starting from 0
 */
template <typename SpMatType>
auto kway_partition(int64_t npart, const SpMatType& A) {
  auto [adj_rp, adj_ci] = extract_adjacency(A, 0);

  if (npart < 2) throw std::runtime_error("KWayPart only works for K > 1");

  std::vector<int32_t> part(A.m());

  detail::metis_kway_partitioning(A.m(), npart, adj_rp.data(), adj_ci.data(),
                                  part);

  return part;
}

}  // namespace sparsexx
