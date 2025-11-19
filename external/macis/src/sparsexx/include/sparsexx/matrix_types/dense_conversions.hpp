/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <stdexcept>

#include "coo_matrix.hpp"
#include "csr_matrix.hpp"

namespace sparsexx {

/**
 * @brief Converts a CSR sparse matrix to dense format.
 *
 * This function converts a CSR (Compressed Sparse Row) matrix to dense
 * column-major format by extracting non-zero elements and placing them at their
 * corresponding positions in the dense matrix.
 *
 * @tparam Args Template parameter pack for CSR matrix template arguments
 *
 * @param A The CSR matrix to convert
 * @param A_dense Pointer to pre-allocated dense matrix storage (column-major)
 * @param LDAD Leading dimension of the dense matrix (must be >= number of rows)
 *
 * @throws std::runtime_error if M > LDAD (insufficient leading dimension)
 *
 * @note Performance considerations:
 *       - Time complexity: O(nnz)
 *       - Memory access pattern may not be cache-optimal
 *       - Dense matrix should be zero-initialized before calling
 *
 * @warning This function only writes non-zero elements; zero elements are not
 * explicitly written
 */
template <typename... Args>
void convert_to_dense(const csr_matrix<Args...>& A,
                      typename csr_matrix<Args...>::value_type* A_dense,
                      int64_t LDAD) {
  const int64_t M = A.m();

  if (M > LDAD) throw std::runtime_error("M > LDAD");

  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

  for (int64_t i = 0; i < M; ++i) {
    const auto j_st = Arp[i] - indexing;
    const auto j_en = Arp[i + 1] - indexing;
    const auto j_ext = j_en - j_st;

    auto* Ad_i = A_dense + i - indexing * LDAD;

    const auto* Anz_st = Anz + j_st;
    const auto* Aci_st = Aci + j_st;
    for (int64_t j = 0; j < j_ext; ++j) Ad_i[Aci_st[j] * LDAD] = Anz_st[j];
  }
}

/**
 * @brief Converts a COO sparse matrix to dense format
 *
 * This function converts a COO (Coordinate) matrix to dense column-major format
 * by directly mapping each triplet (row, column, value) to its position in the
 * dense matrix.
 *
 * @tparam Args Template parameter pack for COO matrix template arguments
 *
 * @param A The COO matrix to convert
 * @param A_dense Pointer to pre-allocated dense matrix storage (column-major)
 * @param LDAD Leading dimension of the dense matrix (must be >= number of rows)
 *
 * @throws std::runtime_error if M > LDAD (insufficient leading dimension)
 *
 * @note Performance considerations:
 *       - Time complexity: O(nnz) with parallel execution
 *       - Memory access pattern is random (cache-unfriendly)
 *       - Parallel efficiency depends on nnz distribution
 *       - No race conditions as each triplet maps to unique position
 *
 * @warning This function only writes non-zero elements; zero elements are not
 * explicitly written
 * @warning Race conditions may occur if the COO matrix contains duplicate (i,j)
 * entries
 */
template <typename... Args>
void convert_to_dense(const coo_matrix<Args...>& A,
                      typename coo_matrix<Args...>::value_type* A_dense,
                      int64_t LDAD) {
  const int64_t M = A.m();
  // const int64_t N = A.n();

  if (M > LDAD) throw std::runtime_error("M > LDAD");

  const auto* Anz = A.nzval().data();
  const auto* Ari = A.rowind().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();
  const auto nnz = A.nnz();

#pragma omp parallel for
  for (int64_t i = 0; i < nnz; ++i) {
    A_dense[Ari[i] + Aci[i] * LDAD] = Anz[i];
  }
}

}  // namespace sparsexx
