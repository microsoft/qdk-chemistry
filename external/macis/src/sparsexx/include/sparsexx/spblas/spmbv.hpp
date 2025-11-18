/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <sparsexx/sparsexx_config.hpp>
#include <sparsexx/spblas/type_traits.hpp>

namespace sparsexx::spblas {

/**
 *  @brief Generic CSR sparse matrix - dense block vector product.
 *
 *  Computes the operation AV = ALPHA * A * V + BETA * AV where A is a sparse
 * matrix in Compressed Sparse Row (CSR) format and V, AV are dense block
 * vectors. This generic implementation works over any semiring and supports
 * OpenMP parallelization.
 *
 *  The algorithm iterates over each row of the sparse matrix and computes the
 *  matrix-vector product for each column of the block vector. The computation
 *  is performed using the standard CSR format with row pointers, column
 * indices, and non-zero values.
 *
 *  @tparam SpMatType Sparse matrix type that satisfies is_csr_matrix_v trait
 *  @tparam ALPHAT    Type of ALPHA scaling factor, must be convertible to
 * SpMatType::value_type
 *  @tparam BETAT     Type of BETA scaling factor, must be convertible to
 * SpMatType::value_type
 *
 *  @param[in]     K      Number of columns in the block vectors V and AV
 *  @param[in]     ALPHA  Scaling factor for the matrix-vector product A*V
 *  @param[in]     A      Input sparse matrix in CSR format
 *  @param[in]     V      Input dense block vector stored in column-major format
 *  @param[in]     LDV    Leading dimension of input vector V (must be >= A.n())
 *  @param[in]     BETA   Scaling factor for the existing values in AV
 *  @param[in,out] AV     Output dense block vector stored in column-major
 * format
 *  @param[in]     LDAV   Leading dimension of output vector AV (must be >=
 * A.m())
 *
 *  @note This function uses OpenMP parallelization when available.
 *  @note The matrix indexing (0-based or 1-based) is automatically handled.
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
std::enable_if_t<detail::spmbv_uses_generic_csr_v<SpMatType, ALPHAT, BETAT> >
gespmbv(int64_t K, ALPHAT ALPHA, const SpMatType& A,
        const typename SpMatType::value_type* V, int64_t LDV, BETAT BETA,
        typename SpMatType::value_type* AV, int64_t LDAV) {
  using value_type = typename SpMatType::value_type;

  const value_type alpha = ALPHA;
  const value_type beta = BETA;

  const auto M = A.m();
  const auto* Anz = A.nzval().data();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif /* _OPENMP */
  for (int64_t k = 0; k < K; ++k)
    for (int64_t i = 0; i < M; ++i) {
      const auto j_st = Arp[i] - indexing;
      const auto j_en = Arp[i + 1] - indexing;
      const auto j_ext = j_en - j_st;

      const auto* V_k = V + k * LDV - indexing;
      const auto* Anz_st = Anz + j_st;
      const auto* Aci_st = Aci + j_st;

      value_type av = 0.;
      for (int64_t j = 0; j < j_ext; ++j) {
        av += Anz_st[j] * V_k[Aci_st[j]];
      }

      AV[i + k * LDAV] = alpha * av + beta * AV[i + k * LDAV];
    }
}

/**
 *  @brief Generic COO sparse matrix - dense block vector product.
 *
 *  Computes the operation AV = ALPHA * A * V + BETA * AV where A is a sparse
 *  matrix in Coordinate (COO) format and V, AV are dense block vectors. This
 *  generic implementation works over any semiring and supports OpenMP
 *  parallelization.
 *
 *  The algorithm first prescales the output vector AV by BETA, then iterates
 *  through all non-zero elements of the sparse matrix to accumulate the
 *  matrix-vector product. Unlike CSR format, COO format stores each non-zero
 *  element with its explicit row and column indices.
 *
 *  @tparam SpMatType Sparse matrix type that satisfies is_coo_matrix_v trait
 *  @tparam ALPHAT    Type of ALPHA scaling factor, must be convertible to
 *  SpMatType::value_type
 *  @tparam BETAT     Type of BETA scaling factor, must be convertible to
 *  SpMatType::value_type
 *
 *  @param[in]     K      Number of columns in the block vectors V and AV
 *  @param[in]     ALPHA  Scaling factor for the matrix-vector product A*V
 *  @param[in]     A      Input sparse matrix in COO format
 *  @param[in]     V      Input dense block vector stored in column-major format
 *  @param[in]     LDV    Leading dimension of input vector V (must be >= A.n())
 *  @param[in]     BETA   Scaling factor for the existing values in AV
 *  @param[in,out] AV     Output dense block vector stored in column-major
 * format
 *  @param[in]     LDAV   Leading dimension of output vector AV (must be >=
 * A.m())
 *
 *  @note This function uses OpenMP parallelization for the prescaling step when
 * available.
 *  @note The matrix indexing (0-based or 1-based) is automatically handled.
 *  @note COO format may have worse cache performance compared to CSR for SpMV
 * operations.
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
std::enable_if_t<detail::spmbv_uses_generic_coo_v<SpMatType, ALPHAT, BETAT> >
gespmbv(int64_t K, ALPHAT ALPHA, const SpMatType& A,
        const typename SpMatType::value_type* V, int64_t LDV, BETAT BETA,
        typename SpMatType::value_type* AV, int64_t LDAV) {
  using value_type = typename SpMatType::value_type;
  using index_type = typename SpMatType::index_type;

  const value_type alpha = ALPHA;
  const value_type beta = BETA;

  const auto M = A.m();
  const auto N = A.n();
  const auto nnz = A.nnz();
  const auto* Anz = A.nzval().data();
  const auto* Ari = A.rowind().data();
  const auto* Aci = A.colind().data();
  const auto indexing = A.indexing();

// Prescale
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif /* _OPENMP */
  for (index_type k = 0; k < K; ++k)
    for (index_type i = 0; i < M; ++i) AV[i + k * LDAV] *= beta;

  for (index_type k = 0; k < K; ++k)
    for (index_type inz = 0; inz < nnz; ++inz) {
      const auto i = Ari[inz];
      const auto j = Aci[inz];
      AV[i + k * LDAV] += alpha * Anz[inz] * V[j + k * LDV];
    }
}

}  // namespace sparsexx::spblas
