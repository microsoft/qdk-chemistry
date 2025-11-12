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
 *  AV = ALPHA * A * V + BETA * AV
 *
 *  Generic implementation over any semiring.
 *
 *  @tparam SpMatType Sparse matrix type s.t. is_csr_matrix_v is true
 *  @tparam ALPHAT    Type of ALPHA, must be convertible to
 * SpMatType::value_type
 *  @tparam BETAT     Type of BETA, must be convertible to SpMatType::value_type
 *
 *  @param[in]     K      Number of columns in V/AV
 *  @param[in]     ALPHA  First scaling factor
 *  @param[in]     A      Sparse matrix in CSR format
 *  @param[in]     V      Input block vector stored in column major format
 *  @param[in]     LDV    Leading dimension of V
 *  @param[in]     BETA   Second scaling factor
 *  @param[in/out] AV     Output block vector stored in column major format
 *  @param[in]     LDAV   Leading dimension of AV
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
 *  AV = ALPHA * A * V + BETA * AV
 *
 *  Generic implementation over any semiring.
 *
 *  @tparam SpMatType Sparse matrix type s.t. is_csr_matrix_v is true
 *  @tparam ALPHAT    Type of ALPHA, must be convertible to
 * SpMatType::value_type
 *  @tparam BETAT     Type of BETA, must be convertible to SpMatType::value_type
 *
 *  @param[in]     K      Number of columns in V/AV
 *  @param[in]     ALPHA  First scaling factor
 *  @param[in]     A      Sparse matrix in COO format
 *  @param[in]     V      Input block vector stored in column major format
 *  @param[in]     LDV    Leading dimension of V
 *  @param[in]     BETA   Second scaling factor
 *  @param[in/out] AV     Output block vector stored in column major format
 *  @param[in]     LDAV   Leading dimension of AV
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

#if SPARSEXX_ENABLE_MKL
/**
 *  @brief Optimized sparse matrix - dense block vector product.
 *
 *  AV = ALPHA * A * V + BETA * AV
 *
 *  Optimized MKL implementation over any semiring and any MKL supported
 *  sparse matrix format.
 *
 *  @tparam SpMatType Sparse matrix type s.t. is_mkl_matrix_v is true
 *  @tparam ALPHAT    Type of ALPHA, must be convertible to
 * SpMatType::value_type
 *  @tparam BETAT     Type of BETA, must be convertible to SpMatType::value_type
 *
 *  @param[in]     K      Number of columns in V/AV
 *  @param[in]     ALPHA  First scaling factor
 *  @param[in]     A      Sparse matrix stored in any MKL compatible format
 *  @param[in]     V      Input block vector stored in column major format
 *  @param[in]     LDV    Leading dimension of V
 *  @param[in]     BETA   Second scaling factor
 *  @param[in/out] AV     Output block vector stored in column major format
 *  @param[in]     LDAV   Leading dimension of AV
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
std::enable_if_t<detail::spmbv_uses_mkl_v<SpMatType, ALPHAT, BETAT> > gespmbv(
    int64_t K, ALPHAT ALPHA, const SpMatType& A,
    const typename SpMatType::value_type* V, int64_t LDV, BETAT BETA,
    typename SpMatType::value_type* AV, int64_t LDAV) {
  sparse_status_t stat;

  using value_type = typename SpMatType::value_type;
  static_assert(std::is_same_v<value_type, double>,
                "MKL SPMBV is only implemented for double precision");

  value_type alpha = ALPHA;
  value_type beta = BETA;

  stat = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.handle(),
                         A.descr(), SPARSE_LAYOUT_COLUMN_MAJOR, V, K, LDV, beta,
                         AV, LDAV);

  if (stat != SPARSE_STATUS_SUCCESS)
    throw sparsexx::detail::mkl::mkl_sparse_exception(stat);
}
#endif /* SPARSEXX_ENABLE_MKL */

}  // namespace sparsexx::spblas
