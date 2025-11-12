/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <sparsexx/sparsexx_config.hpp>
#if SPARSEXX_ENABLE_MKL

#include <cassert>
#include <iostream>

#include "mkl_exceptions.hpp"
#include "mkl_types.hpp"
namespace sparsexx::detail::mkl {

struct sparse_wrapper_pimpl {
  struct matrix_descr descr;
  sparse_matrix_t mat;

  ~sparse_wrapper_pimpl() noexcept { mkl_sparse_destroy(mat); }

  void optimize() { mkl_sparse_optimize(mat); }
};

template <typename T>
enable_if_mkl_type_supported_t<T, std::unique_ptr<sparse_wrapper_pimpl>>
mkl_csr_factory(int64_t indexing, int64_t m, int64_t n, int_type* rowptr,
                int_type* colind, T* nzval) {
  assert(indexing == 0 or indexing == 1);
  auto mkl_indexing =
      (indexing == 0) ? SPARSE_INDEX_BASE_ZERO : SPARSE_INDEX_BASE_ONE;

  auto pimpl = std::make_unique<sparse_wrapper_pimpl>();

  sparse_status_t stat;
  if constexpr (std::is_same_v<T, float>)
    stat = mkl_sparse_s_create_csr(&pimpl->mat, mkl_indexing, m, n, rowptr,
                                   rowptr + 1, colind, nzval);
  else if constexpr (std::is_same_v<T, double>)
    stat = mkl_sparse_d_create_csr(&pimpl->mat, mkl_indexing, m, n, rowptr,
                                   rowptr + 1, colind, nzval);
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    stat = mkl_sparse_c_create_csr(&pimpl->mat, mkl_indexing, m, n, rowptr,
                                   rowptr + 1, colind, nzval);
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    stat = mkl_sparse_z_create_csr(&pimpl->mat, mkl_indexing, m, n, rowptr,
                                   rowptr + 1, colind, nzval);

  if (stat != SPARSE_STATUS_SUCCESS) throw mkl_sparse_exception(stat);

  pimpl->descr.type = SPARSE_MATRIX_TYPE_GENERAL;  // TODO handle symmetric

  return pimpl;
}

template <typename T>
enable_if_mkl_type_supported_t<T, std::unique_ptr<sparse_wrapper_pimpl>>
mkl_coo_factory(int64_t indexing, int64_t m, int64_t n, int64_t nnz,
                int_type* rowind, int_type* colind, T* nzval) {
  assert(indexing == 0 or indexing == 1);
  auto mkl_indexing =
      (indexing == 0) ? SPARSE_INDEX_BASE_ZERO : SPARSE_INDEX_BASE_ONE;

  auto pimpl = std::make_unique<sparse_wrapper_pimpl>();

  sparse_status_t stat;
  if constexpr (std::is_same_v<T, float>)
    stat = mkl_sparse_s_create_coo(&pimpl->mat, mkl_indexing, m, n, nnz, rowind,
                                   colind, nzval);
  else if constexpr (std::is_same_v<T, double>)
    stat = mkl_sparse_d_create_coo(&pimpl->mat, mkl_indexing, m, n, nnz, rowind,
                                   colind, nzval);
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    stat = mkl_sparse_c_create_coo(&pimpl->mat, mkl_indexing, m, n, nnz, rowind,
                                   colind, nzval);
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    stat = mkl_sparse_z_create_coo(&pimpl->mat, mkl_indexing, m, n, nnz, rowind,
                                   colind, nzval);

  if (stat != SPARSE_STATUS_SUCCESS) throw mkl_sparse_exception(stat);

  pimpl->descr.type = SPARSE_MATRIX_TYPE_GENERAL;  // TODO handle symmetric

  return pimpl;
}

template <typename SpMatType>
enable_if_mkl_compatible_t<SpMatType, std::unique_ptr<sparse_wrapper_pimpl>>
sparse_matrix_factory(SpMatType* ptr) {
  if constexpr (is_csr_matrix_v<SpMatType>)
    return mkl_csr_factory(ptr->indexing(), ptr->m(), ptr->n(),
                           ptr->rowptr().data(), ptr->colind().data(),
                           ptr->nzval().data());
  else if constexpr (is_coo_matrix_v<SpMatType>)
    return mkl_coo_factory(ptr->indexing(), ptr->m(), ptr->n(), ptr->nnz(),
                           ptr->rowind().data(), ptr->colind().data(),
                           ptr->nzval().data());
  else
    throw std::runtime_error("Unrecognized Sparse Matrix Type in MKL Factory");
}

}  // namespace sparsexx::detail::mkl
#endif /* SPARSEXX_ENABLE_MKL */
