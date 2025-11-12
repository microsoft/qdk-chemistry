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

template <typename... Args>
void convert_to_dense(const csr_matrix<Args...>& A,
                      typename csr_matrix<Args...>::value_type* A_dense,
                      int64_t LDAD) {
  const int64_t M = A.m();
  // const int64_t N = A.n();

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
