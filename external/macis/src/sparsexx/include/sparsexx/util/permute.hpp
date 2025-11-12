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
#include <sparsexx/util/submatrix.hpp>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif /* _OPENMP */

namespace sparsexx {

template <typename IndexType>
inline auto perm_from_part(int64_t npart, const std::vector<IndexType>& part) {
  std::vector<IndexType> perm(part.size());
  std::vector<IndexType> partptr(npart + 1);
  std::iota(perm.begin(), perm.end(), 0);

  partptr.at(0) = 0;
  auto cur_part = perm.begin();
  for (int64_t p = 0; p < npart - 1; ++p) {
    cur_part = std::stable_partition(cur_part, perm.end(),
                                     [&](auto i) { return part.at(i) == p; });
    partptr.at(p + 1) = std::distance(perm.begin(), cur_part);
  }
  partptr.at(npart) = perm.size();

  // Sanity check
  for (int64_t p = 0; p < npart; ++p) {
    if (std::any_of(perm.begin() + partptr.at(p),
                    perm.begin() + partptr.at(p + 1),
                    [&](auto i) { return part.at(i) != p; }))
      throw std::runtime_error("PERM NOT PARTITIONED");
  }

  return std::tuple(perm, partptr);
}

template <typename IndexType>
std::vector<IndexType> invert_perm(const std::vector<IndexType>& perm) {
  const auto n = perm.size();
  std::vector<IndexType> iperm(n);

  for (size_t i = 0; i < n; ++i) {
    iperm[perm[i]] = i;
  }

  return iperm;
}

enum class PermuteDirection { Forward, Backward };

/**
 * Permute a vector in place
 *
 * FORWARD
 * Y[P[i]] = X[i]
 *
 * BACKWARD
 * Y[i] = X[P[i]]
 */
template <typename T, typename IndexType>
void permute_vector(int64_t n, const T* V, const IndexType* P, T* Vp,
                    PermuteDirection dir) {
  if (dir == PermuteDirection::Forward)
    for (auto i = 0; i < n; ++i) {
      Vp[P[i]] = V[i];
    }
  else
    for (auto i = 0; i < n; ++i) {
      Vp[i] = V[P[i]];
    }
}

template <typename T, typename IndexType>
void permute_vector(int64_t n, T* V, const IndexType* P, PermuteDirection dir) {
  std::vector<T> Vp(n);
  permute_vector(n, V, P, Vp.data(), dir);
  std::copy_n(Vp.data(), n, V);
}

template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType, SpMatType> permute_rows(
    const SpMatType& A,
    const std::vector<detail::index_type_t<SpMatType>>& perm) {
  SpMatType Ap(A.m(), A.n(), A.nnz(), A.indexing());

  const auto m = A.m();
  const auto indexing = A.indexing();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto* Anz = A.nzval().data();

  auto* Aprp = Ap.rowptr().data();
  auto* Apci = Ap.colind().data();
  auto* Apnz = Ap.nzval().data();

  Aprp[0] = Ap.indexing();

  for (int64_t i = 0; i < m; ++i) {
    const auto A_j_st = Arp[perm[i]] - indexing;
    const auto A_j_en = Arp[perm[i] + 1] - indexing;
    const auto j_ext = A_j_en - A_j_st;

    Aprp[i + 1] = Aprp[i] + j_ext;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < m; ++i) {
    const auto A_j_st = Arp[perm[i]] - indexing;
    const auto A_j_en = Arp[perm[i] + 1] - indexing;
    const auto Ap_j_st = Aprp[i] - indexing;
    const auto Ap_j_en = Aprp[i + 1] - indexing;

    const auto j_ext = A_j_en - A_j_st;

    const auto* Aci_st = Aci + A_j_st;
    const auto* Aci_en = Aci + A_j_en;
    const auto* Anz_st = Anz + A_j_st;
    const auto* Anz_en = Anz + A_j_en;

    auto* Apci_st = Apci + Ap_j_st;
    auto* Apnz_st = Apnz + Ap_j_st;
    std::copy(Aci_st, Aci_en, Apci_st);
    std::copy(Anz_st, Anz_en, Apnz_st);
  }

  return Ap;
}

template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType, SpMatType> permute_cols(
    const SpMatType& A,
    const std::vector<detail::index_type_t<SpMatType>>& perm) {
  SpMatType Ap(A.m(), A.n(), A.nnz(), A.indexing());

  const auto m = A.m();
  const auto indexing = A.indexing();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto* Anz = A.nzval().data();

  auto* Aprp = Ap.rowptr().data();
  auto* Apci = Ap.colind().data();
  auto* Apnz = Ap.nzval().data();

  auto iperm = invert_perm(perm);
  std::vector<std::vector<int64_t>> ind_th(omp_get_max_threads());

  // Copy rowptr
  std::copy(A.rowptr().begin(), A.rowptr().end(), Ap.rowptr().begin());

#pragma omp parallel for
  for (int64_t i = 0; i < m; ++i) {
    const auto A_j_st = Arp[i] - indexing;
    const auto A_j_en = Arp[i + 1] - indexing;
    const auto j_ext = A_j_en - A_j_st;

    const auto* Aci_st = Aci + A_j_st;
    const auto* Anz_st = Anz + A_j_st;

    auto* Apci_st = Apci + A_j_st;
    auto* Apnz_st = Apnz + A_j_st;

    auto& ind = ind_th[omp_get_thread_num()];
    if (ind.size() != j_ext) {
      ind.resize(j_ext);
      std::iota(ind.begin(), ind.end(), 0);
    }

    std::sort(ind.begin(), ind.end(), [&](auto ii, auto jj) {
      return iperm[Aci_st[ii] - indexing] < iperm[Aci_st[jj] - indexing];
    });

    for (auto j = 0; j < j_ext; ++j) {
      Apci_st[j] = iperm[Aci_st[ind[j]] - indexing] + indexing;
      Apnz_st[j] = Anz_st[ind[j]];
    }
  }

  return Ap;
}

template <typename SpMatType>
detail::enable_if_csr_matrix_t<SpMatType, SpMatType> permute_rows_cols(
    const SpMatType& A,
    const std::vector<detail::index_type_t<SpMatType>>& rperm,
    const std::vector<detail::index_type_t<SpMatType>>& cperm) {
  SpMatType Ap(A.m(), A.n(), A.nnz(), A.indexing());

  const auto m = A.m();
  const auto indexing = A.indexing();
  const auto* Arp = A.rowptr().data();
  const auto* Aci = A.colind().data();
  const auto* Anz = A.nzval().data();

  auto* Aprp = Ap.rowptr().data();
  auto* Apci = Ap.colind().data();
  auto* Apnz = Ap.nzval().data();

  auto icperm = invert_perm(cperm);
  std::vector<std::vector<int64_t>> ind_th(omp_get_max_threads());

  Aprp[0] = Ap.indexing();
  for (int64_t i = 0; i < m; ++i) {
    const auto A_j_st = Arp[rperm[i]] - indexing;
    const auto A_j_en = Arp[rperm[i] + 1] - indexing;
    const auto j_ext = A_j_en - A_j_st;

    Aprp[i + 1] = Aprp[i] + j_ext;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < m; ++i) {
    const auto A_j_st = Arp[rperm[i]] - indexing;
    const auto A_j_en = Arp[rperm[i] + 1] - indexing;
    const auto Ap_j_st = Aprp[i] - indexing;
    // const auto Ap_j_en = Aprp[i+1]      - indexing;
    const auto j_ext = A_j_en - A_j_st;

    const auto* Aci_st = Aci + A_j_st;
    const auto* Anz_st = Anz + A_j_st;

    auto* Apci_st = Apci + Ap_j_st;
    auto* Apnz_st = Apnz + Ap_j_st;

    auto& ind = ind_th[omp_get_thread_num()];
    if ((int64_t)ind.size() != j_ext) {
      ind.resize(j_ext);
      std::iota(ind.begin(), ind.end(), 0);
    }

    std::sort(ind.begin(), ind.end(), [&](auto ii, auto jj) {
      return icperm[Aci_st[ii] - indexing] < icperm[Aci_st[jj] - indexing];
    });

    for (auto j = 0; j < j_ext; ++j) {
      Apci_st[j] = icperm[Aci_st[ind[j]] - indexing] + indexing;
      Apnz_st[j] = Anz_st[ind[j]];
    }
  }

  return Ap;
}

}  // namespace sparsexx
