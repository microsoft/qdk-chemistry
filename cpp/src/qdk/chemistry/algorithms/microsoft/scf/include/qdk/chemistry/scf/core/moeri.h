// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/eri.h>

#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cutensor_utils.h>
#endif

namespace qdk::chemistry::scf {
/**
 *  @brief A class to handle the evaluation of MO integrals given an
 *  AO ERI implementation.
 *
 *  (pn|lk) = Ci(p,m) * (mn|lk) [first quarter - customization point]
 *  (pq|lk) = Cj(q,n) * (pn|lk) [second quarter]
 *  (pq|rk) = Ck(r,l) * (pq|lk) [third quarter]
 *  (pq|rs) = Cl(s,k) * (pq|rk) [fourth quarter - final result]
 *
 *  Leverages the ERI::quarter_trans to perform the first quarter transformation
 *  and performs the remainder transformations via cuTensor.
 */
class MOERI {
 public:
  MOERI() = delete;
  ~MOERI() noexcept;

  /**
   * @brief Construct an MOERI instance given an ERI instance
   *
   * @param[in] eri shared_ptr to a valid ERI instance
   */
  MOERI(std::shared_ptr<ERI> eri);

  /**
   *  @brief Compute MO ERIs incore with a single transformation matrix
   *
   *  @param[in]  nao  Number of atomic orbitals
   *  @param[in]  nt  Number of vectors in the MO space
   *  @param[in]  C  First quarter transformation coefficients (row major)
   *  @param[out] out Output MO ERIs (row major)
   */
  void compute(size_t nao, size_t nt, const double* C, double* out) {
    compute(nao, nt, C, C, C, C, out);
  }

  /**
   *  @brief Compute MO ERIs incore with four different transformation matrices.
   *
   *  Note that the resulting vector is sorted column major, meaning that access
   *  to MO integrals (i,j,k,l) happens as i + j*n + k*n*n + l*n*n*n.
   *
   *  If desired, a row major result can be obtained by passing matrices in
   *  reverse order (l,k,j,i).
   *
   *  @param[in]  nao  Number of atomic orbitals
   *  @param[in]  nt  Number of vectors in the MO space
   *  @param[in]  Ci  First quarter transformation coefficients (row major)
   *  @param[in]  Cj  Second quarter transformation coefficients (row major)
   *  @param[in]  Ck  Third quarter transformation coefficients (row major)
   *  @param[in]  Cl  Fourth quarter transformation coefficients (row major)
   *  @param[out] out Output MO ERIs (row major)
   */
  void compute(size_t nao, size_t nt, const double* Ci, const double* Cj,
               const double* Ck, const double* Cl, double* out);

  /**
   *  @brief Compute MO ERIs with different-sized transformation matrices per
   *  index.
   *
   *  Generalized 4-quarter AO→MO integral transform where each coefficient
   *  matrix may have a different number of MO vectors:
   *
   *  (pn|lk) = Ci(p,m) * (mn|lk), Ci is (nao × ni), p ∈ [0, ni)
   *  (pq|lk) = Cj(q,n) * (pn|lk), Cj is (nao × nj), q ∈ [0, nj)
   *  (pq|rk) = Ck(r,l) * (pq|lk), Ck is (nao × nk), r ∈ [0, nk)
   *  (pq|rs) = Cl(s,k) * (pq|rk), Cl is (nao × nl), s ∈ [0, nl)
   *
   *  Output is column major: out[p + q*ni + r*ni*nj + s*ni*nj*nk]
   *
   *  @param[in]  nao  Number of atomic orbitals
   *  @param[in]  ni   Number of MO vectors for first index
   *  @param[in]  Ci   First quarter transformation coefficients (row major,
   *                   nao × ni)
   *  @param[in]  nj   Number of MO vectors for second index
   *  @param[in]  Cj   Second quarter transformation coefficients (row major,
   *                   nao × nj)
   *  @param[in]  nk   Number of MO vectors for third index
   *  @param[in]  Ck   Third quarter transformation coefficients (row major,
   *                   nao × nk)
   *  @param[in]  nl   Number of MO vectors for fourth index
   *  @param[in]  Cl   Fourth quarter transformation coefficients (row major,
   *                   nao × nl)
   *  @param[out] out  Output MO ERIs (ni × nj × nk × nl, column major)
   */
  void compute(size_t nao, size_t ni, const double* Ci, size_t nj,
               const double* Cj, size_t nk, const double* Ck, size_t nl,
               const double* Cl, double* out);

 private:
  std::shared_ptr<ERI> eri_;  ///< ERI instance
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  std::shared_ptr<cutensor::TensorHandle> handle_;  ///< cuTensor instance
#endif
};
}  // namespace qdk::chemistry::scf
