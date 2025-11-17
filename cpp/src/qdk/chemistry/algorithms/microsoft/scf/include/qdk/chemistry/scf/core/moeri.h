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
 *  (pn|lk) = C(p,m) * (mn|lk) [first quarter - customization point]
 *  (pq|lk) = C(q,n) * (pn|lk) [second quarter]
 *  (pq|rk) = C(r,l) * (pq|lk) [third quarter]
 *  (pq|rs) = C(s,k) * (pq|rk) [fourth quarter - final result]
 *
 *  Leverages the ERI::quater_trans to perform the first quarter transformation
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
   *  @brief Compute MO ERIs incore
   *
   *  @param[in]  nb Number of basis functions
   *  @param[in]  nt Number of vectors in the MO space
   *  @param[in]  C  Transformation coefficients (row major)
   *  @param[out] out Output MO ERIs (row major)
   */
  void compute(size_t nb, size_t nt, const double* C, double* out);

 private:
  std::shared_ptr<ERI> eri_;  ///< ERI instance
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  std::shared_ptr<cutensor::TensorHandle> handle_;  ///< cuTensor instance
#endif
};
}  // namespace qdk::chemistry::scf
