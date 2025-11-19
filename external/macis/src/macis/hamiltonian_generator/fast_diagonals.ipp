/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator.hpp>

namespace macis {

template <typename Scalar>
double HamiltonianGeneratorBase<Scalar>::single_orbital_en(
    uint32_t orb, const std::vector<uint32_t>& ss_occ,
    const std::vector<uint32_t>& os_occ) const {
  // One electron component
  double orb_en = T_pq_(orb, orb);

  // Same-spin two-body term
  for (auto q : ss_occ) orb_en += G2_red_(orb, q) + G2_red_(q, orb);
  orb_en -= G2_red_(orb, orb);

  // Opposite-spin two-body term
  for (auto q : os_occ) orb_en += V2_red_(orb, q);

  return orb_en;
}

template <typename Scalar>
std::vector<double> HamiltonianGeneratorBase<Scalar>::single_orbital_ens(
    size_t norb, const std::vector<uint32_t>& ss_occ,
    const std::vector<uint32_t>& os_occ) const {
  std::vector<double> ens(norb);
  for (size_t i = 0; i < norb; ++i) {
    // One electron component
    auto e = T_pq_(i, i);

    // Same-spin two-body term
    for (auto q : ss_occ) e += G2_red_(i, q) + G2_red_(q, i);
    e -= G2_red_(i, i);

    // Opposite-spin two-body term
    for (auto q : os_occ) e += V2_red_(i, q);

    ens[i] = e;
  }

  return ens;
}

template <typename Scalar>
double HamiltonianGeneratorBase<Scalar>::fast_diag_single(
    // These refer to original determinant
    double hol_en, double par_en, uint32_t orb_hol, uint32_t orb_par,
    double orig_det_Hii) const {
  return orig_det_Hii + par_en - hol_en - G2_red_(orb_par, orb_hol) -
         G2_red_(orb_hol, orb_par);
}

template <typename Scalar>
double HamiltonianGeneratorBase<Scalar>::fast_diag_single(
    // These refer to original determinant
    const std::vector<uint32_t>& ss_occ, const std::vector<uint32_t>& os_occ,
    uint32_t orb_hol, uint32_t orb_par, double orig_det_Hii) const {
  const auto hol_en = single_orbital_en(orb_hol, ss_occ, os_occ);
  const auto par_en = single_orbital_en(orb_par, ss_occ, os_occ);
  return fast_diag_single(hol_en, par_en, orb_hol, orb_par, orig_det_Hii);
}

template <typename Scalar>
double HamiltonianGeneratorBase<Scalar>::fast_diag_ss_double(
    // These refer to original determinant
    double hol1_en, double hol2_en, double par1_en, double par2_en,
    uint32_t orb_hol1, uint32_t orb_hol2, uint32_t orb_par1, uint32_t orb_par2,
    double orig_det_Hii) const {
  return orig_det_Hii + par1_en + par2_en - hol1_en - hol2_en +
         G2_red_(orb_hol1, orb_hol2) + G2_red_(orb_hol2, orb_hol1) +
         G2_red_(orb_par1, orb_par2) + G2_red_(orb_par2, orb_par1) -
         G2_red_(orb_par1, orb_hol1) - G2_red_(orb_hol1, orb_par1) -
         G2_red_(orb_par2, orb_hol1) - G2_red_(orb_hol1, orb_par2) -
         G2_red_(orb_par1, orb_hol2) - G2_red_(orb_hol2, orb_par1) -
         G2_red_(orb_par2, orb_hol2) - G2_red_(orb_hol2, orb_par2);
}

template <typename Scalar>
double HamiltonianGeneratorBase<Scalar>::fast_diag_ss_double(
    // These refer to original determinant
    const std::vector<uint32_t>& ss_occ, const std::vector<uint32_t>& os_occ,
    uint32_t orb_hol1, uint32_t orb_hol2, uint32_t orb_par1, uint32_t orb_par2,
    double orig_det_Hii) const {
  auto hol1_en = single_orbital_en(orb_hol1, ss_occ, os_occ);
  auto hol2_en = single_orbital_en(orb_hol2, ss_occ, os_occ);
  auto par1_en = single_orbital_en(orb_par1, ss_occ, os_occ);
  auto par2_en = single_orbital_en(orb_par2, ss_occ, os_occ);
  return fast_diag_ss_double(hol1_en, hol2_en, par1_en, par2_en, orb_hol1,
                             orb_hol2, orb_par1, orb_par2, orig_det_Hii);
}

template <typename Scalar>
double HamiltonianGeneratorBase<Scalar>::fast_diag_os_double(
    // These refer to original determinant
    double en_holu, double en_hold, double en_paru, double en_pard,
    uint32_t orb_holu, uint32_t orb_hold, uint32_t orb_paru, uint32_t orb_pard,
    double orig_det_Hii) const {
  return orig_det_Hii + en_paru + en_pard - en_holu - en_hold +
         V2_red_(orb_holu, orb_hold) + V2_red_(orb_paru, orb_pard) -
         G2_red_(orb_paru, orb_holu) - G2_red_(orb_holu, orb_paru) -
         G2_red_(orb_pard, orb_hold) - G2_red_(orb_hold, orb_pard) -
         V2_red_(orb_paru, orb_hold) - V2_red_(orb_holu, orb_pard);
}

template <typename Scalar>
double HamiltonianGeneratorBase<Scalar>::fast_diag_os_double(
    // These refer to original determinant
    const std::vector<uint32_t>& ss_occ, const std::vector<uint32_t>& os_occ,
    uint32_t orb_holu, uint32_t orb_hold, uint32_t orb_paru, uint32_t orb_pard,
    double orig_det_Hii) const {
  auto holu_en = single_orbital_en(orb_holu, ss_occ, os_occ);
  auto hold_en = single_orbital_en(orb_hold, os_occ, ss_occ);
  auto paru_en = single_orbital_en(orb_paru, ss_occ, os_occ);
  auto pard_en = single_orbital_en(orb_pard, os_occ, ss_occ);
  return fast_diag_os_double(holu_en, hold_en, paru_en, pard_en, orb_holu,
                             orb_hold, orb_paru, orb_pard, orig_det_Hii);
}

}  // namespace macis
