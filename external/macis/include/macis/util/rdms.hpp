/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>

namespace macis {

template <typename T, size_t N>
inline void rdm_contributions_4(wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex, T val,
                                rank4_span<T> trdm) {
  auto [o1, v1, o2, v2, sign] = doubles_sign_indices(bra, ket, ex);

  val *= sign * 0.5;
  trdm(v1, o1, v2, o2) += val;
  trdm(v2, o1, v1, o2) -= val;
  trdm(v1, o2, v2, o1) -= val;
  trdm(v2, o2, v1, o1) += val;
}

template <typename T, size_t N>
inline void rdm_contributions_22(wfn_t<N> bra_alpha, wfn_t<N> ket_alpha,
                                 wfn_t<N> ex_alpha, wfn_t<N> bra_beta,
                                 wfn_t<N> ket_beta, wfn_t<N> ex_beta, T val,
                                 rank4_span<T> trdm) {
  auto [o1, v1, sign_a] =
      single_excitation_sign_indices(bra_alpha, ket_alpha, ex_alpha);
  auto [o2, v2, sign_b] =
      single_excitation_sign_indices(bra_beta, ket_beta, ex_beta);
  auto sign = sign_a * sign_b;

  val *= sign * 0.5;
  trdm(v1, o1, v2, o2) += val;
  trdm(v2, o2, v1, o1) += val;
}

template <typename T, size_t N, typename IndexType>
inline void rdm_contributions_2(wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex,
                                const IndexType& bra_occ_alpha,
                                const IndexType& bra_occ_beta, T val,
                                matrix_span<T> ordm, rank4_span<T> trdm) {
  auto [o1, v1, sign] = single_excitation_sign_indices(bra, ket, ex);

  ordm(v1, o1) += sign * val;

  if (trdm.data_handle()) {
    val *= sign * 0.5;
    for (auto p : bra_occ_alpha) {
      trdm(v1, o1, p, p) += val;
      trdm(p, p, v1, o1) += val;
      trdm(v1, p, p, o1) -= val;
      trdm(p, o1, v1, p) -= val;
    }

    for (auto p : bra_occ_beta) {
      trdm(v1, o1, p, p) += val;
      trdm(p, p, v1, o1) += val;
    }
  }
}

template <typename T, typename IndexType>
inline void rdm_contributions_diag(const IndexType& occ_alpha,
                                   const IndexType& occ_beta, T val,
                                   matrix_span<T> ordm, rank4_span<T> trdm) {
  // One-electron piece
  for (auto p : occ_alpha) ordm(p, p) += val;
  for (auto p : occ_beta) ordm(p, p) += val;

  if (trdm.data_handle()) {
    val *= 0.5;
    // Same-spin two-body term
    for (auto q : occ_alpha)
      for (auto p : occ_alpha) {
        trdm(p, p, q, q) += val;
        trdm(p, q, p, q) -= val;
      }
    for (auto q : occ_beta)
      for (auto p : occ_beta) {
        trdm(p, p, q, q) += val;
        trdm(p, q, p, q) -= val;
      }

    // Opposite-spin two-body term
    for (auto q : occ_beta)
      for (auto p : occ_alpha) {
        trdm(p, p, q, q) += val;
        trdm(q, q, p, p) += val;
      }
  }
}

template <typename T, size_t N, typename IndexType>
inline void rdm_contributions(wfn_t<N> bra_alpha, wfn_t<N> ket_alpha,
                              wfn_t<N> ex_alpha, wfn_t<N> bra_beta,
                              wfn_t<N> ket_beta, wfn_t<N> ex_beta,
                              const IndexType& bra_occ_alpha,
                              const IndexType& bra_occ_beta, T val,
                              matrix_span<T> ordm, rank4_span<T> trdm) {
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  const uint32_t ex_alpha_count = wfn_traits::count(ex_alpha);
  const uint32_t ex_beta_count = wfn_traits::count(ex_beta);

  if ((ex_alpha_count + ex_beta_count) > 4) return;

  const auto trdm_ptr = trdm.data_handle();
  if (ex_alpha_count == 4 and trdm_ptr)
    rdm_contributions_4(bra_alpha, ket_alpha, ex_alpha, val, trdm);

  else if (ex_beta_count == 4 and trdm_ptr)
    rdm_contributions_4(bra_beta, ket_beta, ex_beta, val, trdm);

  else if (ex_alpha_count == 2 and ex_beta_count == 2 and trdm_ptr)
    rdm_contributions_22(bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta,
                         ex_beta, val, trdm);

  else if (ex_alpha_count == 2)
    rdm_contributions_2(bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
                        bra_occ_beta, val, ordm, trdm);

  else if (ex_beta_count == 2)
    rdm_contributions_2(bra_beta, ket_beta, ex_beta, bra_occ_beta,
                        bra_occ_alpha, val, ordm, trdm);

  else
    rdm_contributions_diag(bra_occ_alpha, bra_occ_beta, val, ordm, trdm);
}

}  // namespace macis
