/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>

namespace macis {

/**
 * @brief Compute two-particle reduced density matrix contributions for double
 * excitations within the same spin channel.
 *
 * This function processes a double excitation (four-body operator) within a
 * single spin channel and accumulates the corresponding contributions to the
 * two-particle reduced density matrix (2-RDM). The function extracts orbital
 * indices from the excitation patterns and applies the appropriate
 * antisymmetrization factors.
 *
 * @tparam symm Boolean flag indicating whether to apply symmetrization
 * (true) or not (false)
 * @tparam T Numeric type for the matrix elements (typically double or
 * complex<double>)
 * @tparam N Size parameter for the wavefunction representation
 *
 * @param[in] bra Bra wavefunction state
 * @param[in] ket Ket wavefunction state
 * @param[in] ex Excitation pattern describing the double excitation
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] trdm Four-dimensional span representing the 2-RDM tensor to be
 * updated
 */
template <bool symm, typename T, size_t N>
inline void rdm_contributions_4(wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex, T val,
                                rank4_span<T> trdm) {
  if (not trdm.data_handle()) return;

  auto [o1, v1, o2, v2, sign] = doubles_sign_indices(bra, ket, ex);

  // same spin
  val *= sign * 0.5;
#pragma omp atomic
  trdm(v1, o1, v2, o2) += val;
#pragma omp atomic
  trdm(v2, o1, v1, o2) -= val;
#pragma omp atomic
  trdm(v1, o2, v2, o1) -= val;
#pragma omp atomic
  trdm(v2, o2, v1, o1) += val;

  // same spin
  if constexpr (symm) {
#pragma omp atomic
    trdm(o2, v2, o1, v1) += val;
#pragma omp atomic
    trdm(o2, v1, o1, v2) -= val;
#pragma omp atomic
    trdm(o1, v2, o2, v1) -= val;
#pragma omp atomic
    trdm(o1, v1, o2, v2) += val;
  }
}

/**
 * @brief Compute 2-RDM contributions for mixed-spin double excitations
 * (alpha-beta).
 *
 * This function handles the case where there are simultaneous single
 * excitations in both alpha and beta spin channels, resulting in a mixed-spin
 * double excitation. The contributions are computed without antisymmetrization
 * between different spin channels.
 *
 * @tparam symm Boolean flag indicating whether to apply symmetrization
 * (true) or not (false)
 * @tparam T Numeric type for the matrix elements
 * @tparam N Size parameter for the wavefunction representation
 *
 * @param[in] bra_alpha Bra wavefunction for alpha spin channel
 * @param[in] ket_alpha Ket wavefunction for alpha spin channel
 * @param[in] ex_alpha Excitation pattern for alpha spin channel
 * @param[in] bra_beta Bra wavefunction for beta spin channel
 * @param[in] ket_beta Ket wavefunction for beta spin channel
 * @param[in] ex_beta Excitation pattern for beta spin channel
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] trdm Four-dimensional span representing the 2-RDM tensor to be
 * updated
 */
template <bool symm, typename T, size_t N>
inline void rdm_contributions_22(wfn_t<N> bra_alpha, wfn_t<N> ket_alpha,
                                 wfn_t<N> ex_alpha, wfn_t<N> bra_beta,
                                 wfn_t<N> ket_beta, wfn_t<N> ex_beta, T val,
                                 rank4_span<T> trdm) {
  if (not trdm.data_handle()) return;
  auto [o1, v1, sign_a] =
      single_excitation_sign_indices(bra_alpha, ket_alpha, ex_alpha);
  auto [o2, v2, sign_b] =
      single_excitation_sign_indices(bra_beta, ket_beta, ex_beta);
  auto sign = sign_a * sign_b;

  // opposite spin
  val *= sign * 0.5;
#pragma omp atomic
  trdm(v1, o1, v2, o2) += val;
#pragma omp atomic
  trdm(v2, o2, v1, o1) += val;

  // opposite spin
  if constexpr (symm) {
#pragma omp atomic
    trdm(o2, v2, o1, v1) += val;
#pragma omp atomic
    trdm(o1, v1, o2, v2) += val;
  }
}

/**
 * @brief Compute spin-dependent 2-RDM contributions for mixed-spin double
 * excitations (alpha-beta).
 *
 * This function is the spin-resolved version of rdm_contributions_22, handling
 * simultaneous single excitations in both alpha and beta spin channels. It
 * specifically updates the alpha-alpha-beta-beta component of the
 * spin-dependent 2-RDM tensor.
 *
 * @tparam symm Boolean flag indicating whether to apply symmetrization
 * (true) or not (false)
 * @tparam T Numeric type for the matrix elements
 * @tparam N Size parameter for the wavefunction representation
 *
 * @param[in] bra_alpha Bra wavefunction for alpha spin channel
 * @param[in] ket_alpha Ket wavefunction for alpha spin channel
 * @param[in] ex_alpha Excitation pattern for alpha spin channel
 * @param[in] bra_beta Bra wavefunction for beta spin channel
 * @param[in] ket_beta Ket wavefunction for beta spin channel
 * @param[in] ex_beta Excitation pattern for beta spin channel
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] trdm_aabb Alpha-alpha-beta-beta 2-RDM tensor component
 */
template <bool symm, typename T, size_t N>
inline void rdm_contributions_22_spin_dep(wfn_t<N> bra_alpha,
                                          wfn_t<N> ket_alpha, wfn_t<N> ex_alpha,
                                          wfn_t<N> bra_beta, wfn_t<N> ket_beta,
                                          wfn_t<N> ex_beta, T val,
                                          rank4_span<T> trdm_aabb) {
  if (not trdm_aabb.data_handle()) return;
  auto [o2, v2, sign_b] =
      single_excitation_sign_indices(bra_alpha, ket_alpha, ex_alpha);
  auto [o1, v1, sign_a] =
      single_excitation_sign_indices(bra_beta, ket_beta, ex_beta);
  auto sign = sign_a * sign_b;

  val *= sign * 0.5;
#pragma omp atomic
  trdm_aabb(v1, o1, v2, o2) += val;

  if constexpr (symm) {
#pragma omp atomic
    trdm_aabb(o1, v1, o2, v2) += val;
  }
}

/**
 * @brief Compute both 1-RDM and 2-RDM contributions for single excitations.
 *
 * This function processes single excitations and computes their contributions
 * to both the one-particle reduced density matrix (1-RDM) and the two-particle
 * reduced density matrix (2-RDM). The 2-RDM contributions arise from
 * contractions with the occupied orbitals in the reference state.
 *
 * @tparam symm Boolean flag indicating whether to apply symmetrization
 * @tparam T Numeric type for the matrix elements
 * @tparam N Size parameter for the wavefunction representation
 * @tparam IndexType Container type for orbital indices (e.g., std::vector<int>)
 *
 * @param[in] bra Bra wavefunction state
 * @param[in] ket Ket wavefunction state
 * @param[in] ex Excitation pattern describing the single excitation
 * @param[in] bra_occ_alpha Indices of occupied alpha orbitals in the bra state
 * @param[in] bra_occ_beta Indices of occupied beta orbitals in the bra state
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] ordm Two-dimensional span representing the 1-RDM matrix to be
 * updated
 * @param[in,out] trdm Four-dimensional span representing the 2-RDM tensor to be
 * updated
 */
template <bool symm, typename T, size_t N, typename IndexType>
inline void rdm_contributions_2(wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex,
                                const IndexType& bra_occ_alpha,
                                const IndexType& bra_occ_beta, T val,
                                matrix_span<T> ordm, rank4_span<T> trdm) {
  auto [o1, v1, sign] = single_excitation_sign_indices(bra, ket, ex);

  if (ordm.data_handle()) {
#pragma omp atomic
    ordm(v1, o1) += sign * val;
    if constexpr (symm) {
#pragma omp atomic
      ordm(o1, v1) += sign * val;
    }
  }

  if (trdm.data_handle()) {
    // same spin
    val *= sign * 0.5;
    for (auto p : bra_occ_alpha) {
#pragma omp atomic
      trdm(v1, o1, p, p) += val;
#pragma omp atomic
      trdm(p, p, v1, o1) += val;
#pragma omp atomic
      trdm(v1, p, p, o1) -= val;
#pragma omp atomic
      trdm(p, o1, v1, p) -= val;
    }

    // same spin
    if constexpr (symm) {
      for (auto p : bra_occ_alpha) {
#pragma omp atomic
        trdm(p, p, o1, v1) += val;
#pragma omp atomic
        trdm(o1, v1, p, p) += val;
#pragma omp atomic
        trdm(o1, p, p, v1) -= val;
#pragma omp atomic
        trdm(p, v1, o1, p) -= val;
      }
    }

    // opposite spin
    for (auto p : bra_occ_beta) {
#pragma omp atomic
      trdm(v1, o1, p, p) += val;
#pragma omp atomic
      trdm(p, p, v1, o1) += val;
    }

    // opposite spin
    if constexpr (symm) {
      for (auto p : bra_occ_beta) {
#pragma omp atomic
        trdm(o1, v1, p, p) += val;
#pragma omp atomic
        trdm(p, p, o1, v1) += val;
      }
    }
  }
}

/**
 * @brief Compute spin-dependent 1-RDM and 2-RDM contributions for single
 * excitations.
 *
 * This function is the spin-resolved version of rdm_contributions_2, computing
 * contributions to spin-separated reduced density matrices from single
 * excitations. It handles both same-spin (ss) and opposite-spin (os)
 * contributions separately.
 *
 * @tparam symm Boolean flag indicating whether to apply symmetrization
 * @tparam transpose Boolean flag for transposing indices in opposite-spin terms
 * @tparam T Numeric type for the matrix elements
 * @tparam N Size parameter for the wavefunction representation
 * @tparam IndexType Container type for orbital indices
 *
 * @param[in] bra Bra wavefunction state
 * @param[in] ket Ket wavefunction state
 * @param[in] ex Excitation pattern describing the single excitation
 * @param[in] bra_occ_ss Indices of occupied same-spin orbitals in the bra state
 * @param[in] bra_occ_os Indices of occupied opposite-spin orbitals in the bra
 * state
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] ordm_ss Same-spin 1-RDM matrix component
 * @param[in,out] trdm_ss Same-spin 2-RDM tensor component
 * @param[in,out] trdm_os Opposite-spin 2-RDM tensor component
 */
template <bool symm, bool transpose, typename T, size_t N, typename IndexType>
inline void rdm_contributions_2_spin_dep(
    wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex, const IndexType& bra_occ_ss,
    const IndexType& bra_occ_os, T val, matrix_span<T> ordm_ss,
    rank4_span<T> trdm_ss, rank4_span<T> trdm_os) {
  auto [o1, v1, sign] = single_excitation_sign_indices(bra, ket, ex);

  if (ordm_ss.data_handle()) {
#pragma omp atomic
    ordm_ss(v1, o1) += sign * val;
    if constexpr (symm) {
#pragma omp atomic
      ordm_ss(o1, v1) += sign * val;
    }
  }

  if (trdm_ss.data_handle()) {
    val *= sign * 0.5;
    for (auto p : bra_occ_ss) {
#pragma omp atomic
      trdm_ss(v1, o1, p, p) += val;
#pragma omp atomic
      trdm_ss(p, p, v1, o1) += val;
#pragma omp atomic
      trdm_ss(v1, p, p, o1) -= val;
#pragma omp atomic
      trdm_ss(p, o1, v1, p) -= val;
    }

    if constexpr (symm) {
      for (auto p : bra_occ_ss) {
#pragma omp atomic
        trdm_ss(p, p, o1, v1) += val;
#pragma omp atomic
        trdm_ss(o1, v1, p, p) += val;
#pragma omp atomic
        trdm_ss(o1, p, p, v1) -= val;
#pragma omp atomic
        trdm_ss(p, v1, o1, p) -= val;
      }
    }
  }

  if (trdm_os.data_handle()) {
    for (auto p : bra_occ_os) {
      if constexpr (transpose) {
#pragma omp atomic
        trdm_os(v1, o1, p, p) += val;
      } else {
#pragma omp atomic
        trdm_os(p, p, v1, o1) += val;
      }
    }

    if constexpr (symm) {
      for (auto p : bra_occ_os) {
        if constexpr (transpose) {
#pragma omp atomic
          trdm_os(o1, v1, p, p) += val;
        } else {
#pragma omp atomic
          trdm_os(p, p, o1, v1) += val;
        }
      }
    }
  }
}

/**
 * @brief Compute diagonal contributions to 1-RDM and 2-RDM from the reference
 * state.
 *
 * This function computes the contributions to the reduced density matrices from
 * the diagonal matrix elements (no excitations). These represent the
 * contributions from the occupied orbitals in the reference determinant or
 * configuration.
 *
 * @tparam T Numeric type for the matrix elements
 * @tparam IndexType Container type for orbital indices
 *
 * @param[in] occ_alpha Indices of occupied alpha orbitals
 * @param[in] occ_beta Indices of occupied beta orbitals
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] ordm Two-dimensional span representing the 1-RDM matrix to be
 * updated
 * @param[in,out] trdm Four-dimensional span representing the 2-RDM tensor to be
 * updated
 */
template <typename T, typename IndexType>
inline void rdm_contributions_diag(const IndexType& occ_alpha,
                                   const IndexType& occ_beta, T val,
                                   matrix_span<T> ordm, rank4_span<T> trdm) {
  // One-electron piece
  if (ordm.data_handle()) {
    for (auto p : occ_alpha) {
#pragma omp atomic
      ordm(p, p) += val;
    }
    for (auto p : occ_beta) {
#pragma omp atomic
      ordm(p, p) += val;
    }
  }

  if (trdm.data_handle()) {
    val *= 0.5;
    // same spin
    for (auto q : occ_alpha)
      for (auto p : occ_alpha) {
#pragma omp atomic
        trdm(p, p, q, q) += val;
#pragma omp atomic
        trdm(p, q, q, p) -= val;
      }

    // same spin
    for (auto q : occ_beta)
      for (auto p : occ_beta) {
#pragma omp atomic
        trdm(p, p, q, q) += val;
#pragma omp atomic
        trdm(p, q, q, p) -= val;
      }

    // opposite spin
    for (auto q : occ_beta)
      for (auto p : occ_alpha) {
#pragma omp atomic
        trdm(p, p, q, q) += val;
#pragma omp atomic
        trdm(q, q, p, p) += val;
      }
  }
}

/**
 * @brief Compute spin-dependent diagonal contributions to 1-RDM and 2-RDM from
 * the reference state.
 *
 * This function is the spin-resolved version of rdm_contributions_diag,
 * computing diagonal contributions to spin-separated reduced density matrices.
 * These represent contributions from occupied orbitals in the reference
 * determinant, separated into alpha-alpha, beta-beta, and alpha-beta
 * components.
 *
 * @tparam T Numeric type for the matrix elements
 * @tparam IndexType Container type for orbital indices
 *
 * @param[in] occ_alpha Indices of occupied alpha orbitals
 * @param[in] occ_beta Indices of occupied beta orbitals
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] ordm_aa Alpha-alpha 1-RDM matrix component
 * @param[in,out] ordm_bb Beta-beta 1-RDM matrix component
 * @param[in,out] trdm_aaaa Alpha-alpha-alpha-alpha 2-RDM tensor component
 * @param[in,out] trdm_bbbb Beta-beta-beta-beta 2-RDM tensor component
 * @param[in,out] trdm_aabb Alpha-alpha-beta-beta 2-RDM tensor component
 */
template <typename T, typename IndexType>
inline void rdm_contributions_diag_spin_dep(
    const IndexType& occ_alpha, const IndexType& occ_beta, T val,
    matrix_span<T> ordm_aa, matrix_span<T> ordm_bb, rank4_span<T> trdm_aaaa,
    rank4_span<T> trdm_bbbb, rank4_span<T> trdm_aabb) {
  // One-electron piece
  if (ordm_aa.data_handle()) {
    for (auto p : occ_alpha) {
#pragma omp atomic
      ordm_aa(p, p) += val;
    }
  }
  if (ordm_bb.data_handle()) {
    for (auto p : occ_beta) {
#pragma omp atomic
      ordm_bb(p, p) += val;
    }
  }

  val *= 0.5;
  if (trdm_aaaa.data_handle()) {
    // same spin
    for (auto q : occ_alpha)
      for (auto p : occ_alpha) {
#pragma omp atomic
        trdm_aaaa(p, p, q, q) += val;
#pragma omp atomic
        trdm_aaaa(p, q, q, p) -= val;
      }
  }

  if (trdm_bbbb.data_handle()) {
    for (auto q : occ_beta)
      for (auto p : occ_beta) {
#pragma omp atomic
        trdm_bbbb(p, p, q, q) += val;
#pragma omp atomic
        trdm_bbbb(p, q, q, p) -= val;
      }
  }

  if (trdm_aabb.data_handle()) {
    // opposite spin
    for (auto q : occ_beta)
      for (auto p : occ_alpha) {
#pragma omp atomic
        trdm_aabb(q, q, p, p) += val;
      }
  }
}

/**
 * @brief Main dispatcher function for computing RDM contributions based on
 * excitation level.
 *
 * This function analyzes the excitation patterns in both alpha and beta spin
 * channels and dispatches to the appropriate specialized function to compute
 * the contributions to the reduced density matrices. It supports up to double
 * excitations (4 electrons total).
 *
 * @tparam symm Boolean flag indicating whether to apply symmetrization
 * @tparam T Numeric type for the matrix elements
 * @tparam N Size parameter for the wavefunction representation
 * @tparam IndexType Container type for orbital indices
 *
 * @param[in] bra_alpha Bra wavefunction for alpha spin channel
 * @param[in] ket_alpha Ket wavefunction for alpha spin channel
 * @param[in] ex_alpha Excitation pattern for alpha spin channel
 * @param[in] bra_beta Bra wavefunction for beta spin channel
 * @param[in] ket_beta Ket wavefunction for beta spin channel
 * @param[in] ex_beta Excitation pattern for beta spin channel
 * @param[in] bra_occ_alpha Indices of occupied alpha orbitals in the bra state
 * @param[in] bra_occ_beta Indices of occupied beta orbitals in the bra state
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] ordm Two-dimensional span representing the 1-RDM matrix to be
 * updated
 * @param[in,out] trdm Four-dimensional span representing the 2-RDM tensor to be
 * updated
 */
template <bool symm, typename T, size_t N, typename IndexType>
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

  if (ex_alpha_count == 4) {
    rdm_contributions_4<symm>(bra_alpha, ket_alpha, ex_alpha, val, trdm);
  } else if (ex_beta_count == 4) {
    rdm_contributions_4<symm>(bra_beta, ket_beta, ex_beta, val, trdm);
  } else if (ex_alpha_count == 2 and ex_beta_count == 2) {
    rdm_contributions_22<symm>(bra_alpha, ket_alpha, ex_alpha, bra_beta,
                               ket_beta, ex_beta, val, trdm);
  } else if (ex_alpha_count == 2) {
    rdm_contributions_2<symm>(bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
                              bra_occ_beta, val, ordm, trdm);
  } else if (ex_beta_count == 2) {
    rdm_contributions_2<symm>(bra_beta, ket_beta, ex_beta, bra_occ_beta,
                              bra_occ_alpha, val, ordm, trdm);
  } else {
    rdm_contributions_diag(bra_occ_alpha, bra_occ_beta, val, ordm, trdm);
  }
}

/**
 * @brief Main dispatcher function for computing spin-dependent RDM
 * contributions based on excitation level.
 *
 * This function is the spin-resolved version of rdm_contributions, analyzing
 * excitation patterns in both alpha and beta spin channels and dispatching to
 * the appropriate specialized spin-dependent function. It computes
 * contributions to spin-separated reduced density matrices supporting up to
 * double excitations.
 *
 * @tparam symm Boolean flag indicating whether to apply symmetrization
 * @tparam T Numeric type for the matrix elements
 * @tparam N Size parameter for the wavefunction representation
 * @tparam IndexType Container type for orbital indices
 *
 * @param[in] bra_alpha Bra wavefunction for alpha spin channel
 * @param[in] ket_alpha Ket wavefunction for alpha spin channel
 * @param[in] ex_alpha Excitation pattern for alpha spin channel
 * @param[in] bra_beta Bra wavefunction for beta spin channel
 * @param[in] ket_beta Ket wavefunction for beta spin channel
 * @param[in] ex_beta Excitation pattern for beta spin channel
 * @param[in] bra_occ_alpha Indices of occupied alpha orbitals in the bra state
 * @param[in] bra_occ_beta Indices of occupied beta orbitals in the bra state
 * @param[in] val Amplitude/coefficient value for this contribution
 * @param[in,out] ordm_aa Alpha-alpha 1-RDM matrix component
 * @param[in,out] ordm_bb Beta-beta 1-RDM matrix component
 * @param[in,out] trdm_aaaa Alpha-alpha-alpha-alpha 2-RDM tensor component
 * @param[in,out] trdm_bbbb Beta-beta-beta-beta 2-RDM tensor component
 * @param[in,out] trdm_aabb Alpha-alpha-beta-beta 2-RDM tensor component
 */
template <bool symm, typename T, size_t N, typename IndexType>
inline void rdm_contributions_spin_dep(
    wfn_t<N> bra_alpha, wfn_t<N> ket_alpha, wfn_t<N> ex_alpha,
    wfn_t<N> bra_beta, wfn_t<N> ket_beta, wfn_t<N> ex_beta,
    const IndexType& bra_occ_alpha, const IndexType& bra_occ_beta, T val,
    matrix_span<T> ordm_aa, matrix_span<T> ordm_bb, rank4_span<T> trdm_aaaa,
    rank4_span<T> trdm_bbbb, rank4_span<T> trdm_aabb) {
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  const uint32_t ex_alpha_count = wfn_traits::count(ex_alpha);
  const uint32_t ex_beta_count = wfn_traits::count(ex_beta);

  if ((ex_alpha_count + ex_beta_count) > 4) return;

  if (ex_alpha_count == 4) {
    rdm_contributions_4<symm>(bra_alpha, ket_alpha, ex_alpha, val, trdm_aaaa);
  } else if (ex_beta_count == 4) {
    rdm_contributions_4<symm>(bra_beta, ket_beta, ex_beta, val, trdm_bbbb);
  } else if (ex_alpha_count == 2 and ex_beta_count == 2) {
    rdm_contributions_22_spin_dep<symm>(bra_alpha, ket_alpha, ex_alpha,
                                        bra_beta, ket_beta, ex_beta, val,
                                        trdm_aabb);
  } else if (ex_alpha_count == 2) {
    rdm_contributions_2_spin_dep<symm, false>(bra_alpha, ket_alpha, ex_alpha,
                                              bra_occ_alpha, bra_occ_beta, val,
                                              ordm_aa, trdm_aaaa, trdm_aabb);
  } else if (ex_beta_count == 2) {
    rdm_contributions_2_spin_dep<symm, true>(bra_beta, ket_beta, ex_beta,
                                             bra_occ_beta, bra_occ_alpha, val,
                                             ordm_bb, trdm_bbbb, trdm_aabb);
  } else {
    rdm_contributions_diag_spin_dep(bra_occ_alpha, bra_occ_beta, val, ordm_aa,
                                    ordm_bb, trdm_aaaa, trdm_bbbb, trdm_aabb);
  }
}

}  // namespace macis
