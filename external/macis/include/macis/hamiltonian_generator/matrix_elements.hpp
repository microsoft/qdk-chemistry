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

/**
 * @brief Compute matrix element between two full determinants
 *
 * This method computes the Hamiltonian matrix element ⟨bra|H|ket⟩ between
 * two full determinants by decomposing them into alpha and beta components
 * and delegating to the spin-resolved matrix element calculation.
 *
 * @tparam WfnType Wavefunction type defining determinant representation
 * @param bra The bra determinant
 * @param ket The ket determinant
 * @return Matrix element value
 */
template <typename WfnType>
double HamiltonianGenerator<WfnType>::matrix_element(full_det_t bra,
                                                     full_det_t ket) const {
  using wfn_traits = wavefunction_traits<WfnType>;
  auto bra_alpha = wfn_traits::alpha_string(bra);
  auto ket_alpha = wfn_traits::alpha_string(ket);
  auto bra_beta = wfn_traits::beta_string(bra);
  auto ket_beta = wfn_traits::beta_string(ket);

  auto ex_alpha = bra_alpha ^ ket_alpha;
  auto ex_beta = bra_beta ^ ket_beta;

  auto bra_occ_alpha = bits_to_indices(bra_alpha);
  auto bra_occ_beta = bits_to_indices(bra_beta);

  return matrix_element(bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta,
                        ex_beta, bra_occ_alpha, bra_occ_beta);
}

/**
 * @brief Compute matrix element between spin-resolved determinants with
 * excitation analysis
 *
 * This is the core matrix element computation method that analyzes the
 * excitation patterns between bra and ket determinants and dispatches to
 * specialized methods based on the number and type of excitations.
 *
 * @tparam WfnType Wavefunction type defining determinant representation
 * @param bra_alpha Alpha component of bra determinant
 * @param ket_alpha Alpha component of ket determinant
 * @param ex_alpha Excitation pattern in alpha spin
 * @param bra_beta Beta component of bra determinant
 * @param ket_beta Beta component of ket determinant
 * @param ex_beta Excitation pattern in beta spin
 * @param bra_occ_alpha Occupied alpha orbitals in bra
 * @param bra_occ_beta Occupied beta orbitals in bra
 * @return Matrix element value
 */
template <typename WfnType>
double HamiltonianGenerator<WfnType>::matrix_element(
    spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha,
    spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta,
    const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta) const {
  using spin_wfn_traits = wavefunction_traits<spin_det_t>;
  const uint32_t ex_alpha_count = spin_wfn_traits::count(ex_alpha);
  const uint32_t ex_beta_count = spin_wfn_traits::count(ex_beta);

  if ((ex_alpha_count + ex_beta_count) > 4) return 0.;

  if (ex_alpha_count == 4)
    return matrix_element_4(bra_alpha, ket_alpha, ex_alpha);

  else if (ex_beta_count == 4)
    return matrix_element_4(bra_beta, ket_beta, ex_beta);

  else if (ex_alpha_count == 2 and ex_beta_count == 2)
    return matrix_element_22(bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta,
                             ex_beta);

  else if (ex_alpha_count == 2)
    return matrix_element_2(bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha,
                            bra_occ_beta);

  else if (ex_beta_count == 2)
    return matrix_element_2(bra_beta, ket_beta, ex_beta, bra_occ_beta,
                            bra_occ_alpha);

  else
    return matrix_element_diag(bra_occ_alpha, bra_occ_beta);
}

/**
 * @brief Compute matrix element for same-spin double excitation (4 excitations
 * in one spin)
 *
 * Calculates the matrix element for determinants that differ by a double
 * excitation within the same spin channel. Uses antisymmetrized integrals
 * G_pqrs.
 *
 * @tparam WfnType Wavefunction type defining determinant representation
 * @param bra Bra spin determinant
 * @param ket Ket spin determinant
 * @param ex Excitation pattern
 * @return Matrix element value
 */
template <typename WfnType>
double HamiltonianGenerator<WfnType>::matrix_element_4(spin_det_t bra,
                                                       spin_det_t ket,
                                                       spin_det_t ex) const {
  auto [o1, v1, o2, v2, sign] = doubles_sign_indices(bra, ket, ex);

  return sign * G_pqrs_(v1, o1, v2, o2);
}

/**
 * @brief Compute matrix element for opposite-spin double excitation (2+2
 * excitations)
 *
 * Calculates the matrix element for determinants that differ by single
 * excitations in both alpha and beta spin channels. Uses Coulomb integrals
 * V_pqrs.
 *
 * @tparam WfnType Wavefunction type defining determinant representation
 * @param bra_alpha Alpha component of bra determinant
 * @param ket_alpha Alpha component of ket determinant
 * @param ex_alpha Alpha excitation pattern
 * @param bra_beta Beta component of bra determinant
 * @param ket_beta Beta component of ket determinant
 * @param ex_beta Beta excitation pattern
 * @return Matrix element value
 */
template <typename WfnType>
double HamiltonianGenerator<WfnType>::matrix_element_22(
    spin_det_t bra_alpha, spin_det_t ket_alpha, spin_det_t ex_alpha,
    spin_det_t bra_beta, spin_det_t ket_beta, spin_det_t ex_beta) const {
  auto [o1, v1, sign_a] =
      single_excitation_sign_indices(bra_alpha, ket_alpha, ex_alpha);
  auto [o2, v2, sign_b] =
      single_excitation_sign_indices(bra_beta, ket_beta, ex_beta);
  auto sign = sign_a * sign_b;

  return sign * V_pqrs_(v1, o1, v2, o2);
}

/**
 * @brief Compute matrix element for single excitation (2 excitations in one
 * spin)
 *
 * Calculates the matrix element for determinants that differ by a single
 * excitation. Includes one-electron terms and contracted two-electron terms
 * with other occupied orbitals.
 *
 * @tparam WfnType Wavefunction type defining determinant representation
 * @param bra Bra spin determinant
 * @param ket Ket spin determinant
 * @param ex Excitation pattern
 * @param bra_occ_alpha Occupied alpha orbitals in bra
 * @param bra_occ_beta Occupied beta orbitals in bra
 * @return Matrix element value
 */
template <typename WfnType>
double HamiltonianGenerator<WfnType>::matrix_element_2(
    spin_det_t bra, spin_det_t ket, spin_det_t ex,
    const std::vector<uint32_t>& bra_occ_alpha,
    const std::vector<uint32_t>& bra_occ_beta) const {
  auto [o1, v1, sign] = single_excitation_sign_indices(bra, ket, ex);

  double h_el = T_pq_(v1, o1);

  const double* G_red_ov = &G_red_(0, v1, o1);
  for (auto p : bra_occ_alpha) {
    h_el += G_red_ov[p];
  }

  const double* V_red_ov = &V_red_(0, v1, o1);
  for (auto p : bra_occ_beta) {
    h_el += V_red_ov[p];
  }

  return sign * h_el;
}

/**
 * @brief Compute diagonal matrix element (no excitations)
 *
 * Calculates the diagonal matrix element for identical determinants.
 * Includes one-electron terms and all two-electron interactions between
 * occupied orbitals.
 *
 * @tparam WfnType Wavefunction type defining determinant representation
 * @param occ_alpha Occupied alpha orbitals
 * @param occ_beta Occupied beta orbitals
 * @return Diagonal matrix element value
 */
template <typename WfnType>
double HamiltonianGenerator<WfnType>::matrix_element_diag(
    const std::vector<uint32_t>& occ_alpha,
    const std::vector<uint32_t>& occ_beta) const {
  double h_el = 0;

  // One-electron piece
  for (auto p : occ_alpha) h_el += T_pq_(p, p);
  for (auto p : occ_beta) h_el += T_pq_(p, p);

  // Same-spin two-body term
  for (auto q : occ_alpha)
    for (auto p : occ_alpha) {
      h_el += G2_red_(p, q);
    }
  for (auto q : occ_beta)
    for (auto p : occ_beta) {
      h_el += G2_red_(p, q);
    }

  // Opposite-spin two-body term
  for (auto q : occ_beta)
    for (auto p : occ_alpha) {
      h_el += V2_red_(p, q);
    }

  return h_el;
}
}  // namespace macis
