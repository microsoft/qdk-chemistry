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
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>

namespace macis {

/**
 * @brief Structure representing ASCI (Adaptive Sampling Configuration
 * Interaction) score contributions for candidate determinants
 *
 * For each determinant in the reference state (P-space), connected determinants
 * (Q-space) scored according to their ability to lower the energy via
 * perturbative corrections are stored in this structure. In the ASCI procedure,
 * the Q-space determinants are generatated on the fly by looping over all
 * possible connections to the P-space and three pieces of information are
 * stored for each connection:
 * 1. The Q-space determinant
 * 2. <Q|H|P> * c_P, the product of the Hamiltonian matrix element and the
 * coefficient of the P-space determinant
 * 3. The diagonal Hamiltonian matrix element for the Q-space determinant:
 * <Q|H|Q>
 *
 * @tparam WfnT Wavefunction type representing the quantum state
 */
template <typename WfnT>
struct asci_contrib {
  /// @brief The excited determinant state |Q>
  WfnT state;
  /// @brief Product of coefficient and matrix element with the generating
  /// P-space state
  double c_times_matel;
  /// @brief <Q|H|Q> diagonal matrix element
  double h_diag;

  /**
   * @brief Calculate the ratio value for perturbative selection
   * @return The ratio c_times_matel / h_diag
   */
  auto rv() const { return c_times_matel / h_diag; }

  /**
   * @brief Calculate the second-order perturbation theory contribution
   * @return The PT2 energy contribution (rv() * c_times_matel)
   */
  auto pt2() const { return rv() * c_times_matel; }
};

/// @brief Container type for storing multiple ASCI contributions
template <typename WfnT>
using asci_contrib_container = std::vector<asci_contrib<WfnT>>;

/**
 * @brief Generate single excitation ASCI contributions for a given spin
 *
 * This function computes matrix elements and contributions for all possible
 * single excitations from occupied to virtual orbitals of the specified spin.
 * It calculates the Hamiltonian matrix elements including one-electron and
 * two-electron terms.
 *
 * @tparam Sigma The spin type (Alpha or Beta) for the excitations
 * @tparam WfnType Full wavefunction type
 * @tparam SpinWfnType Spin-specific wavefunction type
 * @param[in] coeff Coefficient of the reference determinant
 * @param[in] state_full Full determinant state (both alpha and beta spins)
 * @param[in] state_same Spin component being excited
 * @param[in] occ_same Occupied orbital indices for the same spin
 * @param[in] vir_same Virtual orbital indices for the same spin
 * @param[in] occ_othr Occupied orbital indices for the opposite spin
 * @param[in] eps_same Orbital energies for the same spin
 * @param[in] T_pq One-electron integral matrix
 * @param[in] LDT Leading dimension of T_pq matrix
 * @param[in] G_kpq Same-spin two-electron integral tensor
 * @param[in] LDG Leading dimension of G_kpq tensor
 * @param[in] V_kpq Opposite-spin two-electron integral tensor
 * @param[in] LDV Leading dimension of V_kpq tensor
 * @param[in] h_el_tol Threshold for matrix element magnitude
 * @param[in] root_diag Root diagonal correction term
 * @param[in] E0 Reference energy
 * @param[in] ham_gen Hamiltonian generator for fast diagonal evaluation
 * @param[in,out] asci_contributions Container to store the computed
 * contributions
 */
template <Spin Sigma, typename WfnType, typename SpinWfnType>
void append_singles_asci_contributions(
    double coeff, WfnType state_full, SpinWfnType state_same,
    const std::vector<uint32_t>& occ_same,
    const std::vector<uint32_t>& vir_same,
    const std::vector<uint32_t>& occ_othr, const double* eps_same,
    const double* T_pq, const size_t LDT, const double* G_kpq, const size_t LDG,
    const double* V_kpq, const size_t LDV, double h_el_tol, double root_diag,
    double E0, const HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  const auto LDG2 = LDG * LDG;
  const auto LDV2 = LDV * LDV;
  for (auto i : occ_same)
    for (auto a : vir_same) {
      // Compute single excitation matrix element
      double h_el = T_pq[a + i * LDT];
      const double* G_ov = G_kpq + a * LDG + i * LDG2;
      const double* V_ov = V_kpq + a * LDV + i * LDV2;
      for (auto p : occ_same) h_el += G_ov[p];
      for (auto p : occ_othr) h_el += V_ov[p];

      // Early Exit
      if (std::abs(coeff * h_el) < h_el_tol) continue;

      // Calculate Excited Determinant
      auto ex_det = wfn_traits::template single_excitation_no_check<Sigma>(
          state_full, i, a);

      // Calculate Excitation Sign in a Canonical Way
      auto sign = single_excitation_sign(state_same, a, i);
      h_el *= sign;

      // Calculate fast diagonal matrix element
      auto h_diag =
          ham_gen.fast_diag_single(eps_same[i], eps_same[a], i, a, root_diag);

      // Append to return values
      asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});

    }  // Loop over single extitations
}

/**
 * @brief Generate same-spin double excitation ASCI contributions
 *
 * This function computes matrix elements and contributions for all possible
 * double excitations within the same spin manifold. It handles same-spin
 * electron correlation effects through two-electron integrals and enforces
 * antisymmetry requirements.
 *
 * @tparam Sigma The spin type (Alpha or Beta) for the excitations
 * @tparam WfnType Full wavefunction type
 * @tparam SpinWfnType Spin-specific wavefunction type
 * @param[in] coeff Coefficient of the reference determinant
 * @param[in] state_full Full determinant state (both alpha and beta spins)
 * @param[in] state_same Spin component being excited
 * @param[in] state_other Opposite spin component (unchanged)
 * @param[in] ss_occ Same-spin occupied orbital indices
 * @param[in] vir Virtual orbital indices
 * @param[in] os_occ Opposite-spin occupied orbital indices
 * @param[in] eps_same Orbital energies for the same spin
 * @param[in] G Same-spin two-electron integral tensor
 * @param[in] LDG Leading dimension of G tensor
 * @param[in] h_el_tol Threshold for matrix element magnitude
 * @param[in] root_diag Root diagonal correction term
 * @param[in] E0 Reference energy
 * @param[in] ham_gen Hamiltonian generator for fast diagonal evaluation
 * @param[in,out] asci_contributions Container to store the computed
 * contributions
 */
template <Spin Sigma, typename WfnType, typename SpinWfnType>
void append_ss_doubles_asci_contributions(
    double coeff, WfnType state_full, SpinWfnType state_same,
    SpinWfnType state_other, const std::vector<uint32_t>& ss_occ,
    const std::vector<uint32_t>& vir, const std::vector<uint32_t>& os_occ,
    const double* eps_same, const double* G, size_t LDG, double h_el_tol,
    double root_diag, double E0,
    const HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using spin_wfn_traits = wavefunction_traits<SpinWfnType>;
  const size_t num_occupied_orbitals = ss_occ.size();
  const size_t num_virtual_orbitals = vir.size();

  const size_t LDG2 = LDG * LDG;
  for (auto ii = 0; ii < num_occupied_orbitals; ++ii)
    for (auto aa = 0; aa < num_virtual_orbitals; ++aa) {
      const auto i = ss_occ[ii];
      const auto a = vir[aa];
      const auto G_ai = G + (a + i * LDG) * LDG2;

      for (auto jj = ii + 1; jj < num_occupied_orbitals; ++jj)
        for (auto bb = aa + 1; bb < num_virtual_orbitals; ++bb) {
          const auto j = ss_occ[jj];
          const auto b = vir[bb];
          const auto jb = b + j * LDG;
          const auto G_aibj = G_ai[jb];

          if (std::abs(coeff * G_aibj) < h_el_tol) continue;

          // Compute excited determinant (spin)
          const auto full_ex_spin = spin_wfn_traits::double_excitation_no_check(
              SpinWfnType(0), i, j, a, b);
          const auto ex_det_spin = state_same ^ full_ex_spin;

          // Calculate the sign in a canonical way
          double sign = doubles_sign(state_same, ex_det_spin, full_ex_spin);

          // Calculate full excited determinant
          auto ex_det =
              wfn_traits::template from_spin<Sigma>(ex_det_spin, state_other);

          // Update sign of matrix element
          auto h_el = sign * G_aibj;

          // Evaluate fast diagonal matrix element
          auto h_diag =
              ham_gen.fast_diag_ss_double(eps_same[i], eps_same[j], eps_same[a],
                                          eps_same[b], i, j, a, b, root_diag);

          // Append {det, c*h_el}
          asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});

        }  // Restricted BJ loop
    }  // AI Loop
}

/**
 * @brief Generate opposite-spin double excitation ASCI contributions
 *
 * This function computes matrix elements and contributions for all possible
 * double excitations between different spin manifolds (alpha-beta). It handles
 * opposite-spin electron correlation effects and is typically the dominant
 * contribution to correlation.
 *
 * @tparam WfnType Full wavefunction type
 * @tparam SpinWfnType Spin-specific wavefunction type
 * @param[in] coeff Coefficient of the reference determinant
 * @param[in] state_full Full determinant state (both alpha and beta spins)
 * @param[in] state_alpha Alpha spin component
 * @param[in] state_beta Beta spin component
 * @param[in] occ_alpha Occupied alpha orbital indices
 * @param[in] occ_beta Occupied beta orbital indices
 * @param[in] vir_alpha Virtual alpha orbital indices
 * @param[in] vir_beta Virtual beta orbital indices
 * @param[in] eps_alpha Alpha orbital energies
 * @param[in] eps_beta Beta orbital energies
 * @param[in] V Opposite-spin two-electron integral tensor
 * @param[in] LDV Leading dimension of V tensor
 * @param[in] h_el_tol Threshold for matrix element magnitude
 * @param[in] root_diag Root diagonal correction term
 * @param[in] E0 Reference energy
 * @param[in] ham_gen Hamiltonian generator for fast diagonal evaluation
 * @param[in,out] asci_contributions Container to store the computed
 * contributions
 */
template <typename WfnType, typename SpinWfnType>
void append_os_doubles_asci_contributions(
    double coeff, WfnType state_full, SpinWfnType state_alpha,
    SpinWfnType state_beta, const std::vector<uint32_t>& occ_alpha,
    const std::vector<uint32_t>& occ_beta,
    const std::vector<uint32_t>& vir_alpha,
    const std::vector<uint32_t>& vir_beta, const double* eps_alpha,
    const double* eps_beta, const double* V, size_t LDV, double h_el_tol,
    double root_diag, double E0,
    const HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  const size_t LDV2 = LDV * LDV;
  for (auto i : occ_alpha)
    for (auto a : vir_alpha) {
      const auto V_ai = V + a + i * LDV;

      double sign_alpha = single_excitation_sign(state_alpha, a, i);
      for (auto j : occ_beta)
        for (auto b : vir_beta) {
          const auto jb = b + j * LDV;
          const auto V_aibj = V_ai[jb * LDV2];

          if (std::abs(coeff * V_aibj) < h_el_tol) continue;

          double sign_beta = single_excitation_sign(state_beta, b, j);
          double sign = sign_alpha * sign_beta;

          auto ex_det =
              wfn_traits::template single_excitation_no_check<Spin::Alpha>(
                  state_full, a, i);
          ex_det = wfn_traits::template single_excitation_no_check<Spin::Beta>(
              ex_det, b, j);
          auto h_el = sign * V_aibj;

          // Evaluate fast diagonal element
          auto h_diag = ham_gen.fast_diag_os_double(eps_alpha[i], eps_beta[j],
                                                    eps_alpha[a], eps_beta[b],
                                                    i, j, a, b, root_diag);

          asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});
        }  // BJ loop
    }  // AI loop
}

/**
 * @brief Generate all unique pairs of orbitals for double excitations
 *
 * This utility function creates all possible pairs from a given set of orbital
 * indices, storing them as bitsets with two bits flipped. This is useful for
 * generating double excitation patterns efficiently.
 *
 * @tparam N Size of the wavefunction bitset
 * @tparam IndContainer Container type for orbital indices
 * @param[in] inds Container of orbital indices to pair
 * @param[out] w Output vector of wavefunction bitsets with orbital pairs
 */
template <size_t N, typename IndContainer>
void generate_pairs(const IndContainer& inds, std::vector<wfn_t<N>>& w) {
  const size_t nind = inds.size();
  w.resize((nind * (nind - 1)) / 2, 0);
  for (int i = 0, ij = 0; i < nind; ++i)
    for (int j = i + 1; j < nind; ++j, ++ij) {
      w[ij].flip(inds[i]).flip(inds[j]);
    }
}

}  // namespace macis

#include <macis/asci/mask_constraints.hpp>
