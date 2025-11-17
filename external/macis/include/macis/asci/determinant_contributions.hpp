/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator.hpp>
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>

namespace macis {

template <typename WfnT>
struct asci_contrib {
  WfnT state;
  double c_times_matel;
  double h_diag;

  auto rv() const { return c_times_matel / h_diag; }
  auto pt2() const { return rv() * c_times_matel; }
};

template <typename WfnT>
using asci_contrib_container = std::vector<asci_contrib<WfnT>>;

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
      // h_el /= (E0 - h_diag);

      // Append to return values
      asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});

    }  // Loop over single extitations
}

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

#if 0
          // Calculate excited determinant string (spin)
          const auto full_ex_spin = wfn_t<N>(0).flip(i).flip(j).flip(a).flip(b);
          auto ex_det_spin = state_spin ^ full_ex_spin;

          // Calculate the sign in a canonical way
          double sign = doubles_sign(state_spin, ex_det_spin, full_ex_spin);

          // Calculate full excited determinant
          const auto full_ex = expand_bitset<2 * N>(full_ex_spin) << NShift;
          auto ex_det = state_full ^ full_ex;
#else
          // TODO: Can this be made faster since the orbital indices are known
          //       in advance?
          // Compute excited determinant (spin)
          const auto full_ex_spin = spin_wfn_traits::double_excitation_no_check(
              SpinWfnType(0), i, j, a, b);
          const auto ex_det_spin = state_same ^ full_ex_spin;

          // Calculate the sign in a canonical way
          double sign = doubles_sign(state_same, ex_det_spin, full_ex_spin);

          // Calculate full excited determinant
          auto ex_det =
              wfn_traits::template from_spin<Sigma>(ex_det_spin, state_other);
#endif /* USE_GENERIC_RDEP */

          // Update sign of matrix element
          auto h_el = sign * G_aibj;

          // Evaluate fast diagonal matrix element
          auto h_diag =
              ham_gen.fast_diag_ss_double(eps_same[i], eps_same[j], eps_same[a],
                                          eps_same[b], i, j, a, b, root_diag);
          // h_el /= (E0 - h_diag);

          // Append {det, c*h_el}
          asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});

        }  // Restricted BJ loop
    }  // AI Loop
}

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
          // auto ex_det = state_full;
          // ex_det.flip(a).flip(i).flip(j + N).flip(b + N);
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
          // h_el /= (E0 - h_diag);

          asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});
        }  // BJ loop
    }  // AI loop
}

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
