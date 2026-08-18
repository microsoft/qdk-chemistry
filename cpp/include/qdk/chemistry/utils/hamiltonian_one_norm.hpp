// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/data/hamiltonian.hpp>

namespace qdk::chemistry::utils {

/// Breakdown of the double-factorization (DF) fermionic LCU 1-norm of a
/// restricted electronic Hamiltonian, following Patel et al.
/// (arXiv:2409.18277), Eqs. (14)-(17). The identity/constant term (core
/// energy and fragment constants) is, by convention, excluded from the
/// 1-norm.
struct HamiltonianOneNorm {
  double one_body = 0.0;  ///< lambda_1e = sum_i |gamma_i| (Eq. 15), where
                          ///< gamma_i are the eigenvalues of the effective
                          ///< one-electron tensor (Eq. 14).
  double two_body = 0.0;  ///< lambda_2e = sum_alpha (1/2)(sum_i |eps_i|)^2
                          ///< (Eq. 17), summed over the DF fragments.
  double total = 0.0;     ///< lambda = lambda_1e + lambda_2e.
};

/// Compute the double-factorization fermionic LCU 1-norm of a restricted
/// Hamiltonian.
///
/// This is a standalone diagnostic utility: it can be called directly on any
/// qdk::chemistry::data::Hamiltonian without creating or configuring an
/// Algorithm (e.g. qdk::chemistry::algorithms::HamiltonianRegularizer). Users
/// who only want to inspect the fermionic 1-norm lambda of a Hamiltonian --
/// without necessarily also computing/applying a BLISS-style shift -- should
/// call this function directly.
///
/// The Hamiltonian is written (chemist convention, spatial orbitals) as
///   H = E_core + sum_ij h_ij E_ij + 1/2 sum_ijkl (ij|kl) (E_ij E_kl - d_jk
///   E_il)
/// where g[i,j,k,l] = (ij|kl) is the container's two-electron tensor. The DF
/// fermionic 1-norm is assembled as:
///
///  * Two-body: double-factorize the physical two-electron coefficient
///    V = 1/2 * g (Eq. (1) uses g_ijkl = 1/2 (ij|kl)) into low-rank fragments
///    (via qdk::chemistry::utils::double_factorize)
///    H^(alpha) = U^(alpha)^dagger (sum_i eps_i n_i)^2 U^(alpha); each fragment
///    contributes lambda_DF^(alpha) = 1/2 (sum_i |eps_i^(alpha)|)^2 (Eq. 17).
///
///  * One-body: fold every fragment's one-electron correction (Eq. 12) into
///    the one-electron Hamiltonian, giving the effective tensor (Eq. 14)
///      Heff_ij = h_ij + sum_k g[i,j,k,k] - 1/2 sum_k g[i,k,k,j],
///    and take lambda_1e = sum_i |gamma_i| over its eigenvalues (Eq. 15).
///
/// NOTE ON CONVENTION: this decomposes the PHYSICAL coefficient V = 1/2 * g.
/// qdk::chemistry::utils::double_factorize is elsewhere applied to the full
/// tensor g (e.g. by HamiltonianRegularizer implementations), so the
/// two_body value here is exactly half of that pipeline's aggregated
/// lambda_df; both are internally consistent, but only V = 1/2 * g reproduces
/// the electronic Hamiltonian exactly (and matches the paper's absolute
/// 1-norms).
///
/// @param hamiltonian Restricted Hamiltonian to analyze.
/// @param df_truncation_threshold Fragments whose reshaped-supermatrix
///        eigenvalue magnitude is below this are dropped from the two-body
///        1-norm.
///        Defaults to 0.0 (no truncation -- exact/lossless factorization
///        unless the caller explicitly opts into compression).
/// @return The one-body, two-body, and total DF fermionic 1-norm.
/// @throws std::runtime_error if the Hamiltonian is not restricted.
HamiltonianOneNorm hamiltonian_one_norm(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    double df_truncation_threshold = 0.0);

}  // namespace qdk::chemistry::utils
