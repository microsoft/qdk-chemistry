/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace qdk::chemistry::algorithms::microsoft::mp2_helpers {

/**
 * @brief Compute same-spin (antisymmetric) T2 amplitudes
 *
 * This helper computes T2 amplitudes for same-spin electron pairs (alpha-alpha
 * or beta-beta), which have antisymmetric exchange integrals.
 *
 * @param eps Orbital energies
 * @param moeri Two-electron repulsion integrals (MO basis)
 * @param n_occ Number of occupied orbitals
 * @param n_vir Number of virtual orbitals
 * @param stride_i Stride for first index in 4D integral array
 * @param stride_j Stride for second index in 4D integral array
 * @param stride_k Stride for third index in 4D integral array
 * @param t2 Output vector for T2 amplitudes (will be filled)
 * @param energy Optional pointer to accumulate energy contribution
 */
inline void compute_same_spin_t2(const Eigen::VectorXd& eps,
                                 const Eigen::VectorXd& moeri, size_t n_occ,
                                 size_t n_vir, size_t stride_i, size_t stride_j,
                                 size_t stride_k, Eigen::VectorXd& t2,
                                 double* energy = nullptr) {
  for (size_t i = 0; i < n_occ; ++i) {
    const size_t i_base = i * stride_i;

    for (size_t a = 0; a < n_vir; ++a) {
      const size_t a_idx = a + n_occ;
      const size_t ia_base = i_base + a_idx * stride_j;
      const double eps_ia = eps[i] - eps[a_idx];

      for (size_t j = i + 1; j < n_occ; ++j) {
        const double eps_ija = eps_ia + eps[j];

        for (size_t b = a + 1; b < n_vir; ++b) {
          const size_t b_idx = b + n_occ;

          const size_t idx_ijab = ia_base + j * stride_k + b_idx;
          const size_t idx_ijba =
              i_base + b_idx * stride_j + j * stride_k + a_idx;

          const double eri_ijab = moeri[idx_ijab];
          const double eri_ijba = moeri[idx_ijba];
          const double antisym_integral = eri_ijab - eri_ijba;
          const double denom = eps_ija - eps[b_idx];

          // T2 amplitude
          const double t2_ijab = antisym_integral / denom;

          // Store T2 amplitude
          size_t t2_flat_idx =
              i * n_occ * n_vir * n_vir + j * n_vir * n_vir + a * n_vir + b;
          t2[t2_flat_idx] = t2_ijab;

          // Energy contribution (if requested)
          if (energy) {
            *energy += t2_ijab * antisym_integral;
          }
        }
      }
    }
  }
}

/**
 * @brief Compute opposite-spin T2 amplitudes
 *
 * This helper computes T2 amplitudes for opposite-spin electron pairs
 * (alpha-beta), which don't have antisymmetric exchange.
 *
 * @param eps_i_spin Orbital energies for i,a indices
 * @param eps_j_spin Orbital energies for j,b indices
 * @param moeri Two-electron repulsion integrals (MO basis)
 * @param n_occ_i Number of occupied orbitals (i spin)
 * @param n_occ_j Number of occupied orbitals (j spin)
 * @param n_vir_i Number of virtual orbitals (i spin)
 * @param n_vir_j Number of virtual orbitals (j spin)
 * @param stride_i Stride for first index in 4D integral array
 * @param stride_j Stride for second index in 4D integral array
 * @param stride_k Stride for third index in 4D integral array
 * @param t2 Output vector for T2 amplitudes (will be filled)
 * @param energy Optional pointer to accumulate energy contribution
 */
inline void compute_opposite_spin_t2(
    const Eigen::VectorXd& eps_i_spin, const Eigen::VectorXd& eps_j_spin,
    const Eigen::VectorXd& moeri, size_t n_occ_i, size_t n_occ_j,
    size_t n_vir_i, size_t n_vir_j, size_t stride_i, size_t stride_j,
    size_t stride_k, Eigen::VectorXd& t2, double* energy = nullptr) {
  for (size_t i = 0; i < n_occ_i; ++i) {
    const double eps_i = eps_i_spin[i];

    for (size_t a = 0; a < n_vir_i; ++a) {
      const size_t a_idx = a + n_occ_i;
      const double eps_ia = eps_i - eps_i_spin[a_idx];

      for (size_t j = 0; j < n_occ_j; ++j) {
        const double eps_ija = eps_ia + eps_j_spin[j];

        for (size_t b = 0; b < n_vir_j; ++b) {
          const size_t b_idx = b + n_occ_j;

          const size_t idx_ijab =
              i * stride_i + a_idx * stride_j + j * stride_k + b_idx;

          const double eri_ijab = moeri[idx_ijab];
          const double denom = eps_ija - eps_j_spin[b_idx];

          // T2 amplitude
          const double t2_ijab = eri_ijab / denom;

          // Store T2 amplitude
          size_t t2_flat_idx = i * n_occ_j * n_vir_i * n_vir_j +
                               j * n_vir_i * n_vir_j + a * n_vir_j + b;
          t2[t2_flat_idx] = t2_ijab;

          // Energy contribution (if requested)
          if (energy) {
            *energy += t2_ijab * eri_ijab;
          }
        }
      }
    }
  }
}

/**
 * @brief Compute restricted T2 amplitudes
 *
 * This helper computes T2 amplitudes for restricted (closed-shell) systems.
 *
 * @param eps Orbital energies
 * @param moeri Two-electron repulsion integrals (MO basis)
 * @param n_occ Number of occupied orbitals
 * @param n_vir Number of virtual orbitals
 * @param stride_i Stride for first index in 4D integral array
 * @param stride_j Stride for second index in 4D integral array
 * @param stride_k Stride for third index in 4D integral array
 * @param t2 Output vector for T2 amplitudes (will be filled)
 * @param energy Optional pointer to accumulate energy contribution
 */
inline void compute_restricted_t2(const Eigen::VectorXd& eps,
                                  const Eigen::VectorXd& moeri, size_t n_occ,
                                  size_t n_vir, size_t stride_i,
                                  size_t stride_j, size_t stride_k,
                                  Eigen::VectorXd& t2,
                                  double* energy = nullptr) {
  for (size_t i = 0; i < n_occ; ++i) {
    const size_t i_base = i * stride_i;

    for (size_t j = 0; j < n_occ; ++j) {
      const double eps_ij = eps[i] + eps[j];

      for (size_t a = 0; a < n_vir; ++a) {
        const size_t a_idx = a + n_occ;
        const size_t ia_base = i_base + a_idx * stride_j;
        const double eps_ija = eps_ij - eps[a_idx];

        for (size_t b = 0; b < n_vir; ++b) {
          const size_t b_idx = b + n_occ;

          // Get integrals <ij|ab> and <ij|ba>
          const size_t idx_ijab = ia_base + j * stride_k + b_idx;
          const double eri_ijab = moeri[idx_ijab];

          // Energy denominator
          const double denom = eps_ija - eps[b_idx];

          // T2 amplitude: T_ijab = <ij|ab> / denominator
          const double t2_ijab = eri_ijab / denom;

          // Store T2 amplitude
          size_t t2_flat_idx =
              i * n_occ * n_vir * n_vir + j * n_vir * n_vir + a * n_vir + b;
          t2[t2_flat_idx] = t2_ijab;

          // MP2 energy: E_MP2 += T_ijab * (2*<ij|ab> - <ij|ba>)
          if (energy) {
            const size_t idx_ijba =
                i_base + b_idx * stride_j + j * stride_k + a_idx;
            const double eri_ijba = moeri[idx_ijba];
            *energy += t2_ijab * (2.0 * eri_ijab - eri_ijba);
          }
        }
      }
    }
  }
}

}  // namespace qdk::chemistry::algorithms::microsoft::mp2_helpers
