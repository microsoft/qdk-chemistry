/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include "mp2.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <optional>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mp2.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <tuple>

namespace qdk::chemistry::algorithms::microsoft {

MP2Calculator::MP2Calculator() = default;

DynamicalCorrelationResult MP2Calculator::_run_impl(
    std::shared_ptr<data::Ansatz> ansatz) const {
  QDK_LOG_TRACE_ENTERING();
  // Extract Hamiltonian and wavefunction from ansatz
  auto hamiltonian = ansatz->get_hamiltonian();
  auto wavefunction = ansatz->get_wavefunction();

  // Get orbitals from the wavefunction
  auto orbitals = wavefunction->get_orbitals();
  if (!orbitals->has_energies()) {
    throw std::runtime_error("Cannot use localized orbitals in MP2.");
  }

  // Get electron counts
  auto [n_alpha, n_beta] = wavefunction->get_active_num_electrons();

  // Compute correlation energy
  double E_corr = 0.0;

  // Determine whether to use restricted or unrestricted MP2
  bool use_unrestricted = hamiltonian->is_unrestricted() || (n_alpha != n_beta);

  if (use_unrestricted) {
    E_corr = calculate_unrestricted_mp2_energy(hamiltonian, orbitals, n_alpha,
                                               n_beta);
  } else {
    // For restricted case, use number of occupied orbitals (doubly occupied)
    if (n_alpha != n_beta) {
      throw std::runtime_error(
          "Restricted MP2 requires equal alpha and beta electrons");
    }

    E_corr = calculate_restricted_mp2_energy(hamiltonian, orbitals, n_alpha);
  }

  // Create MP2Container
  auto mp2_container =
      std::make_unique<data::MP2Container>(hamiltonian, wavefunction);

  auto mp2_wavefunction =
      std::make_shared<data::Wavefunction>(std::move(mp2_container));

  // Calculate total energy = reference energy + correlation energy
  double reference_energy_ = ansatz->calculate_energy();
  double total_energy = reference_energy_ + E_corr;
  return {total_energy, mp2_wavefunction, mp2_wavefunction};
}

double MP2Calculator::calculate_restricted_mp2_energy(
    std::shared_ptr<qdk::chemistry::data::Hamiltonian> ham,
    std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals,
    size_t n_occ) const {
  QDK_LOG_TRACE_ENTERING();
  // Validate input parameters
  if (!ham->is_restricted()) {
    throw std::runtime_error("This function requires a restricted Hamiltonian");
  }

  if (!orbitals->has_energies()) {
    throw std::runtime_error(
        "Orbital energies are required for MP2 calculation");
  }

  int active_space_size;
  if (orbitals->has_active_space()) {
    const auto& [active_space_ind_alpha, active_space_ind_beta] =
        orbitals->get_active_space_indices();
    active_space_size = active_space_ind_alpha.size();
    int active_space_size_beta = active_space_ind_beta.size();
    if (active_space_size_beta != active_space_size) {
      throw std::runtime_error(
          "Active space sizes of alpha and beta should be the same");
    }
  } else {
    throw std::runtime_error("Active space should be assigned.");
  }

  if (n_occ > active_space_size) {
    throw std::invalid_argument(
        "Number of occupied orbitals cannot exceed total orbitals");
  }

  // Get orbital energies (same for alpha and beta in restricted case)
  const auto& [eps_alpha, eps_beta] = orbitals->get_energies();

  // Calculate virtual orbital count
  const size_t n_vir = active_space_size - n_occ;

  // Get two-electron integrals
  const auto& [moeri, moeri_aabb, moeri_bbbb] = ham->get_two_body_integrals();

  double E_MP2 = 0.0;

  // Initialize T2 amplitudes storage (only for energy calculation, discarded
  // after)
  size_t t2_size = n_occ * n_occ * n_vir * n_vir;
  Eigen::VectorXd t2_amplitudes(t2_size);
  t2_amplitudes.setZero();

  // Pre-compute strides for tensor indexing
  const size_t stride_k = active_space_size;
  const size_t stride_j = active_space_size * active_space_size;
  const size_t stride_i =
      active_space_size * active_space_size * active_space_size;

  // Sum over all occupied and virtual orbital pairs
  compute_restricted_t2(eps_alpha, moeri, n_occ, n_vir, stride_i, stride_j,
                        stride_k, t2_amplitudes, &E_MP2);

  return E_MP2;
}

double MP2Calculator::calculate_unrestricted_mp2_energy(
    std::shared_ptr<qdk::chemistry::data::Hamiltonian> ham,
    std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals, size_t n_alpha,
    size_t n_beta) const {
  QDK_LOG_TRACE_ENTERING();
  // Validation
  if (!ham->is_unrestricted()) {
    throw std::runtime_error(
        "MP2 calculation requires an unrestricted Hamiltonian");
  }
  if (!orbitals->has_energies()) {
    throw std::runtime_error(
        "Orbital energies are required for MP2 calculation");
  }

  int active_space_size;
  if (orbitals->has_active_space()) {
    const auto& [active_space_ind_alpha, active_space_ind_beta] =
        orbitals->get_active_space_indices();
    active_space_size = active_space_ind_alpha.size();
    int active_space_size_beta = active_space_ind_beta.size();
    if (active_space_size_beta != active_space_size) {
      throw std::runtime_error(
          "Active space sizes of alpha and beta should be the same");
    }
  } else {
    throw std::runtime_error("There should be an active space");
  }

  if (n_alpha > active_space_size || n_beta > active_space_size) {
    throw std::invalid_argument(
        "Number of electrons cannot exceed number of orbitals");
  }

  // Core computation
  const auto& [eps_alpha, eps_beta] = orbitals->get_energies();
  const size_t n_vir_alpha = active_space_size - n_alpha;
  const size_t n_vir_beta = active_space_size - n_beta;

  const auto& [moeri_aaaa, moeri_aabb, moeri_bbbb] =
      ham->get_two_body_integrals();

  double E_MP2_AA = 0.0, E_MP2_BB = 0.0, E_MP2_AB = 0.0;

  // Initialize T2 amplitudes storage (only for energy calculation, discarded
  // after)
  size_t t2_aa_size = n_alpha * n_alpha * n_vir_alpha * n_vir_alpha;
  size_t t2_ab_size = n_alpha * n_beta * n_vir_alpha * n_vir_beta;
  size_t t2_bb_size = n_beta * n_beta * n_vir_beta * n_vir_beta;

  Eigen::VectorXd t2_aa(t2_aa_size);
  Eigen::VectorXd t2_ab(t2_ab_size);
  Eigen::VectorXd t2_bb(t2_bb_size);
  t2_aa.setZero();
  t2_ab.setZero();
  t2_bb.setZero();

  // Pre-compute strides for tensor indexing
  const size_t stride_k = active_space_size;
  const size_t stride_j = active_space_size * active_space_size;
  const size_t stride_i =
      active_space_size * active_space_size * active_space_size;

  // Alpha-Alpha contribution
  compute_same_spin_t2(eps_alpha, moeri_aaaa, n_alpha, n_vir_alpha, stride_i,
                       stride_j, stride_k, t2_aa, &E_MP2_AA);

  // Alpha-Beta contribution
  compute_opposite_spin_t2(eps_alpha, eps_beta, moeri_aabb, n_alpha, n_beta,
                           n_vir_alpha, n_vir_beta, stride_i, stride_j,
                           stride_k, t2_ab, &E_MP2_AB);

  // Beta-Beta contribution
  compute_same_spin_t2(eps_beta, moeri_bbbb, n_beta, n_vir_beta, stride_i,
                       stride_j, stride_k, t2_bb, &E_MP2_BB);

  double total_energy = E_MP2_AA + E_MP2_BB + E_MP2_AB;
  return total_energy;
}

void MP2Calculator::compute_opposite_spin_t2(
    const Eigen::VectorXd& eps_i_spin, const Eigen::VectorXd& eps_j_spin,
    const Eigen::VectorXd& moeri, size_t n_occ_i, size_t n_occ_j,
    size_t n_vir_i, size_t n_vir_j, size_t stride_i, size_t stride_j,
    size_t stride_k, Eigen::VectorXd& t2, double* energy) {
  QDK_LOG_TRACE_ENTERING();
  // Precompute T2 tensor strides for flattened indexing
  const size_t t2_stride_i = n_occ_j * n_vir_i * n_vir_j;
  const size_t t2_stride_j = n_vir_i * n_vir_j;
  const size_t t2_stride_a = n_vir_j;

  for (size_t i = 0; i < n_occ_i; ++i) {
    const double eps_i = eps_i_spin[i];
    const size_t t2_i_base = i * t2_stride_i;

    for (size_t a = 0; a < n_vir_i; ++a) {
      const size_t a_idx = a + n_occ_i;
      const double eps_ia = eps_i - eps_i_spin[a_idx];
      const size_t t2_ia_base = t2_i_base + a * t2_stride_a;

      for (size_t j = 0; j < n_occ_j; ++j) {
        const double eps_ija = eps_ia + eps_j_spin[j];
        const size_t t2_ija_base = t2_ia_base + j * t2_stride_j;

        for (size_t b = 0; b < n_vir_j; ++b) {
          const size_t b_idx = b + n_occ_j;

          const size_t idx_iajb =
              i * stride_i + a_idx * stride_j + j * stride_k + b_idx;

          const double eri_iajb = moeri[idx_iajb];
          const double denom = eps_ija - eps_j_spin[b_idx];

          // T2 amplitude
          const double t2_iajb = eri_iajb / denom;

          // Store T2 amplitude
          const size_t t2_flat_idx = t2_ija_base + b;
          t2[t2_flat_idx] = t2_iajb;

          // Energy contribution (if requested)
          if (energy) {
            *energy += t2_iajb * eri_iajb;
          }
        }
      }
    }
  }
}

void MP2Calculator::compute_restricted_t2(const Eigen::VectorXd& eps,
                                          const Eigen::VectorXd& moeri,
                                          size_t n_occ, size_t n_vir,
                                          size_t stride_i, size_t stride_j,
                                          size_t stride_k, Eigen::VectorXd& t2,
                                          double* energy) {
  QDK_LOG_TRACE_ENTERING();
  // Precompute T2 tensor strides for flattened indexing
  const size_t t2_stride_i = n_occ * n_vir * n_vir;
  const size_t t2_stride_j = n_vir * n_vir;
  const size_t t2_stride_a = n_vir;

  for (size_t i = 0; i < n_occ; ++i) {
    const size_t i_base = i * stride_i;
    const size_t t2_i_base = i * t2_stride_i;

    for (size_t j = 0; j < n_occ; ++j) {
      const double eps_ij = eps[i] + eps[j];
      const size_t t2_ij_base = t2_i_base + j * t2_stride_j;

      for (size_t a = 0; a < n_vir; ++a) {
        const size_t a_idx = a + n_occ;
        const size_t ia_base = i_base + a_idx * stride_j;
        const double eps_ija = eps_ij - eps[a_idx];
        const size_t t2_ija_base = t2_ij_base + a * t2_stride_a;

        for (size_t b = 0; b < n_vir; ++b) {
          const size_t b_idx = b + n_occ;

          // Get integrals (ia|jb) and (ib|ja) in Mulliken notation
          const size_t idx_iajb = ia_base + j * stride_k + b_idx;
          const double eri_iajb = moeri[idx_iajb];

          // Energy denominator
          const double denom = eps_ija - eps[b_idx];

          // T2 amplitude: T_iajb = (ia|jb) / denominator
          const double t2_iajb = eri_iajb / denom;

          // Store T2 amplitude
          const size_t t2_flat_idx = t2_ija_base + b;
          t2[t2_flat_idx] = t2_iajb;

          // MP2 energy: E_MP2 += T_iajb * (2*(ia|jb) - (ib|ja))
          if (energy) {
            const size_t idx_ibja =
                i_base + b_idx * stride_j + j * stride_k + a_idx;
            const double eri_ibja = moeri[idx_ibja];
            *energy += t2_iajb * (2.0 * eri_iajb - eri_ibja);
          }
        }
      }
    }
  }
}

void MP2Calculator::compute_same_spin_t2(const Eigen::VectorXd& eps,
                                         const Eigen::VectorXd& moeri,
                                         size_t n_occ, size_t n_vir,
                                         size_t stride_i, size_t stride_j,
                                         size_t stride_k, Eigen::VectorXd& t2,
                                         double* energy) {
  QDK_LOG_TRACE_ENTERING();
  // Precompute T2 tensor strides for flattened indexing
  const size_t t2_stride_i = n_occ * n_vir * n_vir;
  const size_t t2_stride_j = n_vir * n_vir;
  const size_t t2_stride_a = n_vir;

  for (size_t i = 0; i < n_occ; ++i) {
    const size_t i_base = i * stride_i;
    const size_t t2_i_base = i * t2_stride_i;

    for (size_t a = 0; a < n_vir; ++a) {
      const size_t a_idx = a + n_occ;
      const size_t ia_base = i_base + a_idx * stride_j;
      const double eps_ia = eps[i] - eps[a_idx];
      const size_t t2_ia_base = t2_i_base + a * t2_stride_a;

      for (size_t j = i + 1; j < n_occ; ++j) {
        const double eps_ija = eps_ia + eps[j];
        const size_t t2_ija_base = t2_ia_base + j * t2_stride_j;

        for (size_t b = a + 1; b < n_vir; ++b) {
          const size_t b_idx = b + n_occ;

          const size_t idx_iajb = ia_base + j * stride_k + b_idx;
          const size_t idx_ibja =
              i_base + b_idx * stride_j + j * stride_k + a_idx;

          const double eri_iajb = moeri[idx_iajb];
          const double eri_ibja = moeri[idx_ibja];
          const double antisym_integral = eri_iajb - eri_ibja;
          const double denom = eps_ija - eps[b_idx];

          // T2 amplitude
          const double t2_iajb = antisym_integral / denom;

          // Store T2 amplitude
          const size_t t2_flat_idx = t2_ija_base + b;
          t2[t2_flat_idx] = t2_iajb;

          // Energy contribution (if requested)
          if (energy) {
            *energy += t2_iajb * antisym_integral;
          }
        }
      }
    }
  }
}
}  // namespace qdk::chemistry::algorithms::microsoft
