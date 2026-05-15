// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "hamiltonian_util.hpp"

// STL Headers
#include <algorithm>
#include <cstring>
#include <set>
#include <tuple>
#include <unordered_set>

// QDK/Chemistry SCF headers
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

// Schwarz screening
#include "scf/src/eri/schwarz.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <blas.hh>
#include <qdk/chemistry/data/hamiltonian_containers/three_center.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

bool validate_contiguous_indices(const std::vector<size_t>& indices,
                                 const std::string& label,
                                 size_t num_molecular_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  if (indices.empty()) return true;

  // Cannot contain more than the total number of MOs
  if (indices.size() > num_molecular_orbitals) {
    throw std::runtime_error("Number of requested " + label +
                             " orbitals exceeds total number of MOs");
  }

  // Make sure that the indices are within bounds
  for (const auto& idx : indices) {
    if (static_cast<size_t>(idx) >= num_molecular_orbitals) {
      throw std::runtime_error(label +
                               " orbital index out of bounds: " +
                               std::to_string(idx));
    }
  }

  // Make sure that the indices are unique
  std::set<size_t> unique_indices(indices.begin(), indices.end());
  if (unique_indices.size() != indices.size()) {
    throw std::runtime_error(label + " orbital indices must be unique");
  }

  // Make sure that the indices are sorted
  std::vector<size_t> sorted_indices(indices.begin(), indices.end());
  std::sort(sorted_indices.begin(), sorted_indices.end());
  if (indices != sorted_indices) {
    throw std::runtime_error(label + " orbital indices must be sorted");
  }

  // Check if indices are contiguous
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    if (indices[i + 1] - indices[i] != 1) {
      return false;
    }
  }

  return true;
}

Eigen::MatrixXd transform_three_center_ao_to_mo(
    const Eigen::MatrixXd& ao_three_center_vectors,
    const Eigen::MatrixXd& mo_coeffs) {
  size_t n_ao = mo_coeffs.rows();
  size_t n_mo = mo_coeffs.cols();
  size_t rank = ao_three_center_vectors.cols();

  // Validate dimensions
  if (n_ao == 0 || n_mo == 0) {
    throw std::invalid_argument("C matrix has zero dimensions");
  }
  if (ao_three_center_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_three_center_vectors dimensions do not match n_ao");
  }

  Eigen::MatrixXd mo_vectors(n_mo * n_mo, rank);

  // iterate over each 3-center vector
  for (size_t k = 0; k < rank; ++k) {
    // Reshape the flat AO vector to a matrix (n_ao x n_ao)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        V_ao(ao_three_center_vectors.col(k).data(), n_ao, n_ao);

    // Transform from AO to MO basis: C^T * V_ao * C
    // Write directly to output column
    Eigen::Map<Eigen::MatrixXd> V_mo_map(mo_vectors.col(k).data(), n_mo, n_mo);
    V_mo_map.noalias() = mo_coeffs.transpose() * V_ao * mo_coeffs;
  }

  return mo_vectors;
}

Eigen::MatrixXd build_J_from_three_center(
    const Eigen::MatrixXd& ao_three_center_vectors,
    const Eigen::MatrixXd& density) {
  size_t n_ao = density.rows();
  size_t rank = ao_three_center_vectors.cols();

  // Validate dimensions
  if (density.cols() != density.rows()) {
    throw std::invalid_argument("Density matrix must be square");
  }
  if (ao_three_center_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_three_center_vectors dimensions do not match density matrix");
  }

  // Flatten density matrix
  Eigen::Map<const Eigen::VectorXd> density_vec(density.data(), n_ao * n_ao);

  // Compute all inner products at once: V_k = sum_{mu,nu} L^k_{mu,nu} *
  // P_{mu,nu} V = L^T * vec(P)
  Eigen::VectorXd V = ao_three_center_vectors.transpose() * density_vec;

  // Reconstruct J: J = sum_k L_k * V_k = L * V (reshaped)
  Eigen::VectorXd J_vec = ao_three_center_vectors * V;

  // Reshape back to matrix
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      J(J_vec.data(), n_ao, n_ao);
  return J;
}

Eigen::MatrixXd build_K_from_three_center(
    const Eigen::MatrixXd& ao_three_center_vectors,
    const Eigen::MatrixXd& coeffs, const std::vector<size_t>& occ_orb_ind) {
  size_t n_ao = coeffs.rows();
  size_t n_occ = occ_orb_ind.size();
  size_t rank = ao_three_center_vectors.cols();

  // Validate dimensions
  if (ao_three_center_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_three_center_vectors dimensions do not match density matrix");
  }

  // Extract occupied orbital coefficients only
  Eigen::MatrixXd C_occ(n_ao, n_occ);
  for (size_t idx = 0; idx < n_occ; ++idx) {
    C_occ.col(idx) = coeffs.col(occ_orb_ind[idx]);
  }

  // Transform to occupied MO basis only: L^k_{\sigma,i_occ} = L^k_{\mu\sigma} *
  // C_{\mu,i_occ}
  Eigen::MatrixXd L_sigma_occ(n_ao * n_occ, rank);
  for (size_t k = 0; k < rank; ++k) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        L_k(ao_three_center_vectors.col(k).data(), n_ao, n_ao);
    Eigen::Map<Eigen::MatrixXd> L_k_occ(L_sigma_occ.col(k).data(), n_ao, n_occ);
    L_k_occ.noalias() = L_k * C_occ;
  }

  // Build K_{\lambda\sigma} = \sum_k L^k_{\lambda,i} * L^k_{\sigma,i}
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_ao, n_ao);
  for (size_t k = 0; k < rank; ++k) {
    Eigen::Map<const Eigen::MatrixXd> L_k_occ(L_sigma_occ.col(k).data(), n_ao,
                                              n_occ);
    K.noalias() += L_k_occ * L_k_occ.transpose();
  }

  return K;
}

std::shared_ptr<data::Hamiltonian>
build_active_space_hamiltonian_from_three_center(
    const Eigen::Ref<const Eigen::MatrixXd>& B_ao,
    const Eigen::MatrixXd& H_full, const Eigen::MatrixXd& Ca,
    const Eigen::MatrixXd& Cb, std::shared_ptr<data::Orbitals> orbitals,
    std::shared_ptr<data::Structure> structure, bool is_restricted_calc,
    bool store_ao_vectors) {
  QDK_LOG_TRACE_ENTERING();

  const size_t num_atomic_orbitals = Ca.rows();
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  // Get active and inactive space indices
  auto [active_indices_alpha, active_indices_beta] =
      orbitals->get_active_space_indices();
  const size_t nactive = active_indices_alpha.size();

  // Validate active space contiguity
  bool alpha_active_is_contiguous = validate_active_contiguous_indices(
      active_indices_alpha, "Alpha", num_molecular_orbitals);
  bool beta_active_is_contiguous = true;
  if (active_indices_beta != active_indices_alpha) {
    beta_active_is_contiguous = validate_active_contiguous_indices(
        active_indices_beta, "Beta", num_molecular_orbitals);
  } else {
    beta_active_is_contiguous = alpha_active_is_contiguous;
  }

  // Build active coefficient matrices
  Eigen::MatrixXd Ca_active(num_atomic_orbitals, nactive);
  Eigen::MatrixXd Cb_active(num_atomic_orbitals, nactive);

  if (alpha_active_is_contiguous) {
    Ca_active = Ca.block(0, active_indices_alpha.front(), num_atomic_orbitals,
                         nactive);
  } else {
    for (size_t i = 0; i < nactive; i++) {
      Ca_active.col(i) = Ca.col(active_indices_alpha[i]);
    }
  }

  if (beta_active_is_contiguous) {
    Cb_active = Cb.block(0, active_indices_beta.front(), num_atomic_orbitals,
                         nactive);
  } else {
    for (size_t i = 0; i < nactive; i++) {
      Cb_active.col(i) = Cb.col(active_indices_beta[i]);
    }
  }

  // Transform AO three-center vectors to MO basis
  Eigen::MatrixXd mo_vectors_aa =
      transform_three_center_ao_to_mo(B_ao, Ca_active);
  Eigen::MatrixXd mo_vectors_bb;
  if (!is_restricted_calc) {
    mo_vectors_bb = transform_three_center_ao_to_mo(B_ao, Cb_active);
  }

  // Prepare optional AO vectors for storage
  std::optional<Eigen::MatrixXd> ao_to_store;
  if (store_ao_vectors) {
    ao_to_store = Eigen::MatrixXd(B_ao);
  }

  // Get inactive space indices
  auto [inactive_indices_alpha, inactive_indices_beta] =
      orbitals->get_inactive_space_indices();

  // For restricted calculations, alpha and beta inactive spaces must match
  if (orbitals->is_restricted() &&
      inactive_indices_alpha != inactive_indices_beta) {
    throw std::runtime_error(
        "For restricted orbitals, alpha and beta inactive spaces must be "
        "identical");
  }

  // No inactive orbitals: all occupied orbitals are in active space
  if (inactive_indices_alpha.empty() && inactive_indices_beta.empty()) {
    if (is_restricted_calc) {
      Eigen::MatrixXd H_active = Ca_active.transpose() * H_full * Ca_active;
      Eigen::MatrixXd dummy_fock = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::ThreeCenterHamiltonianContainer>(
              H_active, mo_vectors_aa, orbitals,
              structure->calculate_nuclear_repulsion_energy(), dummy_fock,
              std::move(ao_to_store)));
    } else {
      Eigen::MatrixXd H_active_alpha =
          Ca_active.transpose() * H_full * Ca_active;
      Eigen::MatrixXd H_active_beta =
          Cb_active.transpose() * H_full * Cb_active;
      Eigen::MatrixXd dummy_fock_alpha = Eigen::MatrixXd::Zero(0, 0);
      Eigen::MatrixXd dummy_fock_beta = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::ThreeCenterHamiltonianContainer>(
              H_active_alpha, H_active_beta, mo_vectors_aa, mo_vectors_bb,
              orbitals, structure->calculate_nuclear_repulsion_energy(),
              dummy_fock_alpha, dummy_fock_beta, std::move(ao_to_store)));
    }
  }

  if (is_restricted_calc) {
    // Restricted case
    const auto& inactive_indices = inactive_indices_alpha;

    bool inactive_is_contiguous = validate_contiguous_indices(
        inactive_indices, "Restricted inactive", num_molecular_orbitals);

    // Compute inactive density matrix
    Eigen::MatrixXd D_inactive =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    if (inactive_is_contiguous) {
      auto C_inactive = Ca.block(0, inactive_indices.front(),
                                 num_atomic_orbitals, inactive_indices.size());
      D_inactive = C_inactive * C_inactive.transpose();
    } else {
      for (size_t i : inactive_indices) {
        D_inactive += Ca.col(i) * Ca.col(i).transpose();
      }
    }

    // Compute the two-electron part of inactive Fock matrix
    Eigen::MatrixXd J_inactive_ao = build_J_from_three_center(B_ao, D_inactive);
    Eigen::MatrixXd K_inactive_ao =
        build_K_from_three_center(B_ao, Ca, inactive_indices);
    Eigen::MatrixXd G_inactive_ao = 2 * J_inactive_ao - K_inactive_ao;

    // Compute inactive Fock matrix in MO basis
    Eigen::MatrixXd F_inactive_ao = G_inactive_ao + H_full;
    Eigen::MatrixXd F_inactive(num_molecular_orbitals, num_molecular_orbitals);
    F_inactive = Ca.transpose() * F_inactive_ao * Ca;

    // Compute inactive energy
    double E_inactive = 0.0;
    Eigen::MatrixXd H_mo = Ca.transpose() * H_full * Ca;
    for (auto i : inactive_indices) {
      E_inactive += H_mo(i, i) + F_inactive(i, i);
    }

    // Extract active space Hamiltonian
    Eigen::MatrixXd H_active(nactive, nactive);
    for (size_t i = 0; i < nactive; i++) {
      for (size_t j = 0; j < nactive; j++) {
        H_active(i, j) =
            F_inactive(active_indices_alpha[i], active_indices_alpha[j]);
      }
    }

    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::ThreeCenterHamiltonianContainer>(
            H_active, mo_vectors_aa, orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive, std::move(ao_to_store)));

  } else {
    // Unrestricted case

    bool alpha_inactive_is_contiguous = validate_contiguous_indices(
        inactive_indices_alpha, "Alpha inactive", num_molecular_orbitals);
    bool beta_inactive_is_contiguous = validate_contiguous_indices(
        inactive_indices_beta, "Beta inactive", num_molecular_orbitals);

    // Compute separate alpha and beta inactive density matrices
    Eigen::MatrixXd D_inactive_alpha =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    Eigen::MatrixXd D_inactive_beta =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);

    if (alpha_inactive_is_contiguous && !inactive_indices_alpha.empty()) {
      auto C_inactive_alpha =
          Ca.block(0, inactive_indices_alpha.front(), num_atomic_orbitals,
                   inactive_indices_alpha.size());
      D_inactive_alpha = C_inactive_alpha * C_inactive_alpha.transpose();
    } else {
      for (size_t i : inactive_indices_alpha) {
        D_inactive_alpha += Ca.col(i) * Ca.col(i).transpose();
      }
    }

    if (beta_inactive_is_contiguous && !inactive_indices_beta.empty()) {
      auto C_inactive_beta =
          Cb.block(0, inactive_indices_beta.front(), num_atomic_orbitals,
                   inactive_indices_beta.size());
      D_inactive_beta = C_inactive_beta * C_inactive_beta.transpose();
    } else {
      for (size_t i : inactive_indices_beta) {
        D_inactive_beta += Cb.col(i) * Cb.col(i).transpose();
      }
    }

    // Compute J and K matrices
    Eigen::MatrixXd J_alpha_ao =
        build_J_from_three_center(B_ao, D_inactive_alpha);
    Eigen::MatrixXd K_alpha_ao =
        build_K_from_three_center(B_ao, Ca, inactive_indices_alpha);
    Eigen::MatrixXd J_beta_ao =
        build_J_from_three_center(B_ao, D_inactive_beta);
    Eigen::MatrixXd K_beta_ao =
        build_K_from_three_center(B_ao, Cb, inactive_indices_beta);

    Eigen::MatrixXd F_inactive_alpha_ao =
        H_full + J_alpha_ao + J_beta_ao - K_alpha_ao;
    Eigen::MatrixXd F_inactive_beta_ao =
        H_full + J_alpha_ao + J_beta_ao - K_beta_ao;

    // Transform to MO basis
    Eigen::MatrixXd F_inactive_alpha(num_molecular_orbitals,
                                     num_molecular_orbitals);
    Eigen::MatrixXd F_inactive_beta(num_molecular_orbitals,
                                    num_molecular_orbitals);
    F_inactive_alpha = Ca.transpose() * F_inactive_alpha_ao * Ca;
    F_inactive_beta = Cb.transpose() * F_inactive_beta_ao * Cb;

    // Compute inactive energy
    Eigen::MatrixXd H_mo_alpha = Ca.transpose() * H_full * Ca;
    Eigen::MatrixXd H_mo_beta = Cb.transpose() * H_full * Cb;

    double E_inactive = 0.0;
    for (auto i : inactive_indices_alpha) {
      E_inactive += H_mo_alpha(i, i) + F_inactive_alpha(i, i);
    }
    for (auto i : inactive_indices_beta) {
      E_inactive += H_mo_beta(i, i) + F_inactive_beta(i, i);
    }
    E_inactive *= 0.5;

    // Extract active space Hamiltonians
    Eigen::MatrixXd H_active_alpha(nactive, nactive);
    Eigen::MatrixXd H_active_beta(nactive, nactive);

    for (size_t i = 0; i < nactive; i++) {
      for (size_t j = 0; j < nactive; j++) {
        H_active_alpha(i, j) =
            F_inactive_alpha(active_indices_alpha[i], active_indices_alpha[j]);
        H_active_beta(i, j) =
            F_inactive_beta(active_indices_beta[i], active_indices_beta[j]);
      }
    }

    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::ThreeCenterHamiltonianContainer>(
            H_active_alpha, H_active_beta, mo_vectors_aa, mo_vectors_bb,
            orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive_alpha, F_inactive_beta, std::move(ao_to_store)));
  }
}

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
