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
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

/**
 * @brief Validate active orbital indices
 * @param indices The indices to validate
 * @param spin_label Label for error messages (e.g., "Alpha", "Beta")
 * @param num_molecular_orbitals Total number of molecular orbitals
 * @return true if the indices are contiguous, false otherwise
 */
bool validate_active_contiguous_indices(const std::vector<size_t>& indices,
                                        const std::string& spin_label,
                                        size_t num_molecular_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  if (indices.empty()) return true;

  // Cannot contain more than the total number of MOs
  if (indices.size() > num_molecular_orbitals) {
    throw std::runtime_error("Number of requested " + spin_label +
                             " active orbitals exceeds total number of MOs");
  }

  // Make sure that the indices are within bounds
  for (const auto& idx : indices) {
    if (static_cast<size_t>(idx) >= num_molecular_orbitals) {
      throw std::runtime_error(
          spin_label +
          " active orbital index out of bounds: " + std::to_string(idx));
    }
  }

  // Make sure that the indices are unique
  std::set<size_t> unique_indices(indices.begin(), indices.end());
  if (unique_indices.size() != indices.size()) {
    throw std::runtime_error(spin_label +
                             " active orbital indices must be unique");
  }

  // Make sure that the indices are sorted
  std::vector<size_t> sorted_indices(indices.begin(), indices.end());
  std::sort(sorted_indices.begin(), sorted_indices.end());
  if (indices != sorted_indices) {
    throw std::runtime_error(spin_label +
                             " active orbital indices must be sorted");
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

  for (size_t k = 0; k < rank; ++k) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        V_ao(ao_three_center_vectors.col(k).data(), n_ao, n_ao);

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

  // Extract occupied orbital coefficients
  Eigen::MatrixXd C_occ(n_ao, n_occ);
  for (size_t idx = 0; idx < n_occ; ++idx) {
    C_occ.col(idx) = coeffs.col(occ_orb_ind[idx]);
  }

  // Half-transform: L^k_{sigma,i} = L^k_{mu,sigma} * C_{mu,i}
  Eigen::MatrixXd L_sigma_occ(n_ao * n_occ, rank);
  for (size_t k = 0; k < rank; ++k) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        L_k(ao_three_center_vectors.col(k).data(), n_ao, n_ao);
    Eigen::Map<Eigen::MatrixXd> L_k_occ(L_sigma_occ.col(k).data(), n_ao, n_occ);
    L_k_occ.noalias() = L_k * C_occ;
  }

  // K_{lambda,sigma} = sum_k L^k_{lambda,i} * L^k_{sigma,i}
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_ao, n_ao);
  for (size_t k = 0; k < rank; ++k) {
    Eigen::Map<const Eigen::MatrixXd> L_k_occ(L_sigma_occ.col(k).data(), n_ao,
                                               n_occ);
    K.noalias() += L_k_occ * L_k_occ.transpose();
  }

  return K;
}

bool is_indices_contiguous(const std::vector<size_t>& indices) {
  if (indices.size() <= 1) return true;
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    if (indices[i + 1] - indices[i] != 1) {
      return false;
    }
  }
  return true;
}

Eigen::MatrixXd build_inactive_density(const Eigen::MatrixXd& C,
                                       const std::vector<size_t>& indices,
                                       size_t n_ao) {
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_ao, n_ao);
  if (indices.empty()) return D;

  if (is_indices_contiguous(indices)) {
    auto C_inactive =
        C.block(0, indices.front(), n_ao, indices.size());
    D = C_inactive * C_inactive.transpose();
  } else {
    for (size_t i : indices) {
      D += C.col(i) * C.col(i).transpose();
    }
  }
  return D;
}

InactiveFockResult compute_restricted_inactive(
    const Eigen::MatrixXd& J_ao, const Eigen::MatrixXd& K_ao,
    const Eigen::MatrixXd& H_full, const Eigen::MatrixXd& Ca,
    const std::vector<size_t>& inactive_indices,
    const std::vector<size_t>& active_indices) {
  size_t num_mo = Ca.cols();
  size_t nactive = active_indices.size();

  // Inactive Fock in AO basis: F = H + 2J - K
  Eigen::MatrixXd F_inactive_ao = H_full + 2 * J_ao - K_ao;

  // Transform to MO basis
  Eigen::MatrixXd F_inactive(num_mo, num_mo);
  F_inactive = Ca.transpose() * F_inactive_ao * Ca;

  // Inactive energy (diagonal of C^T H C only)
  double E_inactive = 0.0;
  for (auto i : inactive_indices) {
    E_inactive += Ca.col(i).dot(H_full * Ca.col(i)) + F_inactive(i, i);
  }

  // Extract active-space one-body Hamiltonian
  Eigen::MatrixXd H_active(nactive, nactive);
  for (size_t i = 0; i < nactive; i++) {
    for (size_t j = 0; j < nactive; j++) {
      H_active(i, j) = F_inactive(active_indices[i], active_indices[j]);
    }
  }

  return {std::move(F_inactive), std::move(H_active), E_inactive};
}

UnrestrictedInactiveFockResult compute_unrestricted_inactive(
    const Eigen::MatrixXd& J_alpha_ao, const Eigen::MatrixXd& K_alpha_ao,
    const Eigen::MatrixXd& J_beta_ao, const Eigen::MatrixXd& K_beta_ao,
    const Eigen::MatrixXd& H_full, const Eigen::MatrixXd& Ca,
    const Eigen::MatrixXd& Cb,
    const std::vector<size_t>& inactive_indices_alpha,
    const std::vector<size_t>& inactive_indices_beta,
    const std::vector<size_t>& active_indices_alpha,
    const std::vector<size_t>& active_indices_beta) {
  size_t num_mo = Ca.cols();
  size_t nactive = active_indices_alpha.size();

  // Fock matrices in AO basis
  Eigen::MatrixXd F_alpha_ao = H_full + J_alpha_ao + J_beta_ao - K_alpha_ao;
  Eigen::MatrixXd F_beta_ao = H_full + J_alpha_ao + J_beta_ao - K_beta_ao;

  // Transform to MO basis
  Eigen::MatrixXd F_inactive_alpha(num_mo, num_mo);
  Eigen::MatrixXd F_inactive_beta(num_mo, num_mo);
  F_inactive_alpha = Ca.transpose() * F_alpha_ao * Ca;
  F_inactive_beta = Cb.transpose() * F_beta_ao * Cb;

  // Inactive energy
  double E_inactive = 0.0;
  for (auto i : inactive_indices_alpha) {
    E_inactive += Ca.col(i).dot(H_full * Ca.col(i)) + F_inactive_alpha(i, i);
  }
  for (auto i : inactive_indices_beta) {
    E_inactive += Cb.col(i).dot(H_full * Cb.col(i)) + F_inactive_beta(i, i);
  }
  E_inactive *= 0.5;

  // Extract active-space one-body Hamiltonians
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

  return {std::move(F_inactive_alpha), std::move(F_inactive_beta),
          std::move(H_active_alpha), std::move(H_active_beta), E_inactive};
}

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
