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

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
