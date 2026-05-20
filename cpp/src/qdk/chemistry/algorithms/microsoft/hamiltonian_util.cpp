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

  // B_ao is (n_ao², rank) col-major. Mapped as (n_ao, n_ao*rank) col-major:
  //   reshaped(nu, Q*n_ao + mu) = B_ao(mu*n_ao + nu, Q)
  // First half-transform (one GEMM): contract nu index with C
  //   T1(j, Q*n_ao + mu) = sum_nu C(nu,j) * B(mu,nu,Q)
  Eigen::Map<const Eigen::MatrixXd> B_reshaped(ao_three_center_vectors.data(),
                                               n_ao, n_ao * rank);
  Eigen::MatrixXd T1 = mo_coeffs.transpose() * B_reshaped;  // (n_mo, n_ao*rank)

  // Second half-transform: contract mu index with C
  //   result(i,j,Q) = sum_mu C(mu,i) * T1(j, Q*n_ao + mu)
  // For each Q, subblock = T1.middleCols(Q*n_ao, n_ao) is (n_mo x n_ao)
  //   out_Q(i,j) = (C^T * subblock^T)(i,j)
  Eigen::MatrixXd mo_vectors(n_mo * n_mo, rank);
  for (size_t Q = 0; Q < rank; ++Q) {
    auto subblock = T1.middleCols(Q * n_ao, n_ao);  // (n_mo x n_ao)
    Eigen::Map<Eigen::MatrixXd> out_Q(mo_vectors.col(Q).data(), n_mo, n_mo);
    out_Q.noalias() = mo_coeffs.transpose() * subblock.transpose();
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

  // One GEMM for half-transform via reshape:
  //   B_ao (n_ao², rank) col-major viewed as (n_ao, n_ao*rank) gives
  //   B_reshaped(nu, Q*n_ao + mu) = B_ao(mu*n_ao + nu, Q).
  //   half = C_occ^T * B_reshaped → (n_occ, n_ao*rank)
  //   half(i, Q*n_ao + mu) = sum_nu C_occ(nu,i) * B(mu,nu,Q)
  Eigen::Map<const Eigen::MatrixXd> B_reshaped(ao_three_center_vectors.data(),
                                               n_ao, n_ao * rank);
  Eigen::MatrixXd half = C_occ.transpose() * B_reshaped;  // (n_occ, n_ao*rank)

  // Transpose + reshape to get mu as row index:
  //   half^T col-major: element (Q*n_ao+mu, i) reshaped 
  //   Reinterpreted as (n_ao, rank*n_occ):  row=mu, col=i*rank+Q 
  //   So half_r(mu, i*rank+Q) = B_half(mu, i, Q)
  Eigen::MatrixXd half_t = half.transpose();  // (n_ao*rank, n_occ)
  Eigen::Map<Eigen::MatrixXd> half_r(half_t.data(), n_ao, rank * n_occ);

  // One DSYRK: K[mu,nu] = sum_{i,Q} half_r(mu,col) * half_r(nu,col)
  Eigen::MatrixXd K = half_r * half_r.transpose();

  return K;
}

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
