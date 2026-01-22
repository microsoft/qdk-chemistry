// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace testing {

/**
 * @brief Compute the Pipek-Mezey metric for a given set of orbitals.
 * This function calculates the Pipek-Mezey metric, which is a measure of
 * the localization of molecular orbitals.
 * @param orbitals The molecular orbitals for which to compute the metric.
 * @return The Pipek-Mezey metric value.
 * @throws std::runtime_error if the orbitals are not set or if the basis set is
 * invalid.
 * @note This function assumes that the orbitals have been set up correctly
 * with coefficients and a valid basis set. It uses the AO overlap matrix
 * and the coefficients to compute the metric.
 */
auto pipek_mezey_metric(const qdk::chemistry::data::Orbitals& orbitals,
                        const Eigen::MatrixXd& C) {
  QDK_LOG_TRACE_ENTERING();
  auto basis_set = orbitals.get_basis_set();
  auto structure = basis_set->get_structure();
  const auto& S = orbitals.get_overlap_matrix();

  QDK_LOGGER().info(
      "Computing Pipek-Mezey metric for {} atomic orbitals and {} molecular "
      "orbitals",
      orbitals.get_num_atomic_orbitals(),
      orbitals.get_num_molecular_orbitals());
  QDK_LOGGER().info("AO Overlap matrix S norm: {:.16e}", S.norm());
  QDK_LOGGER().info("Orbital coefficient matrix C norm: {:.16e}", C.norm());

  Eigen::MatrixXd SC = S * C;
  QDK_LOGGER().info("Computed SC matrix with norm {:.16e}", SC.norm());

  const size_t num_atomic_orbitals = orbitals.get_num_atomic_orbitals();
  const size_t num_molecular_orbitals = orbitals.get_num_molecular_orbitals();
  const size_t natom = structure->get_num_atoms();

  Eigen::MatrixXd Xi = Eigen::MatrixXd::Zero(natom, num_molecular_orbitals);
  for (size_t p = 0; p < num_molecular_orbitals; ++p) {
    for (size_t mu = 0; mu < num_atomic_orbitals; ++mu) {
      const auto iA = basis_set->get_atom_index_for_atomic_orbital(mu);
      Xi(iA, p) += C(mu, p) * SC(mu, p);
      //   QDK_LOGGER().info(
      //       "Accumulating iA: {}; p: {}; mu: {}; C(mu, p): {:.16e}; SC(mu,
      //       p): {:.16e}; Xi(iA, p): {:.16e}", iA, p, mu, C(mu, p), SC(mu, p),
      //       Xi(iA, p));
    }
  }
  QDK_LOGGER().info("Computed Xi matrix with norm {:.16e}", Xi.norm());

  return Xi.cwiseProduct(Xi).sum();
}

/**
 * @brief Computes the Frobenius norm of the difference between U^T * U and the
 * identity matrix.
 *
 * This function measures how far a matrix U is from being unitary/orthogonal.
 * For a perfectly unitary matrix U, U^T * U = I (identity matrix), so this
 * function returns 0. The larger the returned value, the further U is from
 * being unitary.
 *
 * @param U The input matrix to test for unitarity
 * @return The Frobenius norm of (U^T * U - I), where I is the identity matrix
 */
auto norm_diff_from_unitary(const Eigen::MatrixXd& U) {
  return (U.transpose() * U - Eigen::MatrixXd::Identity(U.cols(), U.cols()))
      .norm();
}

/**
 * @brief Computes the Frobenius norms of the occupied-virtual (OV) and
 * virtual-occupied (VO) blocks of a transformation matrix.
 *
 * This function extracts two specific blocks from the transformation matrix U:
 * - The OV block: rows [0, num_occupied_orbitals) and columns
 * [num_occupied_orbitals, num_occupied_orbitals+num_virtual_orbitals)
 * - The VO block: rows [num_occupied_orbitals,
 * num_occupied_orbitals+num_virtual_orbitals) and columns [0,
 * num_occupied_orbitals)
 *
 * @param num_occupied_orbitals Number of occupied orbitals
 * @param num_virtual_orbitals Number of virtual orbitals
 * @param U The transformation matrix of size
 * (num_occupied_orbitals+num_virtual_orbitals) x
 * (num_occupied_orbitals+num_virtual_orbitals)
 * @return std::pair<double, double> A pair containing:
 *         - first: Frobenius norm of the occupied-virtual (OV) block
 *         - second: Frobenius norm of the virtual-occupied (VO) block
 */
auto ov_block_norms(size_t num_occupied_orbitals, size_t num_virtual_orbitals,
                    const Eigen::MatrixXd& U) {
  const auto U_ov = U.block(0, num_occupied_orbitals, num_occupied_orbitals,
                            num_virtual_orbitals);
  const auto U_vo = U.block(num_occupied_orbitals, 0, num_virtual_orbitals,
                            num_occupied_orbitals);
  return std::make_pair(U_ov.norm(), U_vo.norm());
}

}  // namespace testing
