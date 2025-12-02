// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <spdlog/spdlog.h>

#include <blas.hh>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/algorithms/stability.hpp>
#include <qdk/chemistry/utils/orbital_rotation.hpp>
#include <stdexcept>

#include "../algorithms/microsoft/scf/src/util/matrix_exp.h"

namespace qdk::chemistry::utils {

using namespace qdk::chemistry::data;
using BoolMatrix = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;

namespace detail {

// Helper function to create the mask for unique orbital rotation variables
// Following PySCF's uniq_var_indices logic
BoolMatrix create_rotation_mask(size_t num_molecular_orbitals,
                                size_t num_alpha_occupied,
                                size_t num_beta_occupied) {
  BoolMatrix mask =
      BoolMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);

  // occidxa: orbitals with alpha occupation (0 to num_alpha_occupied-1)
  // viridxa: orbitals without alpha occupation (num_alpha_occupied to
  // num_molecular_orbitals-1)

  // It is union of two rectangular blocks if num_alpha_occupied !=
  // num_beta_occupied
  for (size_t j = 0; j < num_molecular_orbitals; ++j) {
    for (size_t i = 0; i < num_molecular_orbitals; ++i) {
      bool viridxa = (i >= num_alpha_occupied);
      bool occidxa = (j < num_alpha_occupied);
      bool viridxb = (i >= num_beta_occupied);
      bool occidxb = (j < num_beta_occupied);

      mask(i, j) = (viridxa && occidxa) || (viridxb && occidxb);
    }
  }

  return mask;
}

// Unpack rotation vector into full anti-Hermitian matrix using mask
// Following PySCF's unpack_uniq_var logic
Eigen::MatrixXd unpack_rotation_vector(const Eigen::VectorXd& rotation_vector,
                                       const BoolMatrix& mask) {
  const size_t num_molecular_orbitals = mask.rows();

  // Count expected size from mask (true entries)
  size_t expected_size = mask.count();
  if (static_cast<size_t>(rotation_vector.size()) != expected_size) {
    throw std::runtime_error("Rotation vector size mismatch: expected " +
                             std::to_string(expected_size) + " elements, got " +
                             std::to_string(rotation_vector.size()));
  }

  Eigen::MatrixXd dr =
      Eigen::MatrixXd::Zero(num_molecular_orbitals, num_molecular_orbitals);

  // Fill masked positions with rotation vector elements and make anti-Hermitian
  // Note rotation_vector from pyscf is in row-major order
  size_t idx = 0;
  for (size_t i = 0; i < num_molecular_orbitals; ++i) {
    for (size_t j = 0; j < num_molecular_orbitals; ++j) {
      if (mask(i, j)) {
        double val = rotation_vector(idx++);
        dr(i, j) = val;
        dr(j, i) = -val;  // Anti-Hermitian: dr(j,i) = -dr(i,j)
      }
    }
  }

  return dr;
}
}  // namespace detail

/**
 * @brief Apply orbital rotation to molecular orbital coefficients.
 *
 * @param mo_coeff Molecular orbital coefficient matrix [n_ao x n_mo]
 * @param rotation_vector Rotation vector (unique variables)
 * @param mask Boolean mask indicating which matrix elements to fill from
 * rotation_vector
 * @return Rotated molecular orbital coefficients [n_ao x n_mo]
 */
Eigen::MatrixXd apply_orbital_rotation(const Eigen::MatrixXd& mo_coeff,
                                       const Eigen::VectorXd& rotation_vector,
                                       const BoolMatrix& mask) {
  // Unpack rotation vector using mask
  Eigen::MatrixXd dr = detail::unpack_rotation_vector(rotation_vector, mask);

  // Compute unitary rotation matrix via matrix exponential
  const int num_molecular_orbitals = static_cast<int>(dr.cols());
  Eigen::MatrixXd u =
      Eigen::MatrixXd::Zero(num_molecular_orbitals, num_molecular_orbitals);
  qdk::chemistry::scf::matrix_exp(dr.data(), u.data(), num_molecular_orbitals);

  // Apply rotation to MO coefficients using BLAS
  Eigen::MatrixXd rotated_coeff(mo_coeff.rows(), num_molecular_orbitals);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             mo_coeff.rows(), num_molecular_orbitals, mo_coeff.cols(), 1.0,
             mo_coeff.data(), mo_coeff.rows(), u.data(), u.rows(), 0.0,
             rotated_coeff.data(), rotated_coeff.rows());

  return rotated_coeff;
}

std::shared_ptr<Orbitals> rotate_orbitals(
    std::shared_ptr<const Orbitals> orbitals,
    const Eigen::VectorXd& rotation_vector, size_t num_alpha_occupied_orbitals,
    size_t num_beta_occupied_orbitals, bool restricted_external) {
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  if (orbitals->is_restricted()) {
    // Restricted case - could be RHF or ROHF
    const Eigen::MatrixXd& mo_coeff = orbitals->get_coefficients_alpha();

    // Create mask for allowed rotations
    auto mask = detail::create_rotation_mask(num_molecular_orbitals,
                                             num_alpha_occupied_orbitals,
                                             num_beta_occupied_orbitals);

    if (restricted_external) {
      // Restricted -> Unrestricted: rotated alpha, unrotated beta
      Eigen::MatrixXd rotated_coeff_alpha =
          apply_orbital_rotation(mo_coeff, rotation_vector, mask);
      const Eigen::MatrixXd& rotated_coeff_beta = mo_coeff;

      // Create new Orbitals object with unrestricted coefficients
      auto rotated_orbitals = std::make_shared<Orbitals>(
          rotated_coeff_alpha, rotated_coeff_beta,
          std::nullopt,  // energies_alpha invalidated
          std::nullopt,  // energies_beta invalidated
          orbitals->has_overlap_matrix()
              ? std::optional<Eigen::MatrixXd>(orbitals->get_overlap_matrix())
              : std::nullopt,
          orbitals->has_basis_set() ? orbitals->get_basis_set() : nullptr);

      return rotated_orbitals;
    } else {
      // Restricted case - single rotation
      Eigen::MatrixXd rotated_coeff =
          apply_orbital_rotation(mo_coeff, rotation_vector, mask);

      // Create new Orbitals object with rotated coefficients
      // Energies are invalidated by rotation
      auto rotated_orbitals = std::make_shared<Orbitals>(
          rotated_coeff,
          std::nullopt,  // energies invalidated
          orbitals->has_overlap_matrix()
              ? std::optional<Eigen::MatrixXd>(orbitals->get_overlap_matrix())
              : std::nullopt,
          orbitals->has_basis_set() ? orbitals->get_basis_set() : nullptr);

      return rotated_orbitals;
    }

  } else {
    // Unrestricted case - separate alpha and beta rotations
    // Rotation vector contains alpha rotations first, then beta
    const size_t num_alpha_virtual_orbitals =
        num_molecular_orbitals - num_alpha_occupied_orbitals;
    const size_t num_beta_virtual_orbitals =
        num_molecular_orbitals - num_beta_occupied_orbitals;
    const size_t alpha_size =
        num_alpha_occupied_orbitals * num_alpha_virtual_orbitals;
    const size_t beta_size =
        num_beta_occupied_orbitals * num_beta_virtual_orbitals;

    if (static_cast<size_t>(rotation_vector.size()) != alpha_size + beta_size) {
      throw std::invalid_argument(
          "Rotation vector size does not match expected size for unrestricted "
          "calculation");
    }

    // Split rotation vector
    Eigen::VectorXd rotation_alpha = rotation_vector.head(alpha_size);
    Eigen::VectorXd rotation_beta = rotation_vector.tail(beta_size);

    // Get alpha and beta coefficients
    const Eigen::MatrixXd& mo_coeff_alpha = orbitals->get_coefficients_alpha();
    const Eigen::MatrixXd& mo_coeff_beta = orbitals->get_coefficients_beta();

    // Create masks for alpha and beta channels (UHF: simple rectangular)
    auto mask_alpha = detail::create_rotation_mask(
        num_molecular_orbitals, num_alpha_occupied_orbitals,
        num_alpha_occupied_orbitals);  // UHF: alpha electrons only
    auto mask_beta = detail::create_rotation_mask(
        num_molecular_orbitals, num_beta_occupied_orbitals,
        num_beta_occupied_orbitals);  // UHF: beta electrons only

    // Rotate both spin channels
    Eigen::MatrixXd rotated_coeff_alpha =
        apply_orbital_rotation(mo_coeff_alpha, rotation_alpha, mask_alpha);
    Eigen::MatrixXd rotated_coeff_beta =
        apply_orbital_rotation(mo_coeff_beta, rotation_beta, mask_beta);

    // Create new Orbitals object with rotated coefficients
    auto rotated_orbitals = std::make_shared<Orbitals>(
        rotated_coeff_alpha, rotated_coeff_beta,
        std::nullopt,  // energies_alpha invalidated
        std::nullopt,  // energies_beta invalidated
        orbitals->has_overlap_matrix()
            ? std::optional<Eigen::MatrixXd>(orbitals->get_overlap_matrix())
            : std::nullopt,
        orbitals->has_basis_set() ? orbitals->get_basis_set() : nullptr);

    return rotated_orbitals;
  }
}

}  // namespace qdk::chemistry::utils
