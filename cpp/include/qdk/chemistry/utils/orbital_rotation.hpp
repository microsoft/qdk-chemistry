// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/algorithms/stability.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

namespace qdk::chemistry::utils {

/**
 * @brief Rotate molecular orbitals using a rotation vector.
 *
 * This function takes Orbitals and applies orbital rotations using a
 * rotation vector, typically taken from stability analysis eigenvectors.
 *
 * The rotation is performed by:
 * 1. Unpacking the rotation vector into an anti-Hermitian matrix
 * 2. Computing the unitary rotation matrix via matrix exponential
 * 3. Applying the rotation to the molecular orbital coefficients
 *
 * @param orbitals The Orbitals to rotate
 * @param rotation_vector The rotation vector (typically from stability
 *                        analysis, corresponding to the lowest eigenvalue).
 *                        See detailed size requirements below.
 * @param num_alpha_occupied_orbitals Number of alpha occupied orbitals
 * @param num_beta_occupied_orbitals Number of beta occupied orbitals
 * @param restricted_external If true and orbitals are restricted, creates
 *                            unrestricted orbitals with rotated coefficients
 *                            for alpha spin and unrotated coefficients for
 *                            beta spin. Default is false.
 * @return A new Orbitals object with rotated molecular orbital
 *         coefficients
 *
 * @section rotation_vector_format Rotation Vector Format
 *
 * The rotation_vector encodes orbital rotation parameters between occupied
 * and virtual orbitals. The required size depends on the orbital type:
 *
 * **RHF (Restricted Hartree-Fock):**
 * - Size: num_occupied_orbitals * num_virtual_orbitals
 * - Where: num_virtual_orbitals = num_molecular_orbitals -
 *                                 num_occupied_orbitals
 * - Elements represent rotations between occupied and virtual spatial orbitals
 * - Both spins rotate together (spin symmetry preserved)
 *
 * **UHF (Unrestricted Hartree-Fock):**
 * - Size: num_alpha_occupied_orbitals * num_alpha_virtual_orbitals +
 *         num_beta_occupied_orbitals * num_beta_virtual_orbitals
 * - Where: num_alpha_virtual_orbitals = num_molecular_orbitals -
 *                                       num_alpha_occupied_orbitals
 *          num_beta_virtual_orbitals = num_molecular_orbitals -
 *                                      num_beta_occupied_orbitals
 * - First num_alpha_occupied_orbitals * num_alpha_virtual_orbitals elements:
 *   alpha rotations
 * - Last num_beta_occupied_orbitals * num_beta_virtual_orbitals elements:
 *   beta rotations
 * - Alpha and beta orbitals rotate independently
 *
 * **ROHF (Restricted Open-shell Hartree-Fock):**
 * - The rotation mask is the union of two rectangular blocks:
 *   1. Alpha block: num_alpha_occupied_orbitals * num_alpha_virtual_orbitals
 *   2. Beta block: num_beta_occupied_orbitals * num_beta_virtual_orbitals
 * - Size calculation for union (assuming num_alpha_occupied >=
 * num_beta_occupied): num_alpha_occupied_orbitals * (num_molecular_orbitals -
 * num_alpha_occupied_orbitals) + (num_alpha_occupied_orbitals -
 * num_beta_occupied_orbitals) * num_beta_occupied_orbitals
 * - This equals the virtual-occupied block plus the additional
 *          doubly-occupied to singly-occupied block
 *
 * The rotation vector elements are typically ordered in row-major format,
 * corresponding to the flattened occupied-virtual rotation matrix.
 *
 * @note restricted_external can break spin symmetry and solve external
 * instabilities of RHF/RKS.
 * @note This function assumes aufbau filling for occupation numbers.
 * @note Orbital energies are invalidated by rotation and set to null.
 *
 * @throws std::runtime_error if rotation vector size is invalid
 *
 * Example usage:
 * @code
 * // RHF case: 5 occupied, 10 total orbitals
 * size_t n_occ = 5, n_vir = 5;
 * Eigen::VectorXd rotation_vector(n_occ * n_vir); // size = 25
 * // ... populate from stability analysis ...
 * auto rotated = rotate_orbitals(orbitals, rotation_vector, n_occ, n_occ);
 *
 * // UHF case: 5 alpha occupied, 3 beta occupied, 10 total orbitals
 * size_t n_occ_a = 5, n_occ_b = 3, num_molecular_orbitals = 10;
 * size_t n_vir_a = num_molecular_orbitals - n_occ_a; // 5
 * size_t n_vir_b = num_molecular_orbitals - n_occ_b; // 7
 * Eigen::VectorXd rotation_vector(n_occ_a * n_vir_a + n_occ_b * n_vir_b);
 * // size = 5*5 + 3*7 = 46
 * // First 25 elements: alpha rotations, next 21: beta rotations
 * auto rotated = rotate_orbitals(orbitals, rotation_vector, n_occ_a, n_occ_b);
 * @endcode
 */
std::shared_ptr<qdk::chemistry::data::Orbitals> rotate_orbitals(
    std::shared_ptr<const qdk::chemistry::data::Orbitals> orbitals,
    const Eigen::VectorXd& rotation_vector, size_t num_alpha_occupied_orbitals,
    size_t num_beta_occupied_orbitals, bool restricted_external = false);

}  // namespace qdk::chemistry::utils
