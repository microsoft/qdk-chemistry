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
 * rotation vector, which can be taken from stability analysis eigenvectors.
 *
 * The rotation is performed by:
 * 1. Unpacking the rotation vector into an anti-Hermitian matrix
 * 2. Computing the unitary rotation matrix via matrix exponential
 * 3. Applying the rotation to the molecular orbital coefficients
 *
 * @param orbitals The Orbitals to rotate
 * @param rotation_vector The rotation vector (eigenvector from stability
 *                        analysis, corresponding to the lowest eigenvalue).
 *                        See StabilityResult::eigenvector_format for
 *                        detailed size and indexing requirements.
 * @param num_alpha_occupied_orbitals Number of alpha occupied orbitals
 * @param num_beta_occupied_orbitals Number of beta occupied orbitals
 * @param restricted_external If true and orbitals are restricted, creates
 *                            unrestricted orbitals with rotated coefficients
 *                            for alpha spin and unrotated coefficients for
 *                            beta spin. Default is false.
 * @return A new Orbitals object with rotated molecular orbital
 *         coefficients
 *
 * @note restricted_external can break spin symmetry and solve external
 * instabilities of RHF/RKS.
 * @note This function assumes aufbau filling for occupation numbers.
 * @note Orbital energies are invalidated by rotation and set to null.
 *
 * @throws std::runtime_error if rotation vector size is invalid
 *
 * @see data::StabilityResult
 */
std::shared_ptr<qdk::chemistry::data::Orbitals> rotate_orbitals(
    std::shared_ptr<const qdk::chemistry::data::Orbitals> orbitals,
    const Eigen::VectorXd& rotation_vector, size_t num_alpha_occupied_orbitals,
    size_t num_beta_occupied_orbitals, bool restricted_external = false);

}  // namespace qdk::chemistry::utils
