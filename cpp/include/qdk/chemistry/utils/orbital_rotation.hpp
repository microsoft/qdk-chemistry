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
 * This function takes QATK orbitals and applies orbital rotations using a
 * rotation vector, typically taken from stability analysis eigenvectors.
 *
 * The rotation is performed by:
 * 1. Unpacking the rotation vector into an anti-Hermitian matrix
 * 2. Computing the unitary rotation matrix via matrix exponential
 * 3. Applying the rotation to the molecular orbital coefficients
 *
 * @param orbitals The QATK orbitals to rotate
 * @param rotation_vector The rotation vector (typically from stability
 *                        analysis, corresponding to the lowest eigenvalue)
 * @param num_alpha_occupied_orbitals Number of alpha occupied orbitals
 * @param num_beta_occupied_orbitals Number of beta occupied orbitals
 * @param restricted_external If true and orbitals are restricted, creates
 *                            unrestricted orbitals with rotated coefficients
 *                            for alpha spin and unrotated coefficients for
 *                            beta spin. Default is false.
 * @return A new QATK Orbitals object with rotated molecular orbital
 *         coefficients
 *
 * @note restricted_external can break spin symmetry and solve external
 * instabilities of RHF/RKS.
 * @note For unrestricted calculations, the rotation vector should contain
 *       alpha rotations first (n_occ_alpha * n_vir_alpha elements),
 *       then beta rotations (n_occ_beta * n_vir_beta elements).
 * @note This function assumes aufbau filling for occupation numbers.
 * @note Orbital energies are invalidated by rotation and set to null.
 *
 * @throws std::runtime_error if rotation vector size is invalid
 *
 * Example usage:
 * @code
 * // After stability analysis that finds instability
 * auto rotation_vector = ...; // eigenvector from stability analysis
 * auto rotated_orbitals = rotate_orbitals(orbitals, rotation_vector,
 *                                              num_alpha_occupied_orbitals,
 * num_beta_occupied_orbitals);
 * @endcode
 */
std::shared_ptr<qdk::chemistry::data::Orbitals> rotate_orbitals(
    std::shared_ptr<const qdk::chemistry::data::Orbitals> orbitals,
    const Eigen::VectorXd& rotation_vector, size_t num_alpha_occupied_orbitals,
    size_t num_beta_occupied_orbitals, bool restricted_external = false);

/**
 * @brief Apply orbital rotation to molecular orbital coefficients.
 *
 * This is the core rotation function that:
 * 1. Unpacks the rotation vector into an anti-Hermitian matrix using mask
 * 2. Computes the unitary rotation matrix U = exp(dr)
 * 3. Applies the rotation: mo_coeff_new = mo_coeff * U
 *
 * @param mo_coeff Molecular orbital coefficient matrix [n_ao x n_mo]
 * @param rotation_vector Rotation vector (unique variables)
 * @param mask Double mask indicating which matrix elements to fill from
 * rotation_vector
 * @return Rotated molecular orbital coefficients [n_ao x n_mo]
 */
Eigen::MatrixXd rotate_mo(const Eigen::MatrixXd& mo_coeff,
                          const Eigen::VectorXd& rotation_vector,
                          const Eigen::MatrixXd& mask);

/**
 * @brief Run SCF with iterative stability checking and orbital rotation
 * workflow.
 *
 * This workflow function iteratively performs:
 * 1. Run SCF calculation
 * 2. Check stability of resulting wavefunction (internal and external for
 * restricted)
 * 3. If internally unstable, rotate orbitals and restart SCF
 * 4. If externally unstable (restricted only), break spin symmetry and switch
 * to unrestricted
 * 5. Repeat until stable or max_stability_iterations is reached
 *
 * For restricted calculations (RHF/ROHF), both internal and external stability
 * are checked. If external instability is detected, the workflow automatically
 * switches to unrestricted (UHF) by rotating alpha orbitals while keeping beta
 * orbitals unchanged, effectively breaking spin symmetry.
 *
 * @param structure The molecular structure
 * @param charge The molecular charge
 * @param spin_multiplicity The spin multiplicity
 * @param scf_solver Pre-configured SCF solver instance (settings should be
 * configured before passing)
 * @param stability_checker Pre-configured stability checker instance (settings
 * should be configured before passing)
 * @param initial_guess Optional initial orbital guess for the first SCF
 * calculation
 * @param max_stability_iterations Maximum number of stability check and
 * rotation cycles (default: 5)
 *
 * @return A tuple containing:
 *         - double: Final SCF energy in Hartree
 *         - std::shared_ptr<data::Wavefunction>: Final wavefunction from the
 * last SCF cycle
 *         - bool: Overall stability status (true if stable, false if unstable
 * after max iterations)
 *         - std::shared_ptr<data::StabilityResult>: Detailed stability result
 * from the last cycle
 *
 * @note For restricted wavefunctions, external stability checks are
 * automatically enabled
 * @note Internal instabilities are resolved first before checking external
 * stability
 * @note External instabilities trigger automatic RHF→UHF transition
 * @note After switching to unrestricted, only internal stability is checked
 * @note When external instability is detected and switching to unrestricted,
 * new solver instances are created by copying the original solvers' settings
 *
 * @throws std::runtime_error If SCF or stability check fails
 * @throws std::invalid_argument If input parameters are invalid
 *
 * Example usage:
 * @code
 * auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
 * scf_solver->settings().set("reference_type", "auto");
 * auto stability_checker =
 * algorithms::StabilityCheckerFactory::create("pyscf");
 * stability_checker->settings().set("stability_tolerance", -1e-4);
 * stability_checker->settings().set("davidson_tolerance", 1e-4);
 * stability_checker->settings().set("nroots", 3);
 *
 * auto [energy, wfn, is_stable, result] = run_scf_with_stability_workflow(
 *     structure, 0, 1, scf_solver, stability_checker, std::nullopt, 5);
 *
 * if (is_stable) {
 *   std::cout << "Converged to stable wavefunction with energy " << energy <<
 * std::endl; } else { std::cout << "Max iterations reached, still unstable" <<
 * std::endl;
 * }
 * @endcode
 */
std::tuple<double, std::shared_ptr<qdk::chemistry::data::Wavefunction>, bool,
           std::shared_ptr<qdk::chemistry::data::StabilityResult>>
run_scf_with_stability_workflow(
    std::shared_ptr<qdk::chemistry::data::Structure> structure, int charge,
    int spin_multiplicity,
    std::shared_ptr<qdk::chemistry::algorithms::ScfSolver> scf_solver,
    std::shared_ptr<qdk::chemistry::algorithms::StabilityChecker>
        stability_checker,
    std::optional<std::shared_ptr<qdk::chemistry::data::Orbitals>>
        initial_guess = std::nullopt,
    int max_stability_iterations = 5);

}  // namespace qdk::chemistry::utils
