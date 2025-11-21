// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <spdlog/spdlog.h>

#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/algorithms/stability.hpp>
#include <qdk/chemistry/utils/orbital_rotation.hpp>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

namespace qdk::chemistry::utils {

using namespace qdk::chemistry::data;

namespace {
// Helper function to create the mask for unique orbital rotation variables
// Following PySCF's uniq_var_indices logic
Eigen::MatrixXd create_rotation_mask(size_t num_molecular_orbitals,
                                     size_t num_alpha_occupied,
                                     size_t num_beta_occupied) {
  Eigen::MatrixXd mask =
      Eigen::MatrixXd::Zero(num_molecular_orbitals, num_molecular_orbitals);

  // occidxa: orbitals with alpha occupation (0 to num_alpha_occupied-1)
  // viridxa: orbitals without alpha occupation (num_alpha_occupied to
  // num_molecular_orbitals-1) occidxb: orbitals with beta occupation (0 to
  // num_beta_occupied-1) viridxb: orbitals without beta occupation
  // (num_beta_occupied to num_molecular_orbitals-1)

  // mask = (viridxa[:,None] & occidxa) | (viridxb[:,None] & occidxb)
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
                                       const Eigen::MatrixXd& mask) {
  const size_t num_molecular_orbitals = mask.rows();

  // Count expected size from mask (non-zero entries). MatrixXd lacks count(),
  // so we compare against zero to form a boolean array and count true values.
  size_t expected_size = (mask.array() != 0.0).count();
  if (static_cast<size_t>(rotation_vector.size()) != expected_size) {
    throw std::runtime_error("Rotation vector size mismatch: expected " +
                             std::to_string(expected_size) + " elements, got " +
                             std::to_string(rotation_vector.size()));
  }

  Eigen::MatrixXd dr =
      Eigen::MatrixXd::Zero(num_molecular_orbitals, num_molecular_orbitals);

  // Fill masked positions with rotation vector elements
  // Note rotation_vector from pyscf is in row-major order
  size_t idx = 0;
  for (size_t i = 0; i < num_molecular_orbitals; ++i) {
    for (size_t j = 0; j < num_molecular_orbitals; ++j) {
      if (mask(i, j)) {
        dr(i, j) = rotation_vector(idx++);
      }
    }
  }

  // Make anti-Hermitian: dr = dr - dr^T
  dr = dr - dr.transpose();

  return dr;
}
}  // anonymous namespace

Eigen::MatrixXd rotate_mo(const Eigen::MatrixXd& mo_coeff,
                          const Eigen::VectorXd& rotation_vector,
                          const Eigen::MatrixXd& mask) {
  // Unpack rotation vector using mask
  Eigen::MatrixXd dr = unpack_rotation_vector(rotation_vector, mask);

  // Compute unitary rotation matrix via matrix exponential
  Eigen::MatrixXd u = dr.exp();

  // Apply rotation to MO coefficients
  Eigen::MatrixXd rotated_coeff = mo_coeff * u;

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
    Eigen::MatrixXd mask = create_rotation_mask(num_molecular_orbitals,
                                                num_alpha_occupied_orbitals,
                                                num_beta_occupied_orbitals);

    if (restricted_external) {
      // Restricted -> Unrestricted: rotated alpha, unrotated beta
      Eigen::MatrixXd rotated_coeff_alpha =
          rotate_mo(mo_coeff, rotation_vector, mask);
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
          rotate_mo(mo_coeff, rotation_vector, mask);

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
    Eigen::MatrixXd mask_alpha = create_rotation_mask(
        num_molecular_orbitals, num_alpha_occupied_orbitals,
        num_alpha_occupied_orbitals);  // UHF: alpha electrons only
    Eigen::MatrixXd mask_beta = create_rotation_mask(
        num_molecular_orbitals, num_beta_occupied_orbitals,
        num_beta_occupied_orbitals);  // UHF: beta electrons only

    // Rotate both spin channels
    Eigen::MatrixXd rotated_coeff_alpha =
        rotate_mo(mo_coeff_alpha, rotation_alpha, mask_alpha);
    Eigen::MatrixXd rotated_coeff_beta =
        rotate_mo(mo_coeff_beta, rotation_beta, mask_beta);

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

std::tuple<double, std::shared_ptr<data::Wavefunction>, bool,
           std::shared_ptr<data::StabilityResult>>
run_scf_with_stability_workflow(
    std::shared_ptr<data::Structure> structure, int charge,
    int spin_multiplicity, const std::string& scf_solver_name,
    const std::string& stability_checker_name,
    std::optional<std::shared_ptr<data::Orbitals>> initial_guess,
    int max_stability_iterations, double stability_tolerance,
    const std::string& reference_type) {
  if (!structure) {
    throw std::invalid_argument("Structure cannot be null");
  }
  if (max_stability_iterations < 1) {
    throw std::invalid_argument("max_stability_iterations must be at least 1");
  }

  // Create SCF solver instance
  auto scf_solver = algorithms::ScfSolverFactory::create(scf_solver_name);
  if (!scf_solver) {
    throw std::runtime_error("Failed to create SCF solver: " + scf_solver_name);
  }

  // Configure reference_type
  if (reference_type != "auto") {
    scf_solver->settings().set("reference_type", reference_type);
    spdlog::info(
        "SCF solver configured with reference_type={} at first iteration",
        reference_type);
  }

  // Create stability checker instance
  auto stability_checker =
      algorithms::StabilityCheckerFactory::create(stability_checker_name);
  if (!stability_checker) {
    throw std::runtime_error("Failed to create stability checker: " +
                             stability_checker_name);
  }

  stability_checker->settings().set("stability_tolerance", stability_tolerance);

  spdlog::info(
      "Starting SCF with stability workflow: scf_solver={}, "
      "stability_checker={}, max_iterations={}, tolerance={}",
      scf_solver_name, stability_checker_name, max_stability_iterations,
      stability_tolerance);

  // Run initial SCF calculation
  auto [energy, wavefunction] =
      scf_solver->run(structure, charge, spin_multiplicity, initial_guess);

  spdlog::info("Initial SCF energy: {} Hartree", energy);

  // Determine if calculation is restricted from initial wavefunction
  bool is_restricted_calculation =
      wavefunction->get_orbitals()->is_restricted() && spin_multiplicity == 1;

  // Configure stability checker based on calculation type
  if (is_restricted_calculation) {
    stability_checker->settings().set("external", true);
    spdlog::info(
        "Restricted calculation detected: enabling both internal and external "
        "stability checks");
  } else {
    stability_checker->settings().set("external", false);
    spdlog::info(
        "Unrestricted calculation detected: checking internal stability only");
  }

  std::shared_ptr<data::StabilityResult> stability_result;
  bool is_stable = false;
  int iteration = 0;

  // Iterative stability check and orbital rotation
  while (iteration < max_stability_iterations) {
    iteration++;
    spdlog::info("Stability check iteration {}/{}", iteration,
                 max_stability_iterations);

    // Perform stability analysis
    auto [stable, result] = stability_checker->run(wavefunction);
    stability_result = result;
    is_stable = stable;

    if (is_stable) {
      spdlog::info("Wavefunction is stable at iteration {}", iteration);
      break;
    }

    // Check if instability is significant
    double smallest_eigenvalue = stability_result->get_smallest_eigenvalue();
    spdlog::info("Smallest eigenvalue: {}", smallest_eigenvalue);

    if (smallest_eigenvalue >= stability_tolerance) {
      spdlog::info("Eigenvalue {} is above tolerance ({}), considering stable",
                   smallest_eigenvalue, stability_tolerance);
      is_stable = true;
      break;
    }

    // Last iteration check - don't rotate if we've reached the limit
    if (iteration >= max_stability_iterations) {
      spdlog::warn(
          "Maximum stability iterations ({}) reached, wavefunction remains "
          "unstable",
          max_stability_iterations);
      break;
    }

    double eigenvalue;
    Eigen::VectorXd rotation_vector;
    bool do_external = false;
    // Get the rotation vector corresponding to the smallest eigenvalue
    if (!stability_result->is_internal_stable()) {
      // First solve internal stability if present
      std::tie(eigenvalue, rotation_vector) =
          stability_result->get_smallest_internal_eigenvalue_and_vector();
      spdlog::info(
          "Internal instability detected (eigenvalue={}). Rotating orbitals "
          "and restarting SCF...",
          eigenvalue);
    } else if (!stability_result->is_external_stable() &&
               stability_result->has_external_result()) {
      std::tie(eigenvalue, rotation_vector) =
          stability_result->get_smallest_external_eigenvalue_and_vector();
      do_external = true;
      spdlog::info(
          "External instability detected (eigenvalue={}). Rotating orbitals "
          "and restarting SCF...",
          eigenvalue);
    } else {
      spdlog::warn(
          "Unstable wavefunction but no instability found in results. This "
          "may indicate a logic error.");
      break;
    }

    // Get occupation numbers from wavefunction
    auto orbitals = wavefunction->get_orbitals();
    auto [num_alpha_electrons, num_beta_electrons] =
        wavefunction->get_total_num_electrons();

    // Rotate the orbitals
    auto rotated_orbitals =
        rotate_orbitals(orbitals, rotation_vector, num_alpha_electrons,
                        num_beta_electrons, do_external);

    // If external instability, switch to unrestricted and disable external
    // checks
    if (do_external) {
      spdlog::info(
          "Breaking spin symmetry: switching to unrestricted calculation. "
          "Stability checker reconfigured for internal-only analysis");
      is_restricted_calculation = false;
      // Since settings are locked, we create new scf_solver and
      // stability_checker
      scf_solver = algorithms::ScfSolverFactory::create(scf_solver_name);
      scf_solver->settings().set("reference_type", "unrestricted");
      stability_checker =
          algorithms::StabilityCheckerFactory::create(stability_checker_name);
      stability_checker->settings().set("external", false);
      stability_checker->settings().set("stability_tolerance",
                                        stability_tolerance);
    }

    // Restart SCF with rotated orbitals
    auto [new_energy, new_wavefunction] =
        scf_solver->run(structure, charge, spin_multiplicity, rotated_orbitals);

    spdlog::info("SCF after rotation: energy = {} Hartree (previous: {})",
                 new_energy, energy);

    if (new_energy < energy) {
      spdlog::info("Energy decreased by {} Hartree", energy - new_energy);
    } else {
      spdlog::warn(
          "Energy increased by {} Hartree after rotation (this may indicate "
          "convergence issues)",
          new_energy - energy);
    }

    // Update for next iteration
    energy = new_energy;
    wavefunction = new_wavefunction;
  }

  if (is_stable) {
    spdlog::info(
        "Stability workflow completed successfully: converged to stable "
        "wavefunction after {} iteration(s)",
        iteration);
  } else {
    spdlog::warn(
        "Stability workflow completed: wavefunction remains unstable after {} "
        "iteration(s)",
        iteration);
  }

  return std::make_tuple(energy, wavefunction, is_stable, stability_result);
}

}  // namespace qdk::chemistry::utils
