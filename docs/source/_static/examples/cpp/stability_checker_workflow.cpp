// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Stability Checker workflow example with orbital rotation.
// --------------------------------------------------------------------------------------------
// start-cell-create
#include <iomanip>
#include <iostream>
#include <qdk/chemistry.hpp>
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/utils/orbital_rotation.hpp>
#include <string>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::utils;

// Create the default StabilityChecker instance
auto stability_checker = StabilityCheckerFactory::create();
// end-cell-create
// --------------------------------------------------------------------------------------------

int main() {
  // --------------------------------------------------------------------------------------------
  // start-cell-configure
  // Configure stability checker settings
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set(
      "external", true);  // Will be adjusted based on calculation type
  stability_checker->settings().set("stability_tolerance", -1e-4);
  stability_checker->settings().set("davidson_tolerance", 1e-4);
  stability_checker->settings().set("max_subspace", static_cast<int64_t>(30));
  stability_checker->settings().set("method", "hf");

  // end-cell-configure
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-run
  // Create N2 molecule at stretched geometry (1.4 Angstrom)
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}};
  std::vector<std::string> symbols = {"N", "N"};
  for (auto& coord : coords) {
    coord *= qdk::chemistry::constants::angstrom_to_bohr;
  }
  auto n2 = std::make_shared<Structure>(coords, symbols);

  // Create and configure SCF solver with auto scf_type
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", "auto");
  scf_solver->settings().set("method", "hf");

  // Run initial SCF calculation
  int spin_multiplicity = 1;
  auto [energy, wavefunction] =
      scf_solver->run(n2, 0, spin_multiplicity, "def2-svp");
  std::cout << "Initial SCF Energy: " << std::fixed << std::setprecision(10)
            << energy << " Hartree" << std::endl;

  // Determine if calculation is restricted and configure stability checker
  // accordingly
  bool is_restricted =
      wavefunction->get_orbitals()->is_restricted() && spin_multiplicity == 1;
  if (is_restricted) {
    stability_checker->settings().set("external", true);
  } else {
    stability_checker->settings().set("external", false);
  }

  // Iterative workflow: Check stability and rotate orbitals until convergence
  const int max_iterations = 5;
  int iteration = 0;
  bool is_stable = false;
  std::shared_ptr<StabilityResult> result;

  std::cout << "\n=== Starting Iterative Stability Workflow ===\n" << std::endl;

  while (iteration < max_iterations) {
    iteration++;
    std::cout << "--- Iteration " << iteration << " ---" << std::endl;
    std::cout << "Current Energy: " << std::fixed << std::setprecision(10)
              << energy << " Hartree" << std::endl;

    // Check stability - handle potential Davidson convergence failure
    try {
      auto [stable, stability_result] = stability_checker->run(wavefunction);
      is_stable = stable;
      result = stability_result;
    } catch (const std::runtime_error& e) {
      std::string error_msg = e.what();
      if (error_msg.find("Davidson Did Not Converge!") != std::string::npos) {
        std::cout
            << "Try increasing max_subspace or adjusting davidson_tolerance"
            << std::endl;
        throw std::runtime_error("Davidson Did Not Converge!");
      } else {
        throw std::runtime_error("Stability check failed: " + error_msg);
      }
    }

    if (is_stable) {
      std::cout << "\nConverged to stable wavefunction!" << std::endl;
      break;
    }

    // Determine rotation type based on which instability is present
    bool do_external = false;
    double smallest_eigenvalue;
    Eigen::VectorXd rotation_vector;

    if (!result->is_internal_stable()) {
      auto [eigenvalue, eigenvector] =
          result->get_smallest_internal_eigenvalue_and_vector();
      smallest_eigenvalue = eigenvalue;
      rotation_vector = eigenvector;
      std::cout << "Internal instability detected. Smallest eigenvalue: "
                << std::fixed << std::setprecision(6) << smallest_eigenvalue
                << std::endl;
    } else if (!result->is_external_stable() && result->has_external_result()) {
      auto [eigenvalue, eigenvector] =
          result->get_smallest_external_eigenvalue_and_vector();
      smallest_eigenvalue = eigenvalue;
      rotation_vector = eigenvector;
      do_external = true;
      std::cout << "External instability detected. Smallest eigenvalue: "
                << std::fixed << std::setprecision(6) << smallest_eigenvalue
                << std::endl;
    } else {
      throw std::runtime_error(
          "Unexpected state: neither internal nor external instability "
          "detected");
    }

    // Rotate orbitals along the instability direction
    auto [num_alpha, num_beta] = wavefunction->get_total_num_electrons();
    auto orbitals = wavefunction->get_orbitals();
    auto rotated_orbitals = rotate_orbitals(orbitals, rotation_vector,
                                            num_alpha, num_beta, do_external);

    // If external instability detected, switch to unrestricted calculation
    if (do_external) {
      std::cout
          << "Switching to unrestricted calculation due to external instability"
          << std::endl;

      // Get current settings values to preserve them
      std::string method = scf_solver->settings().get<std::string>("method");
      std::string stab_method =
          stability_checker->settings().get<std::string>("method");
      double davidson_tol =
          stability_checker->settings().get<double>("davidson_tolerance");
      double stability_tol =
          stability_checker->settings().get<double>("stability_tolerance");
      int64_t max_sub =
          stability_checker->settings().get<int64_t>("max_subspace");
      bool internal = stability_checker->settings().get<bool>("internal");

      // Create new solver instances
      scf_solver = ScfSolverFactory::create();
      stability_checker = StabilityCheckerFactory::create();

      // Reconfigure with updated settings
      scf_solver->settings().set("scf_type", "unrestricted");
      scf_solver->settings().set("method", method);

      stability_checker->settings().set("internal", internal);
      stability_checker->settings().set("external", false);
      stability_checker->settings().set("method", stab_method);
      stability_checker->settings().set("davidson_tolerance", davidson_tol);
      stability_checker->settings().set("stability_tolerance", stability_tol);
      stability_checker->settings().set("max_subspace", max_sub);
    }

    // Re-run SCF with rotated orbitals as initial guess
    auto [new_energy, new_wavefunction] =
        scf_solver->run(n2, 0, spin_multiplicity, rotated_orbitals);
    energy = new_energy;
    wavefunction = new_wavefunction;
    std::cout << "New Energy after rotation: " << std::fixed
              << std::setprecision(10) << energy << " Hartree" << std::endl;
    std::cout << std::endl;
  }

  std::cout << "\nFinal Energy: " << std::fixed << std::setprecision(10)
            << energy << " Hartree" << std::endl;
  std::cout << "Final stability status: " << (is_stable ? "stable" : "unstable")
            << std::endl;
  // end-cell-run
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-list-implementations
  auto available = StabilityCheckerFactory::available();
  std::cout << "\nAvailable stability checker implementations: ";
  for (const auto& name : available) {
    std::cout << name << " ";
  }
  std::cout << std::endl;
  // end-cell-list-implementations
  // --------------------------------------------------------------------------------------------

  return 0;
}
