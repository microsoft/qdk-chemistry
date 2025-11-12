// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/scf.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class ScfSettings
 * @brief Settings container for the internal SCF solver implementation
 *
 * This class extends the ElectronicStructureSettings class to provide specific
 * default settings for the internal SCF solver. It pre-configures common SCF
 * calculation parameters with sensible default values.
 *
 * Default settings include:
 * - Inherits electronic structure parameters from ElectronicStructureSettings.
 * - method: "hf" - The default method (Hartree-Fock) for SCF calculations.
 *
 * Users can override these defaults by modifying the settings object
 * obtained from the ScfSolver instance.
 *
 * Example:
 * ```cpp
 * auto solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
 * auto& settings = solver->settings();
 * settings.set("basis_set", "6-31G*");  // Override the default basis set
 * settings.set("max_iterations", 100);   // Add additional settings
 * ```
 *
 * @see qdk::chemistry::algorithms::ElectronicStructureSettings
 * @see qdk::chemistry::algorithms::microsoft::ScfSolver
 */
class ScfSettings
    : public qdk::chemistry::algorithms::ElectronicStructureSettings {
 public:
  /**
   * @brief Constructor that initializes default SCF settings
   *
   * Creates an SCF settings object with the following defaults:
   * - Inherits from ElectronicStructureSettings: basis_set, charge,
   * spin_multiplicity, etc.
   * - Additional SCF-specific convergence and algorithm parameters
   *
   */
  ScfSettings() : qdk::chemistry::algorithms::ElectronicStructureSettings() {
    // TODO enable these
    // set_default("diis_enabled", true);
    // set_default("diis_start", 2);
    // set_default("level_shift", 0.0);
  }
};

/**
 * @class ScfSolver
 * @brief Internal implementation of the SCF solver
 *
 * This class provides a concrete implementation of the SCF (Self-Consistent
 * Field) solver using the internal backend.
 * It inherits from the base `ScfSolver` class and implements the
 * `solve` method to perform self-consistent field calculations on molecular
 * structures.
 *
 * Typical usage:
 * ```cpp
 * // Create a molecular structure
 * qdk::chemistry::data::Structure water({
 *   {0.0, 0.0, 0.0},
 *   {0.0, 0.76, 0.59},
 *   {0.0, -0.76, 0.59}
 * }, {
 *   qdk::chemistry::data::Element::O,
 *   qdk::chemistry::data::Element::H,
 *   qdk::chemistry::data::Element::H
 * });
 *
 * // Create an SCF solver instance
 * auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
 *
 * // Configure settings if needed
 * scf_solver->settings().set("basis_set", "sto-3g");
 *
 * // Perform SCF calculation
 * auto [energy, orbitals] = scf_solver->run(water, 0, 1);
 * ```
 *
 * @see qdk::chemistry::algorithms::ScfSolver
 * @see qdk::chemistry::data::Structure
 * @see qdk::chemistry::data::Orbitals
 * @see qdk::chemistry::algorithms::ScfSolverFactory
 */
class ScfSolver : public qdk::chemistry::algorithms::ScfSolver {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes an SCF solver with default settings.
   */
  ScfSolver() { _settings = std::make_unique<ScfSettings>(); };

  /**
   * @brief Virtual destructor
   */
  ~ScfSolver() = default;

  virtual std::string name() const final { return "qdk"; }

 protected:
  /**
   * @brief Perform an SCF calculation on the given molecular structure
   *
   * This method performs a self-consistent field calculation using the
   * internal implementation. It configures
   * the internal SCF engine according to the current settings, runs the
   * calculation, and converts the results back into QDK data structures.
   *
   * @param structure The molecular structure containing atomic positions and
   * elements
   * @param charge The molecular charge
   * @param spin_multiplicity The spin multiplicity of the molecular system
   * @param initial_guess Initial orbital guess for the SCF calculation
   * (optional, defaults to a standard configurable guess)
   * @return A pair containing:
   *         - The final SCF total energy (double)
   *         - The converged wavefunction (data::Wavefunction)
   *
   * @throws std::runtime_error If the SCF calculation fails to converge
   *
   * @see qdk::chemistry::data::Structure
   * @see qdk::chemistry::data::Wavefunction
   */
  std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity,
      std::optional<std::shared_ptr<data::Orbitals>> initial_guess)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
