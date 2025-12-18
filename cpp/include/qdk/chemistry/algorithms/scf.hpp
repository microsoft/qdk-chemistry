// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <iostream>
#include <optional>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @class ElectronicStructureSettings
 * @brief Base class for electronic structure algorithms
 *
 * Contains common settings for algorithms that work with electronic structure
 * like basis sets, molecular charge, spin multiplicity.
 */
class ElectronicStructureSettings : public data::Settings {
 public:
  ElectronicStructureSettings() {
    set_default("method", "hf");
    set_default("convergence_threshold", 1e-7);
    set_default("max_iterations", 50, "Maximum number of SCF iterations",
                qdk::chemistry::data::BoundConstraint<int64_t>{
                    1, std::numeric_limits<int64_t>::max()});
    set_default("scf_type", "auto");
  }
};

/**
 * @brief Type alias for basis set specification or initial guess
 *
 * Can be one of:
 * - A shared pointer to Orbitals for initial guess
 * - A shared pointer to BasisSet for custom basis
 * - A string for standard basis set name (e.g., "sto-3g")
 */
using BasisOrGuessType =
    std::variant<std::shared_ptr<data::Orbitals>,
                 std::shared_ptr<data::BasisSet>, std::string>;

/**
 * @brief Abstract base class for Self-Consistent Field (SCF) solvers
 *
 * The ScfSolver class provides a common interface for various SCF algorithms
 * used in quantum chemistry calculations. This class uses the Factory design
 * pattern to enable the dynamic creation of different SCF solver
 * implementations.
 *
 * The solver takes a molecular structure as input and produces the converged
 * total energy (electronic + nuclear repulsion) and corresponding molecular
 * orbitals and single determinant wavefunction.
 *
 * Example usage:
 * @code
 * // Create a concrete SCF solver (e.g., HartreeFockSolver)
 * auto scf_solver = qdk::chemistry::ScfSolver::create("default");
 *
 * // Configure solver settings
 * scf_solver->settings().set("max_iterations", 100);
 *
 * // Perform SCF calculation
 * auto [energy, wavefunction] = scf_solver->run(molecular_structure, 0, 1,
 * "cc-pvdz");
 *
 * std::cout << "SCF Total Energy: " << energy << " Hartree" << std::endl;
 * @endcode
 */
class ScfSolver
    : public Algorithm<
          ScfSolver, std::pair<double, std::shared_ptr<data::Wavefunction>>,
          std::shared_ptr<data::Structure>, int, int, BasisOrGuessType> {
 public:
  /**
   * @brief Default constructor
   */
  ScfSolver() { _settings = std::make_unique<ElectronicStructureSettings>(); };

  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~ScfSolver() = default;

  /**
   * @brief Solve the SCF equations for a given molecular structure
   *
   * This method performs the iterative SCF procedure to find the
   * self-consistent molecular orbitals and total energy.
   *
   * The specific algorithm (HF, DFT, etc.) is determined by the
   * concrete implementation.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param structure The molecular structure containing atomic positions,
   *                  nuclear charges, and other structural information
   * @param charge The molecular charge
   * @param spin_multiplicity The spin multiplicity of the molecular system
   * @param basis_or_guess Basis set information, which can be provided as:
   *         - A shared pointer to a `data::BasisSet` object
   *         - A string specifying the name of a standard basis set (e.g.,
   * "sto-3g")
   *         - A shared pointer to a `data::Orbitals` object to be used as an
   * initial guess
   * \endcond
   *
   * @return A pair containing:
   *         - double: The converged total energy in Hartree
   *         - data::Wavefunction: The converged wavefunction, including
   *           orbitals, their energies, coefficients, and occupancies
   *
   * @throws std::runtime_error If SCF fails to converge
   * @throws std::invalid_argument If the input structure is invalid
   * @throws qdk::chemistry::data::SettingsAreLocked If attempting to modify
   * settings after run() is called
   *
   * @note Settings are automatically locked when this method is called and
   *       cannot be modified during or after execution.
   * @note The specific requirements for the inputs and settings affecting
   *       the SCF calculation may vary by implementation. See the
   * documentation for the specific SCF solver being used.
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name
   *
   * @return The algorithm's type name
   */
  std::string type_name() const final { return "scf_solver"; };

 protected:
  /**
   * @brief Implementation of the SCF calculation
   *
   * This method contains the actual SCF algorithm logic. It is automatically
   * called by run() with identical arguments after settings have been locked.
   *
   * @param structure The molecular structure
   * @param charge The molecular charge
   * @param spin_multiplicity The spin multiplicity
   * @param basis_or_guess Basis set information, which can be provided as:
   *         - A shared pointer to a `data::BasisSet` object
   *         - A string specifying the name of a standard basis set (e.g.,
   * "sto-3g")
   *         - A shared pointer to a `data::Orbitals` object to be used as an
   * initial guess (the basis set will be inferred from the orbitals)
   * @return A pair containing the energy and wavefunction
   */
  virtual std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity, BasisOrGuessType basis_or_guess) const = 0;
};

/**
 * @brief Factory class for creating SCF solvers.
 */
struct ScfSolverFactory : public AlgorithmFactory<ScfSolver, ScfSolverFactory> {
  static std::string algorithm_type_name() { return "scf_solver"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk"; }
};

}  // namespace qdk::chemistry::algorithms
