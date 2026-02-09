// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <iostream>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @brief Abstract base class for multi-configurational Self-Consistent Field
 * (MCSCF) calculations
 *
 * The MultiConfigurationScf class provides an interface for
 * multi-configurational Self-Consistent Field methods that simultaneously
 * optimize both the molecular orbital coefficients and the configuration
 * interaction coefficients.
 *
 * The solver takes a Hamiltonian as input and produces the optimized energy and
 * multi-configurational wavefunction.
 *
 * @see data::Hamiltonian
 * @see data::Wavefunction
 * @see data::Settings
 */
class MultiConfigurationScf
    : public Algorithm<MultiConfigurationScf,
                       std::pair<double, std::shared_ptr<data::Wavefunction>>,
                       std::shared_ptr<data::Orbitals>,
                       std::shared_ptr<HamiltonianConstructor>,
                       std::shared_ptr<MultiConfigurationCalculator>,
                       unsigned int, unsigned int> {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an MultiConfigurationScf solver with default settings.
   */
  MultiConfigurationScf() = default;

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of derived classes.
   */
  virtual ~MultiConfigurationScf() = default;

  /**
   * @brief Perform an MultiConfigurationScf calculation
   *
   * This pure virtual method must be implemented by derived classes to perform
   * the MultiConfigurationScf calculation, which optimizes both orbital and CI
   * coefficients. The method takes a Hamiltonian describing the quantum system
   * and returns both the calculated energy and the optimized
   * multi-configurational wavefunction, including both optimized orbital
   * coefficients and CI coefficients.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param orbitals The initial molecular orbitals for the calculation
   * @param hamiltonian_ctor The Hamiltonian constructor for building the
   * Hamiltonian
   * @param mc_calculator The multi-configurational calculator for evaluation
   * of the active space
   * @param n_active_alpha_electrons The number of alpha electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @param n_active_beta_electrons The number of beta electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * \endcond
   *
   * @return A pair containing the calculated energy (first) and the resulting
   *         multi-configurational wavefunction (second)
   *
   * @throw std::runtime_error if calculation fails to converge
   * @throw std::invalid_argument if hamiltonian is invalid
   *
   * @note This method is const as it should not modify the solver's state
   * @see data::Hamiltonian
   * @see data::Wavefunction
   * @note The specific requirements for the inputs and settings affecting
   * this method may vary by implementation. See the documentation for
   * the specific MultiConfigurationScf solver being used (docstring).
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
  std::string type_name() const final { return "multi_configuration_scf"; };

 protected:
  /**
   * @brief Perform an MCSCF calculation
   *
   * This pure virtual method must be implemented by derived classes to perform
   * the MCSCF calculation, which optimizes both orbital and CI coefficients.
   * The method takes initial orbitals, a Hamiltonian constructor and a
   * multi-configurational calculator, then returns both the calculated energy
   * and the optimized multi-configurational wavefunction, including both
   * optimized orbital coefficients and CI coefficients.
   *
   * @param orbitals The initial molecular orbitals for the calculation
   * @param hamiltonian_ctor The Hamiltonian constructor for building the
   * Hamiltonian
   * @param mc_calculator The multi-configurational calculator for evaluation
   * of the active space
   * @param n_active_alpha_electrons The number of alpha electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @param n_active_beta_electrons The number of beta electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   *
   * @return A pair containing the calculated energy (first) and the resulting
   *         multi-configurational wavefunction (second)
   */
  virtual std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals,
      std::shared_ptr<HamiltonianConstructor> hamiltonian_ctor,
      std::shared_ptr<MultiConfigurationCalculator> mc_calculator,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const = 0;
};

/**
 * @brief Factory class for creating MultiConfigurationScf solver instances.
 *
 * The MultiConfigurationScfFactory implements the Factory design pattern to
 * dynamically create and manage different implementations of
 * MultiConfigurationScf solvers. This allows the library to support multiple
 * MultiConfigurationScf methods while providing a unified interface for
 * clients.
 *
 * The factory maintains a registry of solver implementations identified by
 * string keys. New implementations can be registered at runtime using the
 * register_builder method, and instances can be created using the create
 * method.
 *
 * Typical usage:
 * ```
 * // Register a custom implementation
 * MultiConfigurationScfFactory::register_instance("my_mcscf", []() {
 *     return std::make_unique<MyMultiConfigurationScf>();
 * });
 *
 * // Create an instance
 * auto mcscf = MultiConfigurationScfFactory::create("my_mcscf");
 * ```
 *
 * @see MultiConfigurationScf for the interface implemented by concrete
 * solvers
 */
struct MultiConfigurationScfFactory
    : public AlgorithmFactory<MultiConfigurationScf,
                              MultiConfigurationScfFactory> {
  static std::string algorithm_type_name() { return "multi_configuration_scf"; }
  static void register_default_instances() {};
  static std::string default_algorithm_name() { return "pyscf"; }
};

}  // namespace qdk::chemistry::algorithms
