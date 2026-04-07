// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <iostream>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @class HamiltonianConstructor
 * @brief Abstract base class for constructing Hamiltonian operators
 *
 * This class provides the interface for constructing Hamiltonian operators
 * from orbital data in quantum chemistry calculations. It serves as a base
 * for various Hamiltonian construction methods.
 *
 * The constructor takes orbital information as input and produces a complete
 * Hamiltonian operator that can be used in subsequent quantum chemistry
 * calculations.
 *
 * @see data::Hamiltonian
 * @see data::Orbitals
 * @see data::Settings
 */
class HamiltonianConstructor
    : public Algorithm<HamiltonianConstructor,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Orbitals>> {
 public:
  /**
   * @brief Default constructor
   *
   * Creates a Hamiltonian constructor with default settings.
   */
  HamiltonianConstructor() = default;

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of derived classes.
   */
  virtual ~HamiltonianConstructor() = default;

  /**
   * @brief Construct Hamiltonian operator from orbital data
   *
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param orbitals The orbital data from which to construct the Hamiltonian
   * \endcond
   * @return The constructed Hamiltonian operator ready for use in quantum
   *         chemistry calculations
   *
   * @throw std::runtime_error if Hamiltonian construction fails
   * @throw std::invalid_argument if orbital data is incomplete or invalid
   * @throws qdk::chemistry::data::SettingsAreLocked if attempting to modify
   * settings after run) is called
   *
   * @note Settings are automatically locked when this method is called and
   * cannot be modified during or after execution.
   *
   * @see data::Orbitals
   * @see data::Hamiltonian
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
  std::string type_name() const final { return "hamiltonian_constructor"; };

 protected:
  /**
   * @brief Implementation of Hamiltonian construction
   *
   * This method contains the actual construction logic. It is automatically
   * called by run() after settings have been locked.
   *
   * @param orbitals The orbital data from which to construct the Hamiltonian
   * @return The constructed Hamiltonian operator
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals) const = 0;
};

/**
 * @brief Factory class for creating Hamiltonian constructor instances.
 *
 * The HamiltonianConstructorFactory implements the Factory design pattern to
 * dynamically create and manage different implementations of Hamiltonian
 * constructors. This allows the library to support multiple Hamiltonian
 * construction methods while providing a unified interface for clients.
 *
 * The factory maintains a registry of constructor implementations identified
 * by string keys. New implementations can be registered at runtime using the
 * register_instance method, and instances can be created using the create
 * method.
 *
 * Typical usage:
 * ```
 * // Register a custom implementation
 * HamiltonianConstructorFactory::register_instance("my_method", []() {
 *     return std::make_unique<MyHamiltonianConstructor>();
 * });
 *
 * // Create an instance
 * auto constructor = HamiltonianConstructorFactory::create("my_method");
 *
 * // Get available implementations
 * auto available = HamiltonianConstructorFactory::available();
 * ```
 *
 * @see HamiltonianConstructor for the interface implemented by concrete
 * constructors
 */
struct HamiltonianConstructorFactory
    : public AlgorithmFactory<HamiltonianConstructor,
                              HamiltonianConstructorFactory> {
  static std::string algorithm_type_name() { return "hamiltonian_constructor"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk"; }
};

}  // namespace qdk::chemistry::algorithms
