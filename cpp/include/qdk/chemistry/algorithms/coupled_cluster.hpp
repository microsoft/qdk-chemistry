// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <iostream>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/coupled_cluster.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @class CoupledClusterCalculator
 * @brief Abstract base class for coupled-cluster calculations in quantum
 * chemistry
 *
 * This class provides the interface for coupled-cluster-based quantum
 * chemistry calculations. It serves as a base for various coupled-cluster
 * methods, such as CCSD, CCSD(T), and other single-reference
 * coupled-cluster algorithms.
 *
 * The calculator takes a Hamiltonian as input and returns both the calculated
 * (total) energy and the corresponding coupled-cluster amplitudes and
 * orbitals.
 *
 * @see data::Hamiltonian
 * @see data::CoupledClusterAmplitudes
 * @see data::Settings
 */
class CoupledClusterCalculator
    : public Algorithm<
          CoupledClusterCalculator,
          std::pair<double, std::shared_ptr<data::CoupledClusterAmplitudes>>,
          std::shared_ptr<data::Ansatz>> {
 public:
  /**
   * @brief Default constructor
   *
   * Creates a coupled-cluster calculator with default settings.
   */
  CoupledClusterCalculator() = default;

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of derived classes.
   */
  virtual ~CoupledClusterCalculator() = default;

  /**
   * @brief Perform coupled-cluster calculation
   *
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param ansatz The Ansatz (Wavefunction and Hamiltonian) describing the
   * quantum system
   * \endcond
   * @return A pair containing the calculated total energy (first) and the
   * resulting coupled-cluster amplitudes (second)
   *
   * @throw std::runtime_error if the calculation fails
   * @throw std::invalid_argument if the Ansatz is invalid or electron counts
   * are invalid
   * @throws SettingsAreLocked if attempting to modify settings after
   * calculate() is called
   *
   * @note Settings are automatically locked when this method is called and
   * cannot be modified during or after execution.
   *
   * @see data::Ansatz
   * @see data::CoupledClusterAmplitudes
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
  std::string type_name() const final { return "coupled_cluster_calculator"; };

 protected:
  /**
   * @brief Implementation of coupled-cluster calculation
   *
   * This method contains the actual calculation logic. It is automatically
   * called by run()/calculate() after settings have been locked.
   *
   * @param ansatz The Ansatz describing the quantum system
   * @return A pair containing the energy and coupled-cluster amplitudes
   */
  virtual std::pair<double, std::shared_ptr<data::CoupledClusterAmplitudes>>
  _run_impl(std::shared_ptr<data::Ansatz> ansatz) const = 0;
};

/**
 * @brief Factory class for creating coupled-cluster calculator instances.
 *
 * The CoupledClusterCalculatorFactory implements the Factory design pattern
 * to dynamically create and manage different implementations of
 * coupled-cluster calculators. This allows the library to support multiple CC
 * calculation methods while providing a unified interface for users.
 *
 * The factory maintains a registry of calculator implementations identified
 * by string keys. New implementations can be registered at runtime using the
 * register_builder method, and instances can be created using the create
 * method.
 *
 * Typical usage:
 * ```
 * // Register a custom implementation
 * CoupledClusterCalculatorFactory::register_instance("my_method", []() {
 *     return std::make_unique<MyCCCalculator>();
 * });
 *
 * // Create an instance
 * auto calculator = CoupledClusterCalculatorFactory::create("my_method");
 *
 * // Get available implementations
 * auto available = CoupledClusterCalculatorFactory::available();
 * ```
 *
 * @see CoupledClusterCalculator for the interface implemented by concrete
 * calculators
 */
struct CoupledClusterCalculatorFactory
    : public AlgorithmFactory<CoupledClusterCalculator,
                              CoupledClusterCalculatorFactory> {
  static std::string algorithm_type_name() {
    return "coupled_cluster_calculator";
  }
  static void register_default_instances() {};
  static std::string default_algorithm_name() { return "pyscf"; }
};

}  // namespace qdk::chemistry::algorithms
