/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @class DynamicalCorrelationCalculator
 * @brief Base class for reference-derived quantum chemistry methods
 *
 * This abstract base class provides a unified interface for quantum chemistry
 * methods that provide corrections on top of a reference wavefunction, such as
 * Møller-Plesset perturbation theory (MP2) and Coupled Cluster (CC) methods.
 *
 * @see data::Ansatz
 * @see data::Wavefunction
 * @see MP2Calculator
 */
class DynamicalCorrelationCalculator
    : public Algorithm<DynamicalCorrelationCalculator,
                       std::pair<double, std::shared_ptr<data::Wavefunction>>,
                       std::shared_ptr<data::Ansatz>> {
 public:
  /**
   * @brief Default constructor
   */
  DynamicalCorrelationCalculator() = default;

  /**
   * @brief Virtual destructor
   */
  virtual ~DynamicalCorrelationCalculator() = default;

  /**
   * @brief Run main calculation
   *
   * This method performs the calculation using the provided ansatz and returns
   * both the total energy and the resulting wavefunction.
   *
   * @param args Arguments to pass to the underlying algorithm, typically an
   * Ansatz (Wavefunction and Hamiltonian) describing the system of interest
   * @return A pair containing the total energy (first) and the resulting
   *         wavefunction (second)
   *
   * @throw std::runtime_error if the calculation fails
   * @throw std::invalid_argument if the Ansatz is invalid
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
  std::string type_name() const override {
    return "dynamical_correlation_calculator";
  }

 protected:
  /**
   * @brief Implementation of the main calculation
   *
   * This method contains the actual calculation logic and must be implemented
   * by derived classes.
   *
   * @param ansatz The Ansatz describing the quantum system
   * @return A pair containing the total energy and resulting wavefunction
   */
  virtual std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Ansatz> ansatz) const = 0;
};

/**
 * @brief Factory class for creating reference-derived calculator instances.
 */
struct DynamicalCorrelationCalculatorFactory
    : public AlgorithmFactory<DynamicalCorrelationCalculator,
                              DynamicalCorrelationCalculatorFactory> {
  static std::string algorithm_type_name() {
    return "dynamical_correlation_calculator";
  }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_mp2_calculator"; }
};

}  // namespace qdk::chemistry::algorithms
