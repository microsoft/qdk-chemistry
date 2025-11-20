// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <iostream>
#include <qdk/chemistry/algorithms/reference_derived_calculator.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft {

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
 * orbitals stored in a Wavefunction object with a CoupledClusterContainer.
 *
 * @see data::Hamiltonian
 * @see data::Wavefunction
 * @see data::CoupledClusterContainer
 * @see data::Settings
 */
class CoupledClusterCalculator : public ReferenceDerivedCalculator {
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
   * @param ansatz The Ansatz (Wavefunction and Hamiltonian) describing the
   * quantum system
   * @return A pair containing the total energy (first) and the resulting
   * wavefunction (second)
   *
   * @throw std::runtime_error if the calculation fails
   * @throw std::invalid_argument if the Ansatz is invalid or electron counts
   * are invalid
   * @throws SettingsAreLocked if attempting to modify settings after
   * run() is called
   *
   * @note Settings are automatically locked when this method is called and
   * cannot be modified during or after execution.
   *
   * @see data::Ansatz
   * @see data::Wavefunction
   */
  using ReferenceDerivedCalculator::run;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Get the algorithm's type name
   *
   * @return The algorithm's type name
   */
  std::string type_name() const override {
    return "reference_derived_calculator";
  }

 protected:
  /**
   * @brief Implementation of coupled-cluster calculation
   *
   * This method contains the actual calculation logic and returns the
   * total energy and resulting wavefunction.
   *
   * @param ansatz The Ansatz describing the quantum system
   * @return A pair containing the total energy and resulting wavefunction
   */
  std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Ansatz> ansatz) const override = 0;
};

}  // namespace qdk::chemistry::algorithms::microsoft
