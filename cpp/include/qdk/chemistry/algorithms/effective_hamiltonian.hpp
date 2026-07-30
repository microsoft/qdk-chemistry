// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class EffectiveHamiltonianConstructor
 * @brief Abstract base for downfolding an active-space effective Hamiltonian.
 *
 * Given a reference `Wavefunction` (whose active space is the kept subspace
 * P and whose occupations/RDMs define the reference) and an input `Hamiltonian`
 * built over the whole downfolding window W = P u Q, a concrete constructor
 * folds the external space Q into an effective Hamiltonian acting on P. This is
 * a distinct algorithm type from `HamiltonianConstructor` (which builds the
 * bare integral Hamiltonian from `Orbitals`).
 *
 * The input `Hamiltonian` must be built with its active space set to the whole
 * window W (every orbital to be folded is "active" to the integral
 * constructor), otherwise the P<->Q couplings are already gone.
 *
 * @see data::Hamiltonian
 * @see data::Wavefunction
 */
class EffectiveHamiltonianConstructor
    : public Algorithm<EffectiveHamiltonianConstructor,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Wavefunction>,
                       std::shared_ptr<data::Hamiltonian>> {
 public:
  EffectiveHamiltonianConstructor() = default;
  virtual ~EffectiveHamiltonianConstructor() = default;

  using Algorithm::run;

  /**
   * @brief Access the algorithm's name.
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name.
   */
  std::string type_name() const final {
    return "effective_hamiltonian_constructor";
  }

 protected:
  /**
   * @brief Fold the window Hamiltonian onto the reference active space.
   *
   * @param reference Reference wavefunction; its active space is the kept
   *        subspace P and its occupations define the reference.
   * @param hamiltonian Input Hamiltonian built over the whole window W = P u Q.
   * @return The effective Hamiltonian acting on the active space P.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian) const = 0;
};

/**
 * @brief Factory for creating effective-Hamiltonian constructor instances.
 *
 * @see EffectiveHamiltonianConstructor
 */
struct EffectiveHamiltonianConstructorFactory
    : public AlgorithmFactory<EffectiveHamiltonianConstructor,
                              EffectiveHamiltonianConstructorFactory> {
  static std::string algorithm_type_name() {
    return "effective_hamiltonian_constructor";
  }
  // Concrete variants (swpt2, ...) register here (effective_hamiltonian.cpp).
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_swpt2"; }
};

}  // namespace qdk::chemistry::algorithms
